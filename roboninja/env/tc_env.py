import enum
import os

import numpy as np
from roboninja.env.base_env import BaseEnv
import taichi as ti
import cut_simulation.utils.geom as geom_utils
from cut_simulation.configs.macros import *
from roboninja.utils.dynamics import inverse_dynamics
from roboninja.utils.misc import get_vulkan_offset
from scipy.spatial.transform import Rotation
import copy


def get_cut_env(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.cuda_id)
    os.environ['TI_VISIBLE_DEVICE'] = str(cfg.cuda_id + get_vulkan_offset())
    
    from cut_simulation.engine.taichi_env import TaichiEnv
    taichi_env = TaichiEnv(
        quality=2,
        particle_density=8e6,
        max_steps_local=30 * cfg.gpu_type,
        max_steps_global=30 * cfg.horizon + 1,
        device_memory_GB=10 * cfg.gpu_type,
        horizon=cfg.horizon
    )

    taichi_env.setup_agent(cfg.knife)

    taichi_env.add_static(
        file='chopping_board.obj',
        pos=(0.5, 0.0, 0.5),
        euler=(0.0, 180.0, 0.0),
        scale=(0.5, 0.5, 0.5),
        material=CHOPPINGBOARD,
        has_dynamics=False,
    )

    taichi_env.add_static(
        file='support.obj',
        sdf_res=256,
        pos=(0.57, 0.108, 0.5),
        euler=(90.0, 180.0, 0.0),
        scale=(0.1, 0.1875, 0.216),
        material=SUPPORT,
        has_dynamics=True,
        render_order=None,
    )

    taichi_env.add_static(
        file=f'{cfg.bone.name}.obj',
        material=BONE,
        has_dynamics=True,
        render_order='before',
        normalize=False,
        **cfg.bone
    )

    taichi_env.add_body(
        type='cube',
        lower=(0.4, 0.025, 0.425),
        upper=(0.6, 0.2, 0.575),
        material=MEAT,
        obstacles=[taichi_env.statics[2]]
    )

    taichi_env.setup_boundary(
        type='cube',
        lower=(0.05, 0.025, 0.05),
        upper=(0.95, 0.95, 0.95),
    )

    if cfg.render is not None:
        taichi_env.setup_renderer(**cfg.render)

    taichi_env.setup_loss(weights=cfg.loss_weight
    )
    taichi_env.build()
    
    return taichi_env


class StepType(enum.Enum):
    Forward = 0
    Cancel = 1
    Backward = 2

# simulation environment in taichi. It enables energy computation
class TCEnv(BaseEnv):
    def __init__(self, render):
        super().__init__()
        self.render = render

    def reset(self, cut_env_cfg, init_action_p, reset_pos, cutting_policy, **kwargs):
        ti.reset()
        cut_env_cfg.horizon = 500
        self.cut_env_cfg = cut_env_cfg
        self.knife_cfg = cut_env_cfg.knife
        self.taichi_env = get_cut_env(cut_env_cfg)

        self.bone = self.taichi_env.statics[2]
        self.init_state = self.taichi_env.get_state()['state']
        self.taichi_env.apply_agent_action_p(init_action_p)

        self.idx = 0
        self.cumulated_energy_list = list()
        self.forward_idx_list = list()
        self.idx_type = list()
        self.render_images = [self.taichi_env.render()] if self.render else list()
        self.step_idx_list = list()

        self.cut_mass = 0
        self.cut_mass_array = list()
        self.num_collision = 0
        self.collision_array = list()
        self.render_info_list = list()

        self.forward_pos_list = [reset_pos]
        self.backward_pos_pair_list = list()
        self.cutting_policy = cutting_policy

        # get T_knife_init
        self.knife = self.taichi_env.agent.effectors[0]
        scale = np.array(self.knife.mesh.scale, dtype=DTYPE_NP)
        pos = np.array(self.knife.mesh.pos, dtype=DTYPE_NP)
        quat = geom_utils.xyzw_to_wxyz(Rotation.from_euler('zyx', self.knife.mesh.euler[::-1], degrees=True).as_quat().astype(DTYPE_NP))
        self.T_knife_init = geom_utils.trans_quat_to_T(pos, quat) @ geom_utils.scale_to_T(scale)

        # get T_bone
        bone_static = self.taichi_env.statics[2]
        scale = np.array(bone_static.scale, dtype=DTYPE_NP)
        pos = np.array(bone_static.pos, dtype=DTYPE_NP)
        quat = geom_utils.xyzw_to_wxyz(Rotation.from_euler('zyx', bone_static.euler[::-1], degrees=True).as_quat().astype(DTYPE_NP))
        self.T_bone = geom_utils.trans_quat_to_T(pos, quat) @ geom_utils.scale_to_T(scale)
        self.bone_path = os.path.join(bone_static.mesh_root, 'raw', bone_static.raw_file)

    def get_T_knife(self):
        f = self.taichi_env.simulator.cur_step_local
        T = geom_utils.trans_quat_to_T(self.knife.pos[f].to_numpy(), self.knife.quat[f].to_numpy())
        return T @ self.T_knife_init

    def get_p_particles(self):
        f = self.taichi_env.simulator.cur_step_local
        frame_state = self.taichi_env.simulator.get_frame_state(f)
        return frame_state.x.to_numpy()

    def move(self, wrd_pos, pre_pos, step_idx=None, roll_back=False, horizontal=False, collision_candidates=None, **kwargs) -> bool:
        self.step_idx_list.append(step_idx)
        # cumulate cut mass
        delta_cut_mass = 0.5 * ((wrd_pos[0] - 0.4) + (pre_pos[0] - 0.4)) * \
            (self.clip(pre_pos[1]) - self.clip(wrd_pos[1]))
        self.cut_mass += delta_cut_mass
        self.cut_mass_array.append(delta_cut_mass)

        # use taichi sdf
        particle_sdf = np.array([0.0])
        self.bone.check_collision(1, np.array([[wrd_pos[0], wrd_pos[1], 0.5]]), particle_sdf)
        stop_signal = particle_sdf[0] < 1e-9
        self.collision_array.append(stop_signal)

        # execution & get loss
        action_v = inverse_dynamics(
            current_pos=pre_pos,
            next_pos=wrd_pos,
            knife_cfg=self.knife_cfg
        )
        action_type = self.knife_cfg.effectors[0].params.action_type
        action_scale_v = np.array(self.knife_cfg.effectors[0].params.get(f'action_scale_v_{action_type}'))
        action_v[:2] /= action_scale_v[:2]

        self.taichi_env.step(action_v)

        # add render info
        if roll_back:
            self.forward_pos_list.pop()
            self.backward_pos_pair_list.append([copy.deepcopy(wrd_pos), copy.deepcopy(pre_pos)])
        else:
            if not horizontal:
                self.forward_pos_list.append(copy.deepcopy(wrd_pos))

        self.render_info_list.append(copy.deepcopy({
            'T_bone': self.T_bone,
            'bone_path': self.bone_path,
            'T_knife': self.get_T_knife(),
            'p_particles': self.get_p_particles(),
            'roll_back': roll_back,
            'forward_pos_list': self.forward_pos_list,
            'backward_pos_pair_list': self.backward_pos_pair_list,
            'policy_render_data': self.cutting_policy.get_render_data(),
            'wrd_pos': wrd_pos,
            'pre_pos': pre_pos,
            'stop_signal': stop_signal,
            'step_idx': step_idx,
        }))

        if self.render:
            self.render_images.append(self.taichi_env.render())
        self.taichi_env.loss.clear()
        cumulated_energy = self.taichi_env.get_loss()['work_loss'] * self.taichi_env.simulator.cur_step_global
        self.cumulated_energy_list.append(cumulated_energy)
        if roll_back:
            cancel_step_idx = self.forward_idx_list.pop()
            self.idx_type[cancel_step_idx] = StepType.Cancel
            self.idx_type.append(StepType.Backward)
        else:
            self.forward_idx_list.append(self.idx)
            self.idx_type.append(StepType.Forward)
        self.idx += 1
        
        if collision_candidates is None:
            return stop_signal
        else:
            collision_results = list()
            NCC = len(collision_candidates)
            particle_sdf = np.zeros([NCC])
            collision_candidates = np.asarray(collision_candidates)
            collision_candidates = np.concatenate([collision_candidates, np.ones([NCC, 1]) * 0.5], axis=1)
            self.bone.check_collision(NCC, collision_candidates, particle_sdf)
            collision_results = particle_sdf < 1e-9

            return collision_results

    @property
    def energy(self):
        energy_array = np.array([])
        last_energy = 0
        for cumulated_energy in self.cumulated_energy_list:
            energy_array = np.append(energy_array, cumulated_energy - last_energy)
            last_energy = cumulated_energy
        idx_type = np.array(self.idx_type)
        energy_info = {
            'energy_forward': np.sum(energy_array[idx_type == StepType.Forward]),
            'energy_cancel': np.sum(energy_array[idx_type == StepType.Cancel]),
            'energy_backward': np.sum(energy_array[idx_type == StepType.Backward]),
            'energy_array': energy_array
        }
        return energy_info
