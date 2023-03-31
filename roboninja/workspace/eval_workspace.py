import os
import pathlib
import sys

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import pickle
from multiprocessing.managers import SharedMemoryManager

import cv2
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

import wandb
from roboninja.cutting_policy import (CuttingPolicyAdaptive, CuttingPolicyBase,
                                      CuttingPolicyGreedy, CuttingPolicyNN,
                                      CuttingPolicyNonAdaptive)
from roboninja.env import BaseEnv, RealEnv, SimEnv, TCEnv
from roboninja.model.close_loop_policy_model import CloseLoopPolicyModel
from roboninja.model.state_estimation_model import StateEstimationModel
from roboninja.real_world.multi_realsense import MultiRealsense
from roboninja.real_world.recorder import RealSenseRecorder
from roboninja.real_world.single_realsense import SingleRealsense
from roboninja.real_world.ur5 import RTDEInterpolationController
from roboninja.real_world.video_recorder import VideoRecorder
from roboninja.utils.dynamics import forward_dynamics, get_init_pos
from roboninja.utils.misc import html_visualize, mkdir
from roboninja.utils.visualizer import Visualizer
from roboninja.workspace.base_workspace import BaseWorkspace
from roboninja.workspace.close_loop_policy_workspace import \
    CloseLoopPolicyWorkspace
from roboninja.workspace.state_estimation_workspace import \
    StateEstimationWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

class EvalWorkspace(BaseWorkspace):
    include_keys = list()
    exclude_keys = list()

    def __init__(self, cfg: OmegaConf):
        # replace all macros with concrete value
        OmegaConf.resolve(cfg)
        super().__init__(cfg)
        cv2.setNumThreads(1)
        threadpool_limits(1)

        self.state_estimation_model = StateEstimationModel(**self.cfg.state_estimation_model)
        self.close_loop_policy_model = CloseLoopPolicyModel(**self.cfg.close_loop_policy_model)

    @classmethod
    def create_from_checkpoints(cls, cfg):
        state_estimation_workspace = StateEstimationWorkspace.create_from_checkpoint(cfg.state_estimation_path, map_location=torch.device('cpu'))
        close_loop_policy_workspace = CloseLoopPolicyWorkspace.create_from_checkpoint(cfg.close_loop_policy_path, map_location=torch.device('cpu'))

        cls_cfg = cfg
        cls_cfg.state_estimation_model = state_estimation_workspace.cfg.model
        cls_cfg.close_loop_policy_model = close_loop_policy_workspace.cfg.model

        obj = cls(cls_cfg)
        obj.state_estimation_model = state_estimation_workspace.model
        obj.close_loop_policy_model = close_loop_policy_workspace.model

        return obj


    @torch.no_grad()
    def eval_run(self,
        eval_env:BaseEnv,
        cutting_policy:CuttingPolicyBase,
        bone_idx:int,
        device:torch.device,
    ):
        visualizer = Visualizer()

        self.cfg.cut_env.bone.name = f'bone_{bone_idx}'
        img_bone, bone_wrd_pts = visualizer.vis_bone(self.cfg.cut_env.bone, color=0, return_wrd=True)

        # get init pos
        init_action_p = np.array(self.cfg.init_action_p)
        reset_pos = get_init_pos(init_action_p, self.cfg.cut_env.knife)

        eval_env.reset(
            bone_wrd_pts=bone_wrd_pts,              # sim
            cut_env_cfg=self.cfg.cut_env,           # taichi
            init_action_p=init_action_p,            # taichi
            reset_pos=reset_pos,                    # real
            output_dir=self.output_dir,             # real
            cutting_policy=cutting_policy
        )

        cutting_policy.reset(bone_idx=bone_idx)

        current_pos = reset_pos

        img_pts = visualizer.empty_image(num_dim=2)
        img_pts_vis = 1 - visualizer.empty_image(num_dim=2)
        exec_info = list()
        pos_history = np.zeros([self.cfg.seq_len * 2, 3])
        pos_history[0] = current_pos.copy()
        step_idx, n_collision = 0, 0
        collision_pos_history = list()
        exec_vis = list()
        success = False

        while True:
            if not isinstance(eval_env, RealEnv) and current_pos[1] < eval_env.min_height:
                success = True
                break

            # state estimation
            logits = self.state_estimation_model(torch.from_numpy(img_pts[None, None]).to(dtype=torch.float32, device=device))[0, 0]
            
            state_estimaiton = (torch.sigmoid(logits) > self.cfg.state_estimation_threshold).cpu().numpy().astype(np.float32)
            cutting_policy.update_state_estimation(state_estimaiton)
            cutting_policy.update_collision_map(img_pts)
            
            if isinstance(eval_env, RealEnv):
                eval_env.update_info(
                    step_idx=step_idx,
                    tolerance=copy.deepcopy(cutting_policy.get_tolerance()),
                    img_bone=img_bone,
                    current_pos=current_pos,
                    state_estimaiton=state_estimaiton,
                    img_pts=img_pts
                )
            else:
                # visualization
                img_vis = visualizer.vis_knife(current_pos, state_estimaiton, color=0.5)
                if isinstance(cutting_policy, CuttingPolicyAdaptive):
                    cv2.putText(
                        img_vis,
                        str(cutting_policy.get_tolerance(step_idx))[:8],
                        (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2, color=0
                    )
                img_vis = np.repeat(img_vis[..., None].copy(), 3, axis=-1)
                img_vis *= 1 - img_pts_vis[..., None]
                img_vis[..., 0] += img_pts_vis
                img_vis = img_vis * 0.8 + img_bone[..., None] * 0.2
                for c in collision_pos_history:
                    img_vis = visualizer.vis_pos(c, img_vis, color=(0.8, 0.2, 0.2))
                img_vis = visualizer.vis_pos(pos_history[:step_idx+1], img_vis, color=(0, 0, 0.8))
                img_vis = (img_vis * 255).astype(np.uint8)
                exec_vis.append(img_vis)

            action = cutting_policy.get_action(step_idx, current_pos)

            if isinstance(eval_env, RealEnv) and current_pos[1] < eval_env.min_height:
                action[0] = np.clip(action[0], -np.pi/4, np.pi/4)
                action[1] = np.clip(action[1], -np.pi/4, np.pi/4)
            pre_pose = current_pos.copy()
            current_pos = forward_dynamics(action, current_pos, self.cfg.cut_env.knife)
            
            pos_history[step_idx + 1] = current_pos.copy()
            
            stop_signal = eval_env.move(
                wrd_pos=current_pos,
                pre_pos=pre_pose,
                step_idx=step_idx,
                sensor=True
            )

            exec_info.append({
                'step_idx': step_idx,
                'action': action,
                'current_pos': pre_pose,
                'next_pos': current_pos,
                'stop_signal': stop_signal
            })

            if stop_signal:
                if current_pos[1] < self.cfg.terminate_height:
                    step_idx = eval_env.roll_back(pos_history, 3, step_idx)
                    success = True
                    break

                n_collision += 1

                delta = np.zeros([2])
                p_pix = visualizer.wrd2pix(current_pos[:2] + delta)
                img_pts_vis = cv2.circle(img_pts_vis, (p_pix[0], p_pix[1]), radius=3, color=1, thickness=-1)
                img_pts[p_pix[1], p_pix[0]] = 0
                
                collision_pos_history.append(pos_history[max(0, step_idx-self.cfg.n_back+1):step_idx+1].copy())
                step_idx = eval_env.roll_back(pos_history, self.cfg.n_back, step_idx)
                current_pos = pos_history[step_idx]
                cutting_policy.collision(step_idx)
            else:
                step_idx += 1

            if n_collision > self.cfg.max_collision:
                break
        
        eval_env.terminate(success=success)
        
        return_data = {
            'exec_vis': exec_vis,
            'img_bone': img_bone,
            'cut_mass': eval_env.cut_mass,
            'cut_mass_array': eval_env.cut_mass_array,
            'num_collision': eval_env.num_collision,
            'collision_array': eval_env.collision_array,
            'energy': eval_env.energy,
            'exec_info': exec_info,
            'render_info_list': eval_env.render_info_list
        }
        if isinstance(eval_env, TCEnv) and self.cfg.taichi_render:
            return_data['render_vis'] = eval_env.render_images

        return return_data

    def run(self):
        # solve ood
        if self.cfg.ood:
            self.cfg.cut_env.bone.mesh_root += '_ood'
            self.cfg.bone_idx_start = 0
            self.cfg.bone_idx_end = 50
        # set device
        self.cfg.cut_env.cuda_id = self.cfg.gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cfg.gpu)
        device = torch.device(f'cuda:0')
        if self.cfg.type != 'real':
            self.state_estimation_model.to(device)
            self.close_loop_policy_model.to(device)
            self.state_estimation_model.eval()
            self.close_loop_policy_model.eval()

        cutting_policy: CuttingPolicyBase
        if self.cfg.policy.cutting_police == 'adaptive':
            cutting_policy = CuttingPolicyAdaptive(
                close_loop_policy_model=self.close_loop_policy_model,
                device=device,
                cfg=self.cfg
            )
        elif self.cfg.policy.cutting_police == 'non_adaptive':
            cutting_policy = CuttingPolicyNonAdaptive(
                close_loop_policy_model=self.close_loop_policy_model,
                device=device,
                cfg=self.cfg
            )
        elif self.cfg.policy.cutting_police == 'greedy':
            cutting_policy = CuttingPolicyGreedy(knife_cfg=self.cfg.cut_env.knife)
        elif self.cfg.policy.cutting_police == 'nn':
            cutting_policy = CuttingPolicyNN(
                knife_cfg=self.cfg.cut_env.knife,
                bone_cfg=self.cfg.cut_env.bone.copy(),
                expert_dir=self.cfg.policy.expert_dir,
                num_train=self.cfg.policy.num_train
            )
        else:
            raise NotImplementedError()
        
        if self.cfg.type in ['sim', 'taichi']:
            bone_idx_list = list(range(self.cfg.bone_idx_start, self.cfg.bone_idx_end))
            eval_env = SimEnv() if self.cfg.type == 'sim' else \
                TCEnv(render=self.cfg.taichi_render)

            self.storage_dir = os.path.join(self.output_dir, 'storage')
            mkdir(self.storage_dir, False)

            if self.cfg.wandb:
                wandb_run = wandb.init(
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                    **self.cfg.logging
                )
            if self.cfg.html:
                html_data = dict()
                ids = [str(x) for x in bone_idx_list]
                cols = ['bone_img', 'exec_vis', 'cut_mass', 'num_c']
                if self.cfg.type == 'taichi':
                    cols += ['energy_sum', 'energy_max']
                    if self.cfg.taichi_render:
                        cols = cols + ['render_vis']
            for bone_idx in tqdm.tqdm(bone_idx_list):
                return_data = self.eval_run(
                    eval_env=eval_env,
                    cutting_policy=cutting_policy,
                    bone_idx=bone_idx,
                    device=device
                )
                if self.cfg.wandb:
                    wandb_log = {
                        'cut_mass': return_data['cut_mass'],
                        'num_collision': return_data['num_collision'],
                        'bone_img': wandb.Image(return_data['img_bone'], caption=f'bone_img'),
                        'exec_vis': wandb.Video(np.stack(return_data['exec_vis']).transpose([0,3,1,2]), fps=10, caption=f'exec_vis')
                    }
                    if self.cfg.type == 'taichi' and self.cfg.taichi_render:
                        wandb_log['render_vis'] = wandb.Video(np.stack(return_data['render_vis']).transpose([0,3,1,2]), fps=10, caption=f'render_vis')
                    wandb_run.log(wandb_log)
                if self.cfg.html:
                    html_data[f'{bone_idx}_cut_mass'] = str(return_data['cut_mass'])[:6]
                    html_data[f'{bone_idx}_num_c'] = str(return_data['num_collision'])
                    html_data[f'{bone_idx}_bone_img'] = return_data['img_bone']
                    html_data[f'{bone_idx}_exec_vis'] = return_data['exec_vis']
                    if self.cfg.type == 'taichi':
                        html_data[f'{bone_idx}_energy_sum'] = str(np.sum(return_data['energy']['energy_array']))[:6]
                        html_data[f'{bone_idx}_energy_max'] = str(np.max(return_data['energy']['energy_array']))[:6]
                        if self.cfg.taichi_render:
                            html_data[f'{bone_idx}_render_vis'] = return_data['render_vis']
                
                pickle.dump(return_data, open(os.path.join(self.storage_dir, f'{bone_idx}.pkl'), 'wb'))
            if self.cfg.html:
                html_visualize(self.output_dir, html_data, ids, cols, title=self.cfg.name, clean=False)

        elif self.cfg.type == 'real':
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()
            shm_manager = SharedMemoryManager()
            shm_manager.start()
            def transform(data):
                img = data['color'].copy()
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                data['color'] = img
                return data
            
            video_recorder = VideoRecorder.create_h264(
                fps=30, 
                codec='h264_nvenc',
                input_pix_fmt='bgr24', 
                crf=21,
                thread_type='AUTO',
                thread_count=1
            )

            with MultiRealsense(
                serial_numbers=camera_serial_numbers,
                shm_manager=shm_manager,
                resolution=(1920, 1080),
                capture_fps=30,
                put_fps=30,
                record_fps=30,
                enable_color=True,
                enable_depth=False,
                recording_transform=transform,
                video_recorder=video_recorder
            ) as multi_realsense:
                self.state_estimation_model.to(device)
                self.close_loop_policy_model.to(device)
                self.state_estimation_model.eval()
                self.close_loop_policy_model.eval()

                multi_realsense.set_exposure(exposure=[300, 300], gain=[0, 0])
                multi_realsense.set_white_balance(white_balance=[None, None])

                with RTDEInterpolationController(**self.cfg.controller) as controller:
                    eval_env = RealEnv(
                        controller=controller,
                        realsense=multi_realsense,
                        offset=np.array(self.cfg.calibration_offset) + np.array(self.cfg.bone_offset),
                        terminate_height=self.cfg.terminate_height,
                        **self.cfg.real_env
                    )
                    return_data = self.eval_run(
                        eval_env=eval_env,
                        cutting_policy=cutting_policy,
                        bone_idx=self.cfg.bone_idx,
                        device=device
                    )
        else:
            raise NotImplementedError(f'{self.cfg.type} is not supported')


@hydra.main(
    version_base=None,
    config_path='../config', 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = EvalWorkspace.create_from_checkpoints(cfg)
    workspace.run()

if  __name__=='__main__':
    main()
