import numpy as np
import taichi as ti
from cut_simulation.engine.simulators import MPMSimulator
from cut_simulation.engine.agents import *
from cut_simulation.engine.meshes import Statics
from cut_simulation.engine.renderer import Renderer
from cut_simulation.engine.bodies import Bodies
from cut_simulation.engine.losses import CutLoss
from cut_simulation.configs.macros import *
from cut_simulation.utils.misc import *


@ti.data_oriented
class TaichiEnv:
    '''
    TaichiEnv wraps all components in a simulation environment.
    '''
    def __init__(
            self,
            dim=3,
            quality=1,
            particle_density=1e6,
            max_steps_local=160,
            device_memory_GB=42,
            max_steps_global=2500,
            horizon=61,
            ckpt_dest='cpu',
            gravity=(0.0, -10.0, 0.0),
        ):
        ti.init(arch=ti.cuda, device_memory_GB=device_memory_GB, packed=True)

        self.particle_density = particle_density
        self.dim = dim
        self.max_steps_local = max_steps_local
        self.max_steps_global = max_steps_global
        self.horizon = horizon
        self.ckpt_dest = ckpt_dest
        
        # env components
        self.agent = None
        self.statics = Statics()
        self.bodies = Bodies(dim=self.dim, particle_density=self.particle_density)
        self.simulator = MPMSimulator(
            dim=self.dim,
            quality=quality,
            horizon=self.horizon,
            max_steps_local=self.max_steps_local,
            max_steps_global=self.max_steps_global,
            gravity=gravity,
            ckpt_dest=ckpt_dest,
        )
        self.renderer = None
        self.loss = None

    def setup_agent(self, agent_cfg):
        self.agent = eval(agent_cfg.type)(
            max_steps_local=self.max_steps_local,
            max_steps_global=self.max_steps_global,
            max_action_steps_global=self.horizon,
            ckpt_dest=self.ckpt_dest,
            **agent_cfg.get('params', {}),
        )
        for effector_cfg in agent_cfg.effectors:
            self.agent.add_effector(
                type=effector_cfg.type,
                params=effector_cfg.params,
                mesh_cfg=effector_cfg.get('mesh', None),
                boundary_cfg=effector_cfg.boundary,
            )

    def setup_renderer(self, **kwargs):
        self.renderer = Renderer(**kwargs)

    def setup_boundary(self, **kwargs):
        self.simulator.setup_boundary(**kwargs)

    def add_static(self, **kwargs):
        self.statics.add_static(**kwargs)

    def add_body(self, **kwargs):
        self.bodies.add_body(**kwargs)

    def setup_loss(self, **kwargs):
        self.loss = CutLoss(
            max_action_steps_global=self.horizon,
            max_steps_global=self.max_steps_global,
            **kwargs
        )

    def build(self):
        # particles
        self.init_particles, self.particles_material, self.particles_used, particles_color, self.particles_rho, self.particles_body_id = self.bodies.get()
        if self.init_particles is not None:
            self.n_particles = len(self.init_particles)
            self.particles_color = ti.Vector.field(4, ti.f32, shape=(len(self.init_particles,)))
            self.particles_color.from_numpy(particles_color.astype(np.float32))
            self.has_particles = True
        else:
            self.n_particles = 0
            self.has_particles = False
            self.particles_color = None

        # build and initialize states of all environment components
        self.simulator.build(self.agent, self.statics, self.init_particles, self.particles_material, self.particles_used, self.particles_rho, self.particles_body_id)

        if self.agent is not None:
            self.agent.build(self.simulator)

        if self.renderer is not None:
            self.renderer.build(self.n_particles)

        if self.loss is not None:
            self.loss.build(
                sim=self.simulator,
                knife=self.agent.effectors[0],
                bone=self.statics[2]
            )

    def reset_grad(self):
        self.simulator.reset_grad()
        self.agent.reset_grad()
        self.loss.reset_grad()

    def enable_grad(self):
        self.simulator.enable_grad()

    def disable_grad(self):
        self.simulator.disable_grad()

    @property
    def grad_enabled(self):
        return self.simulator.grad_enabled


    def render(self, mode='human', iteration=None, t=None, save=False, **kwargs):
        assert self.renderer is not None, 'No renderer available.'

        if self.has_particles > 0:
            frame_state = self.simulator.get_frame_state(self.simulator.cur_step_local)
        else:
            frame_state = None

        if self.loss is not None and iteration == 0:
            tgt_particles = self.loss.tgt_particles_x_f32
        else:
            tgt_particles = None

        img = self.renderer.render_frame(frame_state, self.particles_material, self.particles_color, self.agent, tgt_particles, self.statics, iteration, t, save)

        return img

    def step(self, action=None):
        if action is not None:
            assert self.agent is not None, 'Environment has no agent to execute action.'
            action = np.array(action).astype(DTYPE_NP)
        self.simulator.step(action=action)

    def step_grad(self, action=None):
        if action is not None:
            assert self.agent is not None, 'Environment has no agent to execute action.'
            action = np.array(action).astype(DTYPE_NP)
        self.simulator.step_grad(action=action)

    def get_loss(self):
        assert self.loss is not None
        # return self.loss.get_loss(self.simulator.cur_step_local)
        return self.loss.get_loss(
            step_num=self.simulator.cur_step_global // self.simulator.n_substeps,
            substep_num=self.simulator.cur_step_global
        )

    def get_loss_grad(self):
        assert self.loss is not None
        # return self.loss.get_loss_grad(self.simulator.cur_step_local)
        return self.loss.get_loss_grad(
            step_num=self.simulator.cur_step_global // self.simulator.n_substeps,
            substep_num=self.simulator.cur_step_global
        )

    def get_state(self):
        return {
            'state': self.simulator.get_state(),
            'grad_enabled': self.grad_enabled
        }

    def set_state(self, state, grad_enabled=False):
        self.simulator.cur_step_global = 0
        self.simulator.set_state(0, state)

        if grad_enabled:
            self.enable_grad()
        else:
            self.disable_grad()

        if self.loss:
            self.loss.clear()

    def apply_agent_action_p(self, action_p):
        assert self.agent is not None, 'Environment has no agent to execute action.'
        self.agent.apply_action_p(action_p)

    def apply_agent_action_p_grad(self, action_p):
        assert self.agent is not None, 'Environment has no agent to execute action.'
        self.agent.apply_action_p_grad(action_p)

    def set_camera(self, position=None, lookat=None):
        if position is not None:
            self.renderer.camera.position(*position)
        if lookat is not None:
            self.renderer.camera.lookat(*lookat)