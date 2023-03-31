import taichi as ti
import numpy as np
import yaml
import torch
from .effector import Effector
from cut_simulation.engine.meshes import Dynamic
from cut_simulation.configs.macros import *

@ti.data_oriented
class Rigid(Effector):
    # Rigid end-effector. Can be a stirrer, ice-cream cone, laddle, whatever...
    def __init__(self, **kwargs):
        super(Rigid, self).__init__(**kwargs)
        self.mesh = None

        # magic, don't touch. (I finally got a chance to write something like this lol, thanks to stupid taichi)
        self.magic = ti.field(dtype=DTYPE_TI, shape=())
        self.magic.fill(0)

    def setup_mesh(self, **kwargs):
        self.mesh = Dynamic(
            container=self,
            has_dynamics=True,
            **kwargs
        )

    def apply_action_p(self, action_p):
        action_p = action_p.astype(DTYPE_NP)
        self.set_action_p_kernel(action_p)
        self.apply_action_p_kernel()

        self.update_latest_pos(0) # modified by xzj
        self.update_mesh_pose(0) # modified by xzj

    def move(self, f):
        self.move_kernel(f)
        self.update_latest_pos(f)
        self.update_mesh_pose(f)
        
    def update_mesh_pose(self, f):
        # For visualization only. No need to compute grad.
        self.mesh.update_vertices(f)

    @ti.func
    def collide(self, f, pos_world, mat_v, dt, mass, f_global):
        return self.mesh.collide(f, pos_world, mat_v, dt, mass, f_global)
