import os
import torch
import numpy as np
import taichi as ti
import pickle as pkl
from geomloss import SamplesLoss
from sklearn.neighbors import KDTree
from cut_simulation.engine.simulators import MPMSimulator
from cut_simulation.configs.macros import *
from cut_simulation.utils.misc import *
import matplotlib.pyplot as plt


@ti.data_oriented
class Loss:
    def __init__(self, weights, target_file):
        self.weights = weights
        self.target_file = target_file
        self.inf = 1e8

        self.loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.chamfer_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.EMD_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)

        self.chamfer_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.EMD_weight = ti.field(dtype=DTYPE_TI, shape=())

        self.EMD_p = 2
        self.EMD_blur = 0.01
        self.EMD_loss_F = SamplesLoss(loss="sinkhorn", p=self.EMD_p, blur=self.EMD_blur, debias=False, potentials=True)
        self.EMD_transport_plan = None
        self.EMD_loss_smallest = self.inf
        self.EMD_loss_plateau_count = 0


    def build(self, sim):
        self.sim = sim
        self.res = sim.res
        self.n_grid = sim.n_grid
        self.dx = sim.dx
        self.dim = sim.dim
        self.grid_mass = sim.grid.mass
        self.particle_x = sim.particles.x
        self.particle_mat = sim.particles_i.mat
        self.particle_used = sim.particles_ng.used

        self.chamfer_weight[None] = self.weights['chamfer']
        self.EMD_weight[None] = self.weights['EMD']

        self.load_target(self.target_file)
        self.clear()

    def reset_grad(self):
        self.chamfer_loss.grad.fill(0)
        self.EMD_loss.grad.fill(0)
        self.loss.grad.fill(1)
        
    def load_target(self, path):
        target = pkl.load(open(path, 'rb'))
        assert self.dim == target['x'].shape[1]
        valid_tgt_ids = np.where(np.logical_and(target['mat'] == MILK, target['used']))[0]

        self.tgt_particles_x_np = target['x'][valid_tgt_ids].astype(DTYPE_NP)

        self.tgt_particles_n = ti.field(dtype=ti.i32, shape=())
        self.tgt_particles_n[None] = len(valid_tgt_ids)

        self.tgt_particles_x = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.tgt_particles_x_np.shape[:1])
        self.tgt_particles_x_f32 = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.tgt_particles_x_np.shape[:1])

        self.tgt_particles_x.from_numpy(self.tgt_particles_x_np)
        self.tgt_particles_x_f32.from_numpy(self.tgt_particles_x_np.astype(np.float32))
        self.tgt_particles_x_tree = KDTree(self.tgt_particles_x_np)
        self.tgt_particles_x_torch = torch.from_numpy(self.tgt_particles_x_np).cuda()


        self.cur_to_tgt_ids = ti.field(dtype=ti.i32, shape=self.tgt_particles_x_np.shape[:1])
        self.tgt_to_cur_ids = ti.field(dtype=ti.i32, shape=self.tgt_particles_x_np.shape[:1])
        self.valid_cur_ids = ti.field(dtype=ti.i32, shape=self.tgt_particles_x_np.shape[:1])

        print(f'===> target loaded from {path}.')

    @ti.kernel
    def set_cur_to_tgt_ids(self, cur_to_tgt_ids: ti.types.ndarray()):
        for i in range(cur_to_tgt_ids.shape[0]):
            self.cur_to_tgt_ids[i] = cur_to_tgt_ids[i]

    @ti.kernel
    def set_tgt_to_cur_ids(self, tgt_to_cur_ids: ti.types.ndarray()):
        for i in range(tgt_to_cur_ids.shape[0]):
            self.tgt_to_cur_ids[i] = tgt_to_cur_ids[i]


    # -----------------------------------------------------------
    # compute loss
    # -----------------------------------------------------------
    def compute_valid_ids(self, f):
        self.valid_cur_ids_np = np.where(np.logical_and(self.particle_mat.to_numpy() == MILK, self.particle_used.to_numpy()[f]))[0]
        self.valid_cur_ids.from_numpy(self.valid_cur_ids_np.astype(np.int32))

    def compute_chamfer_loss(self, f):
        cur_particles_x_np = self.particle_x.to_numpy()[f, self.valid_cur_ids_np]
        cur_particles_x_tree = KDTree(cur_particles_x_np)

        _, cur_to_tgt_ids = self.tgt_particles_x_tree.query(cur_particles_x_np)
        _, tgt_to_cur_ids = cur_particles_x_tree.query(self.tgt_particles_x_np)
        cur_to_tgt_ids = cur_to_tgt_ids[:, 0].astype(np.int32)
        tgt_to_cur_ids = tgt_to_cur_ids[:, 0].astype(np.int32)
        self.set_cur_to_tgt_ids(cur_to_tgt_ids)
        self.set_tgt_to_cur_ids(tgt_to_cur_ids)
        self.compute_chamfer_loss_kernel(f)

    def compute_chamfer_loss_grad(self, f):
        self.compute_chamfer_loss_kernel.grad(f)

    @ti.kernel
    def compute_chamfer_loss_kernel(self, f: ti.i32):
        for i in range(self.tgt_particles_n[None]):
            self.chamfer_loss[None] += ti.abs(self.particle_x[f, self.valid_cur_ids[i]] - self.tgt_particles_x[self.cur_to_tgt_ids[i]]).sum()
        for i in range(self.tgt_particles_n[None]):
            self.chamfer_loss[None] += ti.abs(self.particle_x[f, self.valid_cur_ids[self.tgt_to_cur_ids[i]]] - self.tgt_particles_x[i]).sum()

    # def compute_EMD_loss(self, f):
    #     self.compute_EMD_loss_kernel(f)

    # @ti.kernel
    # def compute_EMD_loss_kernel(self, f: ti.i32):
    #     for i in range(self.tgt_particles_n[None]):
    #         self.EMD_loss[None] += ti.abs(self.particle_x[f, i] - self.tgt_particles_x[i]).sum()

    def compute_EMD_loss(self, f):
        # we re-enable dynamic correspondence finding once we are in a good spot
        update_transport_plan = False
        if self.EMD_loss_plateau_count >= 10:
            self.EMD_loss_plateau_count = 0
            self.EMD_loss_smallest = self.inf
            update_transport_plan = True

        if self.EMD_transport_plan is None or update_transport_plan:
            print('==> Updating transport plan...')
            cur_particles_x_torch = self.particle_x.to_torch()[f, self.valid_cur_ids_np].cuda()

            N, M, D = cur_particles_x_torch.shape[0], self.tgt_particles_x_torch.shape[0], cur_particles_x_torch.shape[1]
            F, G = self.EMD_loss_F(cur_particles_x_torch, self.tgt_particles_x_torch)
            x_i = cur_particles_x_torch.view(N, 1, D)
            y_j = self.tgt_particles_x_torch.view(1, M, D)
            F_i, G_j = F.view(N, 1), G.view(1, M)
            C_ij = (1 / self.EMD_p) * ((x_i - y_j) ** self.EMD_p).sum(-1)  # (N,M) cost matrix
            eps = self.EMD_blur ** self.EMD_p  # temperature epsilon
            P_ij = ((F_i + G_j - C_ij) / eps).exp()  # (N,M) transport plan
            P_ij /= P_ij.sum(1, keepdim=True)
            torch.cuda.synchronize() # without synchronize it's buggy. no idea why..
            # import IPython;IPython.embed()
            # plt.imshow((P_ij/P_ij.max(1, keepdim=True)[0]).detach().cpu().numpy())
            # from time import time
            # plt.savefig(f'tmp/{time()}.png')
            if self.EMD_transport_plan is None:
                self.EMD_transport_plan = ti.field(dtype=DTYPE_TI, shape=(N, M))
            self.EMD_transport_plan.from_torch(P_ij.contiguous())

        self.compute_EMD_loss_kernel(f)

        # check if loss plateaus
        self.EMD_loss_improve_rate = (self.EMD_loss_smallest - self.EMD_loss[None])/self.EMD_loss_smallest
        if self.EMD_loss_improve_rate < 2e-3:
            self.EMD_loss_plateau_count += 1
            print('strike!!!!!!!!!!!!!!!!!!!', self.EMD_loss_plateau_count)
        else:
            self.EMD_loss_plateau_count = 0

        if self.EMD_loss_smallest > self.EMD_loss[None]:
            self.EMD_loss_smallest = self.EMD_loss[None]


    @ti.kernel
    def compute_EMD_loss_kernel(self, f: ti.i32):
        for i, j in ti.ndrange(self.tgt_particles_n[None], self.tgt_particles_n[None]):
            self.EMD_loss[None] += ti.abs(self.particle_x[f, self.valid_cur_ids[i]] - self.tgt_particles_x[j]).sum() * self.EMD_transport_plan[i, j]

    def compute_EMD_loss_grad(self, f):
        self.compute_EMD_loss_kernel.grad(f)


    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] += self.chamfer_loss[None] * self.chamfer_weight[None]
        self.loss[None] += self.EMD_loss[None] * self.EMD_weight[None]

    @ti.kernel
    def clear_losses(self):
        self.chamfer_loss[None] = 0
        self.EMD_loss[None] = 0

        self.chamfer_loss.grad[None] = 0
        self.EMD_loss.grad[None] = 0

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0

    @ti.ad.grad_replaced
    def compute_loss(self, f):
        self.clear_losses()

        self.compute_valid_ids(f)

        self.compute_chamfer_loss(f)
        self.compute_EMD_loss(f)
        self.sum_up_loss_kernel()

    @ti.ad.grad_for(compute_loss)
    def compute_loss_grad(self, f):
        self.clear_losses()

        self.sum_up_loss_kernel.grad()
        self.compute_EMD_loss_grad(f)
        self.compute_chamfer_loss_grad(f)

    def _extract_loss(self, f):
        self.compute_loss(f)

        return {
            'loss': self.loss[None],
            'chamfer_loss': self.chamfer_loss[None],
            'EMD_loss': self.EMD_loss[None],
        }

    def _extract_loss_grad(self, f):
        self.compute_loss_grad(f)

    def reset(self):
        self.clear_loss()
        self._last_loss = 0 # in optim, loss will be clear after ti.Tape; for RL; we reset loss to zero in each step.

    def get_loss(self, f):
        loss_info = self._extract_loss(f)
        print('loss:', self.loss[None])
        r = 0
        cur_step_loss = loss_info['loss'] - self._last_loss
        self._last_loss = loss_info['loss']

        loss_info['reward'] = r
        loss_info['loss'] = cur_step_loss
        return loss_info

    def get_loss_grad(self, f):
        self._extract_loss_grad(f)

    def clear(self):
        self.clear_loss()
        self._last_loss = 0