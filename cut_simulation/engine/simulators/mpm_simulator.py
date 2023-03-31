import taichi as ti
import numpy as np
import pickle as pkl
import uuid
import os
import torch
from cut_simulation.configs.macros import *
from cut_simulation.utils.misc import *
from cut_simulation.engine.boundaries import create_boundary

@ti.data_oriented
class MPMSimulator:
    def __init__(self, dim, quality, gravity, horizon, max_steps_local, max_steps_global, ckpt_dest):

        self.dim       = dim
        self.ckpt_dest = ckpt_dest
        self.sim_id    = str(uuid.uuid4())
        self.gravity   = ti.Vector(gravity)

        self.n_grid           = int(64 * quality)
        self.dx               = 1 / self.n_grid
        self.inv_dx           = float(self.n_grid)
        self.dt               = 2e-4 / quality / 1.5
        self.p_vol            = (self.dx * 0.5) ** 2
        self.res              = (self.n_grid,) * self.dim
        self.max_steps_local  = max_steps_local
        self.max_steps_global = max_steps_global
        self.horizon          = horizon
        self.n_substeps       = int(2e-3 / self.dt)
        assert self.n_substeps * self.horizon < self.max_steps_global
        assert self.max_steps_local % self.n_substeps == 0

        self.boundary      = None
        self.has_particles = False

        self.ground_friction = 2.0

    def setup_boundary(self, **kwargs):
        self.boundary = create_boundary(**kwargs)

    def build(self, agent, statics, x, mat, used, p_rho, body_id):
        # default boundary
        if self.boundary is None:
            self.boundary = create_boundary()

        # statics
        self.n_statics = len(statics)
        self.statics = statics

        # particles and bodies
        if x is not None:
            self.has_particles = True
            self.n_particles = len(x)
            self.setup_particle_fields()
            self.setup_grid_fields()
            self.setup_ckpt_vars()
            self.init_particles_and_bodies(x, mat, used, p_rho, body_id)
        else:
            self.has_particles = False
            self.n_particles = 0

        # agent
        self.agent = agent

        # misc
        self.cur_step_global = 0
        self.disable_grad() # grad disabled by default

    def setup_particle_fields(self):
        # particle state
        particle_state = ti.types.struct(
            x     = ti.types.vector(self.dim, DTYPE_TI),  # position
            v     = ti.types.vector(self.dim, DTYPE_TI),  # velocity
            C     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # affine velocity field
            F     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # deformation gradient
            F_tmp = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # temp deformation gradient
            U     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            V     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            S     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
        )
        # particle state without gradient
        particle_state_ng = ti.types.struct(
            used = ti.i32,
        )
        # single frame particle state
        particle_state_f = ti.types.struct(
            x    = ti.types.vector(self.dim, ti.f32),
            used = ti.i32,
        )
        # particle info
        particle_info = ti.types.struct(
            mu      = DTYPE_TI,
            lam     = DTYPE_TI,
            mat     = ti.i32,
            mat_cls = ti.i32,
            body_id = ti.i32,
            mass    = DTYPE_TI,
            yield_stress = DTYPE_TI,
        )

        # construct fields
        self.particles    = particle_state.field(shape=(self.max_steps_local+1, self.n_particles), needs_grad=True, layout=ti.Layout.SOA)
        self.particles_ng = particle_state_ng.field(shape=(self.max_steps_local+1, self.n_particles), needs_grad=False, layout=ti.Layout.SOA)
        self.particles_f  = particle_state_f.field(shape=(self.n_particles,), needs_grad=False, layout=ti.Layout.SOA)
        self.particles_i  = particle_info.field(shape=(self.n_particles,), needs_grad=False, layout=ti.Layout.SOA)

    def setup_grid_fields(self):
        grid_cell_state = ti.types.struct(
            v_in  = ti.types.vector(self.dim, DTYPE_TI), # input momentum/velocity
            mass  = DTYPE_TI,                            # mass
            v_out = ti.types.vector(self.dim, DTYPE_TI), # output momentum/velocity
        )
        self.grid = grid_cell_state.field(shape=(self.max_steps_local+1, *self.res), needs_grad=True, layout=ti.Layout.SOA)

    def setup_ckpt_vars(self):
        if self.ckpt_dest == 'disk':
            # placeholder np array from checkpointing
            self.x_np    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            self.v_np    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            self.C_np    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            self.F_np    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            self.used_np = np.zeros((self.n_particles,), dtype=np.int32)
        elif self.ckpt_dest == 'cpu' or 'gpu':
            self.ckpt_ram = dict()
        self.actions_buffer = []
        self.setup_ckpt_dir()

    def setup_ckpt_dir(self):
        self.ckpt_dir = os.path.join('/tmp', 'cut_simulation', self.sim_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def init_particles_and_bodies(self, x, mat, used, p_rho, body_id):
        mu      = np.array([MU[mat_i] for mat_i in mat])
        lam     = np.array([LAMDA[mat_i] for mat_i in mat])
        mat_cls = np.array([MAT_CLASS[mat_i] for mat_i in mat])
        yield_stress = np.array([YIELD_STRESS[mat_i] for mat_i in mat])
        x       = x.astype(DTYPE_NP)
        mat     = mat.astype(np.int32)
        mat_cls = mat_cls.astype(np.int32)
        used    = used.astype(np.int32)
        mu      = mu.astype(DTYPE_NP)
        yield_stress = yield_stress.astype(DTYPE_NP)
        lam     = lam.astype(DTYPE_NP)
        p_rho   = p_rho.astype(DTYPE_NP)
        body_id = body_id.astype(np.int32)

        self.init_particles_kernel(x, mat, mat_cls, used, mu, lam, p_rho, body_id, yield_stress)
        self.init_bodies(mat_cls, body_id)

    @ti.kernel
    def init_particles_kernel(
            self,
            x: ti.types.ndarray(),
            mat: ti.types.ndarray(),
            mat_cls: ti.types.ndarray(),
            used: ti.types.ndarray(),
            mu: ti.types.ndarray(),
            lam: ti.types.ndarray(),
            p_rho: ti.types.ndarray(),
            body_id: ti.types.ndarray(),
            yield_stress: ti.types.ndarray()
        ):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[0, i].x[j] = x[i, j]
            self.particles[0, i].v       = ti.Vector.zero(DTYPE_TI, self.dim)
            self.particles[0, i].F       = ti.Matrix.identity(DTYPE_TI, self.dim)
            self.particles[0, i].C       = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            self.particles_ng[0, i].used = used[i]

            self.particles_i[i].mat     = mat[i]
            self.particles_i[i].mat_cls = mat_cls[i]
            self.particles_i[i].mu      = mu[i]
            self.particles_i[i].lam     = lam[i]
            self.particles_i[i].mass  = self.p_vol * p_rho[i]
            self.particles_i[i].yield_stress = yield_stress[i]
            self.particles_i[i].body_id = body_id[i]

    def init_bodies(self, mat_cls, body_id):
        self.body_ids = np.unique(body_id)
        self.n_bodies = len(self.body_ids)
        assert self.n_bodies == self.body_ids.max() + 1

        # body state, for rigidity enforcement
        body_state = ti.types.struct(
            COM_t0 = ti.types.vector(self.dim, DTYPE_TI),
            COM_t1 = ti.types.vector(self.dim, DTYPE_TI),
            H      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            R      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            U      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            S      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            V      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
        )
        # body info
        body_info = ti.types.struct(
            n_particles = ti.i32,
            mat_cls     = ti.i32,
        )
        self.bodies   = body_state.field(shape=(self.n_bodies,), needs_grad=True, layout=ti.Layout.SOA)
        self.bodies_i = body_info.field(shape=(self.n_bodies,), needs_grad=False, layout=ti.Layout.SOA)

        for i in self.body_ids:
            self.bodies_i[i].n_particles = np.sum(body_id == i)
            self.bodies_i[i].mat_cls = mat_cls[body_id == i][0]

    def reset_grad(self):
        self.particles.grad.fill(0)
        self.grid.grad.fill(0)

    def enable_grad(self):
        '''
        If grad_enable == True, we do checkpointing when gpu memory is not enough for storing the whole episode.
        '''
        self.grad_enabled    = True
        self.cur_step_global = 0

    def disable_grad(self):
        self.grad_enabled    = False
        self.cur_step_global = 0

    # --------------------------------- MPM part -----------------------------------
    @ti.kernel
    def reset_grid_and_grad(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.grid[f, I].fill(0)
            self.grid.grad[f, I].fill(0)

    def global_step_to_local(self, step_global):
        step_local = step_global % self.max_steps_local
        assert 0 <= step_local < self.max_steps_local
        return step_local

    @property
    def cur_step_local(self):
        return self.global_step_to_local(self.cur_step_global)

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                self.particles[f, p].F_tmp = (ti.Matrix.identity(DTYPE_TI, self.dim) + self.dt * self.particles[f, p].C) @ self.particles[f, p].F

    @ti.kernel
    def svd(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                self.particles[f, p].U, self.particles[f, p].S, self.particles[f, p].V = ti.svd(self.particles[f, p].F_tmp, DTYPE_TI)

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                self.particles.grad[f, p].F_tmp += self.backward_svd(self.particles.grad[f, p].U, self.particles.grad[f, p].S, self.particles.grad[f, p].V, self.particles[f, p].U, self.particles[f, p].S, self.particles[f, p].V)

    @ti.func
    def backward_svd(self, grad_U, grad_S, grad_V, U, S, V):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = V.transpose()
        ut = U.transpose()
        S_term = U @ grad_S @ vt

        s = ti.Vector.zero(DTYPE_TI, self.dim)
        if ti.static(self.dim==2):
            s = ti.Vector([S[0, 0], S[1, 1]]) ** 2
        else:
            s = ti.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
        F = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
        for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
            if i == j:
                F[i, j] = 0
            else:
                F[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
        v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
        return u_term + v_term + S_term

    @ti.func
    def clamp(self, a):
        # remember that we don't support if return in taichi
        # stop the gradient ...
        if a>=0:
            a = ti.max(a, 1e-6)
        else:
            a = ti.min(a, -1e-6)
        return a

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.kernel
    def advect_used(self, f: ti.i32):
        for p in range(self.n_particles):
            self.particles_ng[f+1, p].used = self.particles_ng[f, p].used

    @ti.kernel
    def process_unused_particles(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used == 0:
                self.particles[f+1, p].v = self.particles[f, p].v
                self.particles[f+1, p].x = self.particles[f, p].x
                self.particles[f+1, p].C = self.particles[f, p].C
                self.particles[f+1, p].F = self.particles[f, p].F

    def agent_act(self, f, is_none_action):
        if not is_none_action:
            self.agent.act(f, self.cur_step_global)

    def agent_act_grad(self, f, is_none_action):
        if not is_none_action:
            self.agent.act_grad(f, self.cur_step_global)


    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.func
    def make_matrix_from_diag(self, d):
        return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=DTYPE_TI)

    @ti.func
    def compute_von_mises(self, F, U, sig, V, yield_stress, mu):
        # epsilon = ti.Vector([0., 0., 0.], dt=self.dtype)
        epsilon = ti.Vector.zero(DTYPE_TI, self.dim)
        sig = ti.max(sig, 0.05)  # add this to prevent NaN in extrem cases
        epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
        epsilon_hat = epsilon - (epsilon.sum() / self.dim)
        epsilon_hat_norm = self.norm(epsilon_hat)
        delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu)

        if delta_gamma > 0:  # Yields
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            sig = self.make_matrix_from_diag(ti.exp(epsilon))
            F = U @ sig @ V.transpose()
        return F

    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                base = (self.particles[f, p].x * self.inv_dx - 0.5).cast(int)
                fx   = self.particles[f, p].x * self.inv_dx - base.cast(DTYPE_TI)
                w    = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

                F_new = self.compute_von_mises(self.particles[f, p].F_tmp, self.particles[f, p].U, self.particles[f, p].S, self.particles[f, p].V, self.particles_i[p].yield_stress, self.particles_i[p].mu)
                J = F_new.determinant()
                r = self.particles[f, p].U @ self.particles[f, p].V.transpose()
                stress = 2 * self.particles_i[p].mu * (F_new - r) @ F_new.transpose() + ti.Matrix.identity(DTYPE_TI, self.dim) * self.particles_i[p].lam * J * (J - 1)
                stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
                
                affine = stress + self.particles_i[p].mass * self.particles[f, p].C
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = (offset.cast(DTYPE_TI) - fx) * self.dx
                    weight = ti.cast(1.0, DTYPE_TI)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]

                    self.grid[f, base + offset].v_in += weight * (self.particles_i[p].mass * self.particles[f, p].v + affine @ dpos)
                    self.grid[f, base + offset].mass += weight * self.particles_i[p].mass
                self.particles[f+1, p].F = F_new

    @ti.kernel
    def grid_op(self, f: ti.i32, f_global: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            if self.grid[f, I].mass > 1e-12:  # No need for epsilon here, 1e-10 is to prevent potential numerical problems ..
                v_out = (1 / self.grid[f, I].mass) * self.grid[f, I].v_in  # Momentum to velocity
                v_out += self.dt * self.gravity # gravity

                # collide with statics
                if ti.static(self.n_statics>0):
                    for i in ti.static(range(self.n_statics)):
                        v_out = self.statics[i].collide(I*self.dx, v_out)

                # collide with agent
                if ti.static(self.agent is not None):
                    if ti.static(self.agent.collide_type in ['grid', 'both']):
                        v_out = self.agent.collide(f, I*self.dx, v_out, self.dt, self.grid[f, I].mass, f_global)
                        # v_out = self.agent.collide(f, I*self.dx, v_out, self.dt, self.grid[f, I].mass, self.cur_step_global)

                bound = 3
                eps = ti.cast(1e-30, DTYPE_TI)
                for d in ti.static(range(self.dim)):
                    if I[d] < bound and v_out[d] < 0:
                        if ti.static(d != 1 or self.ground_friction == 0):
                            v_out[d] = 0  # Boundary conditions
                        else:
                            if ti.static(self.ground_friction < 10):
                                # TODO: 1e-30 problems ...
                                normal = ti.Vector.zero(DTYPE_TI, self.dim)
                                normal[d] = 1.
                                lin = v_out.dot(normal) + eps
                                vit = v_out - lin * normal - I * eps
                                lit = self.norm(vit)
                                v_out = ti.max(1. + ti.static(self.ground_friction) * lin / lit, 0.) * (vit + I * eps)
                            else:
                                v_out = ti.Vector.zero(DTYPE_TI, self.dim)
                    if I[d] > self.res[d] - bound and v_out[d] > 0: v_out[d] = 0
                self.grid[f, I].v_out = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                base = (self.particles[f, p].x * self.inv_dx - 0.5).cast(int)
                fx = self.particles[f, p].x * self.inv_dx - base.cast(DTYPE_TI)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector.zero(DTYPE_TI, self.dim)
                new_C = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = offset.cast(DTYPE_TI) - fx
                    g_v = self.grid[f, base + offset].v_out
                    weight = ti.cast(1.0, DTYPE_TI)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

                # collide with agent
                if ti.static(self.agent is not None):
                    if ti.static(self.agent.collide_type in ['particle', 'both']):
                        new_x_tmp = self.particles[f, p].x + self.dt * new_v
                        new_v = self.agent.collide(f, new_x_tmp, new_v, self.dt)

                # advect to next frame    
                self.particles[f+1, p].v = new_v
                self.particles[f+1, p].C = new_C

    def advect(self, f):
        self.reset_bodies_and_grad()
        self.compute_COM(f)
        self.compute_H(f)
        self.compute_H_svd(f)
        self.compute_R(f)
        self.advect_kernel(f)

    def advect_grad(self, f):
        self.reset_bodies_and_grad()
        self.compute_COM(f)
        self.compute_H(f)
        self.compute_H_svd(f)
        self.compute_R(f)

        self.advect_kernel.grad(f)
        self.compute_R.grad(f)
        self.compute_H_svd_grad(f)
        self.compute_H.grad(f)
        self.compute_COM.grad(f)

    @ti.kernel
    def reset_bodies_and_grad(self):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].fill(0)
                self.bodies.grad[body_id].fill(0)

    @ti.kernel
    def compute_COM(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used and self.particles_i[p].mat_cls == MAT_RIGID:
                body_id = self.particles_i[p].body_id
                self.bodies[body_id].COM_t0 += self.particles[f, p].x / ti.cast(self.bodies_i[body_id].n_particles, DTYPE_TI)
                self.bodies[body_id].COM_t1 += (self.particles[f, p].x + self.dt * self.particles[f+1, p].v) / ti.cast(self.bodies_i[body_id].n_particles, DTYPE_TI)

    @ti.kernel
    def compute_H(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used and self.particles_i[p].mat_cls == MAT_RIGID:
                body_id = self.particles_i[p].body_id
                self.bodies[body_id].H[0, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[0, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[0, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[2]
                self.bodies[body_id].H[1, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[1, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[1, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[2]
                self.bodies[body_id].H[2, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[2, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[2, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[2]

    @ti.kernel
    def compute_H_svd(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].U, self.bodies[body_id].S, self.bodies[body_id].V = ti.svd(self.bodies[body_id].H, DTYPE_TI)

    @ti.kernel
    def compute_H_svd_grad(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies.grad[body_id].H = self.backward_svd(self.bodies.grad[body_id].U, self.bodies.grad[body_id].S, self.bodies.grad[body_id].V, self.bodies[body_id].U, self.bodies[body_id].S, self.bodies[body_id].V)

    @ti.kernel
    def compute_R(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].R = self.bodies[body_id].V @ self.bodies[body_id].U.transpose()

    @ti.kernel
    def advect_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                if self.particles_i[p].mat_cls == MAT_RIGID: # rigid objects
                    body_id = self.particles_i[p].body_id
                    self.particles[f+1, p].x = self.bodies[body_id].R @ (self.particles[f, p].x - self.bodies[body_id].COM_t0) + self.bodies[body_id].COM_t1
                else: # other particles
                    self.particles[f+1, p].x = self.particles[f, p].x + self.dt * self.particles[f+1, p].v

    def agent_move(self, f, is_none_action):
        if not is_none_action:
            self.agent.move(f)

    def agent_move_grad(self, f, is_none_action):
        if not is_none_action:
            self.agent.move_grad(f)

    def substep(self, f, is_none_action):
        # reset force
        self.agent.effectors[0].reset_force_and_work(self.cur_step_global)

        if self.has_particles:
            self.reset_grid_and_grad(f)
            self.advect_used(f)
            self.process_unused_particles(f)

        self.agent_act(f, is_none_action)

        if self.has_particles:
            self.compute_F_tmp(f)
            self.svd(f)
            self.p2g(f)

        self.agent_move(f, is_none_action)

        if self.has_particles:
            self.grid_op(f, self.cur_step_global)
            self.g2p(f)
            self.advect(f)


    def substep_grad(self, f, is_none_action):
        if self.has_particles:
            self.advect_grad(f)
            self.g2p.grad(f)
            self.grid_op.grad(f, self.cur_step_global)

        self.agent_move_grad(f, is_none_action)

        if self.has_particles:
            self.p2g.grad(f)
            self.svd_grad(f)
            self.compute_F_tmp.grad(f)

        self.agent_act_grad(f, is_none_action)

        if self.has_particles:
            self.process_unused_particles.grad(f)
            self.advect_used.grad(f)


    # ------------------------------------ io -------------------------------------#
    @ti.kernel
    def readframe(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), F: ti.types.ndarray(), C: ti.types.ndarray(), used: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[f, i].x[j]
                v[i, j] = self.particles[f, i].v[j]
                for k in ti.static(range(self.dim)):
                    F[i, j, k] = self.particles[f, i].F[j, k]
                    C[i, j, k] = self.particles[f, i].C[j, k]
            used[i] = self.particles_ng[f, i].used

    @ti.kernel
    def setframe(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), F: ti.types.ndarray(), C: ti.types.ndarray(), used: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[f, i].x[j] = x[i, j]
                self.particles[f, i].v[j] = v[i, j]
                for k in ti.static(range(self.dim)):
                    self.particles[f, i].F[j, k] = F[i, j, k]
                    self.particles[f, i].C[j, k] = C[i, j, k]
            self.particles_ng[f, i].used = used[i]

    @ti.kernel
    def copy_frame_particle(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_particles):
            self.particles[target, i].x = self.particles[source, i].x
            self.particles[target, i].v = self.particles[source, i].v
            self.particles[target, i].F = self.particles[source, i].F
            self.particles[target, i].C = self.particles[source, i].C
            self.particles_ng[target, i].used = self.particles_ng[source, i].used

    @ti.kernel
    def copy_frame_agent(self, source: ti.i32, target: ti.i32):
        if ti.static(self.agent is not None):
            self.agent.copy_frame(source, target)

    @ti.kernel
    def copy_grad_particle(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_particles):
            self.particles.grad[target, i].x = self.particles.grad[source, i].x
            self.particles.grad[target, i].v = self.particles.grad[source, i].v
            self.particles.grad[target, i].F = self.particles.grad[source, i].F
            self.particles.grad[target, i].C = self.particles.grad[source, i].C
            self.particles_ng[target, i].used = self.particles_ng[source, i].used

    @ti.kernel
    def copy_grad_agent(self, source: ti.i32, target: ti.i32):
        if ti.static(self.agent is not None):
            self.agent.copy_grad(source, target)

    @ti.kernel
    def reset_grad_till_frame_particle(self, f: ti.i32):
        for i, j in ti.ndrange(f, self.n_particles):
            self.particles.grad[i, j].fill(0)

    @ti.kernel
    def reset_grad_till_frame_agent(self, f: ti.i32):
        if ti.static(self.agent is not None):
            self.agent.reset_grad_till_frame(f)

    def get_state(self):
        f = self.cur_step_local
        state = {}
        if self.has_particles:
            state['x']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['F']    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['C']    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['used'] = np.zeros((self.n_particles,), dtype=np.int32)
            self.readframe(f, state['x'], state['v'], state['F'], state['C'], state['used'])
        if self.agent is not None:
            state['agent'] = self.agent.get_state(f)
        return state

    def set_state(self, f_global, state):
        f = self.global_step_to_local(f_global)
        if self.has_particles:
            self.setframe(f, state['x'], state['v'], state['F'], state['C'], state['used'])
        if self.agent is not None:
            self.agent.set_state(f, state['agent'])

    @ti.kernel
    def get_x_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[f, i].x[j]

    @ti.kernel
    def get_frame_state_kernel(self, f: ti.i32):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles_f[i].x[j] = ti.cast(self.particles[f, i].x[j], ti.f32)
            self.particles_f[i].used = ti.cast(self.particles_ng[f, i].used, ti.i32)

    def get_x(self, f=None):
        if f is None:
            f = self.cur_step_local

        x = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_x_kernel(f, x)
        return x

    def get_frame_state(self, f):
        self.get_frame_state_kernel(f)
        return self.particles_f

    @ti.kernel
    def get_v_kernel(self, f: ti.i32, v: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                v[i, j] = self.particles[f, i].v[j]

    @ti.kernel
    def get_ckpt_kernel(self, x_np: ti.types.ndarray(), v_np: ti.types.ndarray(), C_np: ti.types.ndarray(), F_np: ti.types.ndarray(), used_np: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x_np[i, j] = self.particles[0, i].x[j]
                v_np[i, j] = self.particles[0, i].v[j]
                for k in ti.static(range(self.dim)):
                    C_np[i, j, k] = self.particles[0, i].C[j, k]
                    F_np[i, j, k] = self.particles[0, i].F[j, k]
            used_np[i] = self.particles_ng[0, i].used

    @ti.kernel
    def set_ckpt_kernel(self, x_np: ti.types.ndarray(), v_np: ti.types.ndarray(), C_np: ti.types.ndarray(), F_np: ti.types.ndarray(), used_np: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[0, i].x[j] = x_np[i, j]
                self.particles[0, i].v[j] = v_np[i, j]
                for k in ti.static(range(self.dim)):
                    self.particles[0, i].C[j, k] = C_np[i, j, k]
                    self.particles[0, i].F[j, k] = F_np[i, j, k]
            self.particles_ng[0, i].used = used_np[i]

    def get_v(self, f):
        v = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_v_kernel(f, v)
        return v

    def step(self, action=None):
        if self.grad_enabled:
            if self.cur_step_local == 0:
                self.actions_buffer = []

        self.step_(action)

        if self.grad_enabled:
            self.actions_buffer.append(action)

        if self.cur_step_local == 0:
            self.memory_to_cache()


    def step_(self, action=None):
        is_none_action = action is None
        if not is_none_action:
            self.agent.set_action(
                s=self.cur_step_local//self.n_substeps,
                s_global=self.cur_step_global//self.n_substeps,
                n_substeps=self.n_substeps,
                action=action
            )

        for i in range(0, self.n_substeps):
            self.substep(self.cur_step_local, is_none_action)
            self.cur_step_global += 1

        assert self.cur_step_global <= self.max_steps_global

    def step_grad(self, action=None):     
        if self.cur_step_local == 0:
            self.memory_from_cache()
            
        is_none_action = action is None

        for i in range(self.n_substeps-1, -1, -1):
            self.cur_step_global -= 1
            self.substep_grad(self.cur_step_local, is_none_action)

        if not is_none_action:
            self.agent.set_action_grad(
                s=self.cur_step_local//self.n_substeps,
                s_global=self.cur_step_global//self.n_substeps, 
                n_substeps=self.n_substeps,
                action=action
            )

    def memory_to_cache(self):
        if self.grad_enabled:
            ckpt_start_step = self.cur_step_global - self.max_steps_local
            ckpt_end_step = self.cur_step_global - 1
            ckpt_name = f'{ckpt_start_step:06d}'

            if self.ckpt_dest == 'disk':
                ckpt = {}
                if self.has_particles:
                    self.get_ckpt_kernel(self.x_np, self.v_np, self.C_np, self.F_np, self.used_np)
                    ckpt['x']       = self.x_np
                    ckpt['v']       = self.v_np
                    ckpt['C']       = self.C_np
                    ckpt['F']       = self.F_np
                    ckpt['used']    = self.used_np
                    ckpt['actions'] = self.actions_buffer

                if self.agent is not None:
                    ckpt['agent'] = self.agent.get_ckpt()

                # save to /tmp
                ckpt_file = os.path.join(self.ckpt_dir, f'{ckpt_name}.pkl')
                if os.path.exists(ckpt_file):
                    os.remove(ckpt_file)
                pkl.dump(ckpt, open(ckpt_file, 'wb'))

            elif self.ckpt_dest in ['cpu', 'gpu']:
                if not ckpt_name in self.ckpt_ram:
                    self.ckpt_ram[ckpt_name] = {}
                    if self.ckpt_dest == 'cpu':
                        device = 'cpu'
                    elif self.ckpt_dest == 'gpu':
                        device = 'cuda'

                    if self.has_particles:
                        self.ckpt_ram[ckpt_name]['x']    = torch.zeros((self.n_particles, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['v']    = torch.zeros((self.n_particles, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['C']    = torch.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['F']    = torch.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['used'] = torch.zeros((self.n_particles,), dtype=torch.int32, device=device)
                
                if self.has_particles:
                    self.get_ckpt_kernel(
                        self.ckpt_ram[ckpt_name]['x'], 
                        self.ckpt_ram[ckpt_name]['v'],
                        self.ckpt_ram[ckpt_name]['C'],
                        self.ckpt_ram[ckpt_name]['F'],
                        self.ckpt_ram[ckpt_name]['used']
                    )
                self.ckpt_ram[ckpt_name]['actions'] = list(self.actions_buffer)

                if self.agent is not None:
                    self.agent.get_ckpt(ckpt_name)

            else:
                assert False

        if self.has_particles:
            self.copy_frame_particle(self.max_steps_local, 0) # restart from frame 0 in memory
        self.copy_frame_agent(self.max_steps_local, 0)

    def memory_from_cache(self):
        assert self.grad_enabled
        if self.has_particles:
            self.copy_frame_particle(0, self.max_steps_local)
            self.copy_grad_particle(0, self.max_steps_local)
            self.reset_grad_till_frame_particle(self.max_steps_local)
            
        self.copy_frame_agent(0, self.max_steps_local)
        self.copy_grad_agent(0, self.max_steps_local)
        self.reset_grad_till_frame_agent(self.max_steps_local)

        ckpt_start_step = self.cur_step_global - self.max_steps_local
        ckpt_end_step = self.cur_step_global - 1
        ckpt_name = f'{ckpt_start_step:06d}'

        if self.ckpt_dest == 'disk':
            ckpt_file = os.path.join(self.ckpt_dir, f'{ckpt_start_step:06d}.pkl')
            assert os.path.exists(ckpt_file)
            ckpt = pkl.load(open(ckpt_file, 'rb'))
            if self.agent is not None:
                self.agent.set_ckpt(ckpt=ckpt['agent'])

        elif self.ckpt_dest in ['cpu', 'gpu']:
            ckpt = self.ckpt_ram[ckpt_name]
            if self.agent is not None:
                self.agent.set_ckpt(ckpt_name=ckpt_name)

        else:
            assert False

        if self.has_particles:
            self.set_ckpt_kernel(ckpt['x'], ckpt['v'], ckpt['C'], ckpt['F'], ckpt['used'])

        # now that we loaded the first frame, we do a forward pass to fill up the rest 
        self.cur_step_global = ckpt_start_step
        for action in ckpt['actions']:
            self.step_(action)