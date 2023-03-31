import taichi as ti
import numpy as np
from .mesh import Mesh
import cut_simulation.utils.geom as geom_utils
from cut_simulation.configs.macros import *

@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

@ti.data_oriented
class Dynamic(Mesh):
    # Dynamic mesh-based object
    def __init__(self, container, use_knife_sdf=False, **kwargs):
        self.container = container
        super(Dynamic, self).__init__(**kwargs)

        self.use_knife_sdf = use_knife_sdf

        self.knife_width = ti.field(DTYPE_TI, shape=())
        self.knife_width[None] = 0.14 * self.scale[1]
        self.h = ti.Vector.field(2, DTYPE_TI, shape=())
        self.prot = ti.Vector.field(4, DTYPE_TI, shape=())
        self.size = ti.Vector.field(3, DTYPE_TI, shape=())
        self.h[None] = np.array([self.knife_width[None] * 2 / 3, 0.1])
        self.prot[None] = np.array([1.0, 0.0, 0.0, 0.58])
        self.size[None] = np.array([0.016, 0.2, 0.2])

    def init_transform(self):
        super(Dynamic, self).init_transform()
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))

    @ti.kernel
    def update_vertices(self, f: ti.i32):
        for i in self.vertices:
            self.vertices[i] = ti.cast(geom_utils.transform_by_trans_quat_ti(self.init_vertices[i], self.container.pos[f], self.container.quat[f]), self.vertices.dtype)

    @ti.func
    def sdf_knife_(self, pos_mesh):
        # sdf value from mesh coordinate
        q = ti.abs(pos_mesh) - self.size[None]
        sdf_box = geom_utils.length(ti.max(q, 0.0)) + ti.min(ti.max(q[0], ti.max(q[1], q[2])), 0.0)

        pos_mesh = pos_mesh - ti.Vector([0., self.h[None][0], 0.])
        inv_quat = ti.Vector([self.prot[None][0], -self.prot[None][1],
                              -self.prot[None][2], -self.prot[None][3]]).normalized()
        pos_mesh = geom_utils.transform_by_quat_ti(pos_mesh, inv_quat)
        q = ti.abs(pos_mesh)
        sdf_prism = ti.max(q[2] - self.h[None][1], ti.max(q[0] * ti.sqrt(3.0) * 0.5 + pos_mesh[1] * 0.5, -pos_mesh[1]) - self.h[None][0] * 0.5)
        return ti.max(sdf_box, sdf_prism)
    

    @ti.func
    def normal_knife_(self, pos_mesh):
        d = ti.cast(1e-4, DTYPE_TI)
        normal_vec = ti.Vector([0, 0, 0], dt=DTYPE_TI)
        for i in ti.static(range(3)):
            inc = pos_mesh
            dec = pos_mesh
            inc[i] += d
            dec[i] -= d
            normal_vec[i] = (self.sdf_knife_(inc) - self.sdf_knife_(dec)) / (2 * d)

        normal_vec = geom_utils.normalize(normal_vec)
        return normal_vec

    @ti.func
    def sdf(self, f, pos_world):
        # sdf value from world coordinate
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        sdf = ti.cast(0.0, DTYPE_TI)
        if self.use_knife_sdf:
            sdf = self.sdf_knife_(pos_mesh)
        else:
            pos_voxels = geom_utils.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)
            sdf = self.sdf_(pos_voxels)
        return sdf

    @ti.func
    def sdf_(self, pos_voxels):
        # sdf value from voxels coordinate
        base = ti.floor(pos_voxels, ti.i32)
        signed_dist = ti.cast(0.0, DTYPE_TI)
        if (base >= self.sdf_voxels_res - 1).any() or (base < 0).any():
            signed_dist = 1.0
        else:
            signed_dist = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                voxel_pos = base + offset
                w_xyz = 1 - ti.abs(pos_voxels - voxel_pos)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                signed_dist += w * self.sdf_voxels[voxel_pos]

        return signed_dist

    @ti.func
    def normal(self, f, pos_world):
        # compute normal with finite difference
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        
        normal_vec_mesh = ti.Vector([0, 0, 0], dt=DTYPE_TI)
        if self.use_knife_sdf:
            normal_vec_mesh = self.normal_knife_(pos_mesh)
        else:
            pos_voxels = geom_utils.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)
            normal_vec_voxels = self.normal_(pos_voxels)
            R_voxels_to_mesh = self.T_mesh_to_voxels[None][:3, :3].inverse()
            normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels

        normal_vec_world = geom_utils.transform_by_quat_ti(normal_vec_mesh, self.container.quat[f])
        normal_vec_world = geom_utils.normalize(normal_vec_world)

        if isnan(normal_vec_world[0]):
            print('=>nan!!!!!')

        return normal_vec_world

    @ti.func
    def normal_(self, pos_voxels):
        # since we are in voxels frame, delta can be a relatively big value
        delta = ti.cast(1e-2, DTYPE_TI)
        normal_vec = ti.Vector([0, 0, 0], dt=DTYPE_TI)

        for i in ti.static(range(3)):
            inc = pos_voxels
            dec = pos_voxels
            inc[i] += delta
            dec[i] -= delta
            normal_vec[i] = (self.sdf_(inc) - self.sdf_(dec)) / (2 * delta)

        normal_vec = geom_utils.normalize(normal_vec)

        return normal_vec

    @ti.func
    def collider_v(self, f, pos_world, dt):
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_world_new = geom_utils.transform_by_trans_quat_ti(pos_mesh, self.container.pos[f+1], self.container.quat[f+1])
        collider_v = (pos_world_new - pos_world) / dt
        return collider_v

    @ti.func
    def collide(self, f, pos_world, mat_v, dt, mass, f_global):
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(f, pos_world)
            if signed_dist <= 0:
                mat_v_in = mat_v
                collider_v = self.collider_v(f, pos_world, dt)

                if ti.static(self.friction > 10.0):
                    mat_v = collider_v
                else:
                    # v w.r.t collider
                    rel_v = mat_v - collider_v
                    normal_vec = self.normal(f, pos_world)
                    normal_component = rel_v.dot(normal_vec)

                    # remove inward velocity, if any
                    rel_v_t = rel_v - ti.min(normal_component, 0) * normal_vec
                    rel_v_t_norm = rel_v_t.norm()

                    # tangential component after friction (if friction exists)
                    rel_v_t_friction = rel_v_t / rel_v_t_norm * ti.max(0, rel_v_t_norm + normal_component * self.friction)

                    # tangential component after friction
                    flag = ti.cast(normal_component < 0 and rel_v_t_norm > 1e-30, DTYPE_TI)
                    rel_v_t = rel_v_t_friction * flag + rel_v_t * (1 - flag)
                    mat_v = collider_v + rel_v_t
                    
                # compute force (impulse)
                force = -(mat_v - mat_v_in) * mass
                self.container.force[f_global] += force.norm(1e-14)
                self.container.work[f_global] += ti.math.dot(-force, collider_v)

        return mat_v

    @ti.kernel
    def check_collision(self, n_particle: ti.i32, pos_world: ti.types.ndarray(), particle_sdf: ti.types.ndarray()):
        for i in range(n_particle):
            particle_sdf[i] = self.sdf(0, ti.Vector([pos_world[i, 0], pos_world[i, 1], pos_world[i, 2]], DTYPE_TI))