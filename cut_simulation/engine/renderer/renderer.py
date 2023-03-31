"""
Adapted from taichi element
"""
import os
import taichi as ti
import numpy as np
import time
from scipy import ndimage
import trimesh
import skimage
from cut_simulation.configs.macros import *

@ti.data_oriented
class Renderer:
    def __init__(self, 
        res=(960, 960),
        camera_pos=(0.5, 2.5, 3.5),
        camera_lookat=(0.5, 0.5, 0.5),
        fov=30,
        mode='human',
        particle_radius=0.004,
        lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        show_bone=False
    ):
        self.res = res
        self.mode = mode
        self.show_window = mode == 'human'
        self.window = ti.ui.Window("cut_simulation", tuple(self.res), vsync=True, show_window=self.show_window)
        self.camera_pos = np.array(camera_pos)
        self.camera_lookat = np.array(camera_lookat)
        self.camera_vec = self.camera_pos - self.camera_lookat
        self.camera_init_xz_rad = np.arctan2(self.camera_vec[0], self.camera_vec[2])
        self.lights = []

        for light in lights:
            self.add_light(light['pos'], light['color'])

        self.fov = fov
        self.particle_radius = particle_radius
        self.frame = ti.Vector.field(3, dtype=ti.f32, shape=(9,))
        self.frames = [ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,))]
        self.color_target = None
        self.show_bone = show_bone

    def add_light(self, pos, color=(0.5, 0.5, 0.5)):
        light = {
            'pos': pos,
            'color': color
        }
        self.lights.append(light)

    def build(self, n_particles):
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1,1,1))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(*self.camera_pos)
        self.camera.lookat(*self.camera_lookat)
        self.camera.fov(self.fov)

        for i in range(200):
            self.frames[0][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[1][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([0., 1., 0.])
            self.frames[2][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[3][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([-1., 0., 0.])
            self.frames[4][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([0., -1., 0.])
            self.frames[5][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([0., 0., -1.])
            self.frames[6][i] = ti.Vector([0., 1., 0.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[7][i] = ti.Vector([0., 1., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[8][i] = ti.Vector([1., 0., 0.]) + i/200 * ti.Vector([0., 1., 0.])
            self.frames[9][i] = ti.Vector([1., 0., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[10][i] = ti.Vector([0., 0., 1.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[11][i] = ti.Vector([0., 0., 1.]) + i/200 * ti.Vector([0., 1., 0.])

        self.particles_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(50000))
        self.particles_faces = ti.field(dtype=ti.i32, shape=(200000))
        self.particles_colors = ti.Vector.field(4, dtype=ti.f32, shape=(50000))

        self.n_particles = n_particles
        self.selected_particles  = ti.Vector.field(3, dtype=ti.f32, shape=(n_particles))


    def update_camera(self, t=None, rotate=False):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)

        speed = 1e-2
        if self.window.is_pressed(ti.ui.UP):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.DOWN):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) - camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) - camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('u'):
            camera_dir = np.array([0, 1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('i'):
            camera_dir = np.array([0, -1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)

        # rotate
        if rotate and t is not None:
            speed = 7.5e-4
            xz_radius = np.linalg.norm([self.camera_vec[0], self.camera_vec[2]])
            rad = speed * np.pi * t + self.camera_init_xz_rad
            x = xz_radius * np.sin(rad)
            z = xz_radius * np.cos(rad)
            new_camera_pos = np.array([
                    x + self.camera_lookat[0],
                    self.camera_pos[1],
                    z + self.camera_lookat[2]]) 
            self.camera.position(*new_camera_pos)

        self.scene.set_camera(self.camera)

    @ti.kernel
    def set_faces(self, faces_np: ti.types.ndarray(), n_faces: ti.i32):
        for i in range(200000):
            if i < n_faces:
                self.particles_faces[i] = faces_np[i]
    @ti.kernel
    def set_vertices(self, vertices_np: ti.types.ndarray(), n_vertices: ti.i32):
        for i in range(50000):
            if i < n_vertices:
                for j in ti.static(range(3)):
                    self.particles_vertices[i][j] = vertices_np[i, j]
    @ti.kernel
    def set_colors(self, colors_np: ti.types.ndarray(), n_colors: ti.i32):
        for i in range(50000):
            if i < n_colors:
                for j in ti.static(range(4)):
                    self.particles_colors[i][j] = colors_np[i, j]
    @ti.kernel
    def set_particles(self, particles_np: ti.types.ndarray(), n_particles: ti.i32):
        self.selected_particles.fill(0)
        for i in range(n_particles):
            for j in ti.static(range(3)):
                self.selected_particles[i][j] = particles_np[i, j]
        

    def render_frame(self, frame_state, mat_particles, color_particles, agent, x_target, statics, iteration=None, t=None, save=False, meshing=False):
        self.update_camera(t)

        # x_particles = frame_state.x
        # self.scene.particles(x_particles, per_vertex_color=color_particles, radius=self.particle_radius)

        # pos = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        # pos.fill(0.5)
        # self.scene.particles(pos, color=(0.5, 0.5, 0.5, 1), radius=0.01)

        # reference frame
        # for i in range(12):
        #     self.scene.particles(self.frames[i], color=COLOR[FRAME], radius=self.particle_radius*0.5)
            
        # statics
        if len(statics) != 0:
            for static in statics:
                if static.render_order == 'before':
                    self.scene.mesh(static.vertices, static.faces, per_vertex_color=static.colors)
        
        # effectors
        if agent is not None and agent.n_effectors != 0:
            for effector in agent.effectors:
                if effector.mesh.render_order == 'before':
                    if effector.mesh is not None:
                        self.scene.mesh(effector.mesh.vertices, effector.mesh.faces, per_vertex_color=effector.mesh.colors)
                    # self.scene.particles(effector.latest_pos, color=COLOR[EFFECTOR], radius=self.particle_radius*2)

        if frame_state is not None:
            x_particles = frame_state.x
            used_particles = frame_state.used
            # particles
            # meshing = True
            if meshing:
                import open3d as o3d
                mats = np.unique(mat_particles)
                pcd_particles = x_particles.to_numpy()
                used = used_particles.to_numpy()
                vertices = []
                colors = []
                faces = []

                vertex_count = 0
                for mat in mats:
                    pcd_indices_mat = np.where(
                            np.logical_and(mat_particles == mat, used))[0]
                    pcd_mat = pcd_particles[pcd_indices_mat]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pcd_mat)
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)

                    o3d_voxels = voxel_grid.get_voxels()
                    o3d_voxels_indices = np.stack(list(vx.grid_index for vx in o3d_voxels))

                    padding = 2
                    indices_shape = o3d_voxels_indices.max(0) + 1
                    voxel_grid_shape = indices_shape + 2 * padding
                    voxels_np = np.zeros(voxel_grid_shape)
                    o3d_voxels_indices += padding
                    voxels_np[o3d_voxels_indices[:, 0], o3d_voxels_indices[:, 1], o3d_voxels_indices[:, 2]] = 1
                    voxels_np = ndimage.binary_dilation(voxels_np)
                    voxels_np = ndimage.binary_erosion(voxels_np)

                    voxels_np = voxels_np * -2 + 1
                    vertices_mat, faces_mat, normals, _ = skimage.measure.marching_cubes(voxels_np, level=0)

                    mesh_mat = trimesh.Trimesh(vertices=vertices_mat, faces=faces_mat, vertex_normals=normals)
                    if mat == COFFEE:
                        trimesh.smoothing.filter_laplacian(mesh_mat)

                    voxel_grid_max_bound = voxel_grid.get_max_bound()
                    voxel_grid_min_bound = voxel_grid.get_min_bound()
                    voxel_grid_extent = voxel_grid_max_bound - voxel_grid_min_bound

                    vertices_mat = (np.array(mesh_mat.vertices) - padding) / indices_shape * voxel_grid_extent + voxel_grid_min_bound
                    faces_mat = np.array(mesh_mat.faces)
                    faces_mat += vertex_count
                    vertex_count += vertices_mat.shape[0]
                    vertices.append(vertices_mat)
                    colors.append(np.tile(COLOR[mat], [len(vertices_mat), 1]))
                    faces.append(faces_mat)
                vertices = np.concatenate(vertices).astype(np.float32)
                colors = np.concatenate(colors).astype(np.float32)
                faces = np.concatenate(faces).flatten().astype(np.int32)

                self.set_faces(faces, len(faces))
                self.set_vertices(vertices, len(vertices))
                self.set_colors(colors, len(colors))
                self.scene.mesh(self.particles_vertices, self.particles_faces, per_vertex_color=self.particles_colors, index_count=len(faces))
            else:
                if self.show_bone:
                    x_particles_np = x_particles.to_numpy()
                    select_idx = x_particles_np[:, 2] > 0.47
                    x_particles_np = x_particles_np[select_idx]
                    self.set_particles(x_particles_np, len(x_particles_np))
                    self.scene.particles(self.selected_particles, per_vertex_color=color_particles, radius=self.particle_radius)
                else:
                    self.scene.particles(x_particles, per_vertex_color=color_particles, radius=self.particle_radius)

            # statics
            if len(statics) != 0:
                for static in statics:
                    if static.render_order == 'after':
                        self.scene.mesh(static.vertices, static.faces, per_vertex_color=static.colors)

            # effectors
            if agent is not None and agent.n_effectors != 0:
                for effector in agent.effectors:
                    if effector.mesh.render_order == 'after':
                        if effector.mesh is not None:
                            self.scene.mesh(effector.mesh.vertices, effector.mesh.faces, per_vertex_color=effector.mesh.colors)
                        # self.scene.particles(effector.latest_pos, color=COLOR[EFFECTOR], radius=self.particle_radius*2)

            # target particles
            if x_target is not None:
                if self.color_target is None:
                    self.color_target = ti.Vector.field(4, ti.f32, x_target.shape)
                    self.color_target.from_numpy(np.repeat(np.array([COLOR[TARGET]]).astype(np.float32), x_target.shape[0], axis=0))
                self.scene.particles(x_target, per_vertex_color=self.color_target, radius=self.particle_radius)

        for light in self.lights:
            self.scene.point_light(pos=light['pos'], color=light['color'])

        self.canvas.scene(self.scene)

        if False:
            self.window.GUI.begin("Camera", 0.05, 0.1, 0.2, 0.15)
            self.window.GUI.text(f'pos:    {self.camera.curr_position[0]:.2f}, {self.camera.curr_position[1]:.2f}, {self.camera.curr_position[2]:.2f}')
            self.window.GUI.text(f'lookat: {self.camera.curr_lookat[0]:.2f}, {self.camera.curr_lookat[1]:.2f}, {self.camera.curr_lookat[2]:.2f}')
            self.window.GUI.end()

        # if save:
        #     os.makedirs(f'tmp/iter_{iteration}', exist_ok=True)
        #     self.window.save_image(f'tmp/iter_{iteration}/{t:04d}.png')

        # if self.mode == 'human':
        #     self.window.show()
        # else:
        #     assert False


        img = self.window.get_image_buffer_as_numpy()[:, :, :3]
        img = (np.flip(img.transpose([1, 0, 2]), 0) * 255.0).astype(np.uint8)

        return img



