import os
import pickle

import cv2
import numpy as np
import enum
from roboninja.utils.dynamics import normalized_action2pos
from roboninja.utils.geom import get_camera_matrix, project_pts_to_2d
from shapely.geometry import Polygon


class Visualizer:
    class RenderMode(enum.Enum):
        SIMPLE = 0
        TAICHI = 1
        
    def __init__(self, render_cfg=None, res=256):
        if render_cfg is None:
            self.mode = self.RenderMode.SIMPLE
            self.res = np.array([res, res])
            self.bnd = np.array([[0.38, 0.62], [0.0, 0.24]])
            self.knife_width = 0.14 * 0.3
        else:
            self.mode = self.RenderMode.TAICHI
            self.res = render_cfg.res
            self.cam_intr, self.cam_pose = get_camera_matrix(
                cam_pos=render_cfg.camera_pos,
                cam_size=render_cfg.res,
                cam_fov=render_cfg.fov / 180 * np.pi,
                cam_lookat=render_cfg.camera_lookat
            )
            self.cam_view = np.linalg.inv(self.cam_pose)
            self.dim_z = 0.47

    def empty_image(self, num_dim=3):
        if num_dim == 3:
            return np.ones([self.res[0] ,self.res[1], 3])
        elif num_dim == 2:
            return np.ones([self.res[0] ,self.res[1]])
        else:
            raise NotImplementedError(f'num_dim doesn\'t support {num_dim}')

    def wrd2pix(self, wrd):
        if self.mode == self.RenderMode.SIMPLE:
            pix = self.res - 1 - (wrd - self.bnd[:, 0]) / (self.bnd[:, 1] - self.bnd[:, 0]) * (self.res - 1)
            return pix.astype(int)
        elif self.mode == self.RenderMode.TAICHI:
            wrd_copy = np.array(wrd)
            if len(wrd.shape) == 1: wrd_copy = wrd_copy[None]
            wrd_copy = np.concatenate([wrd_copy[:, :2], np.ones([len(wrd_copy), 1]) * self.dim_z], axis=1)
            pix = project_pts_to_2d(wrd_copy, self.cam_view, self.cam_intr)[:, :2].astype(int)
            pix = pix[:, [1, 0]]
            pix[:, 0] = self.res[0] - 1 - pix[:, 0]
            if len(wrd.shape) == 1: pix = pix[0]
            return pix 


    def vis_bone(self, bone_cfg, color=(0, 0, 0), return_coords=False, return_wrd=False):
        bone_info_path = os.path.join(bone_cfg.mesh_root, 'info', f'{bone_cfg.name}.pkl')
        bone_info = pickle.load(open(bone_info_path, 'rb'))
        points = np.stack([bone_info['normalized_x'], bone_info['normalized_y']], axis=1)
        points *= np.array([bone_cfg.scale[1], bone_cfg.scale[0]])
        theta = bone_cfg.euler[2] / 180 * np.pi
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points = (rot_mat @ points.T).T + np.array(bone_cfg.pos[:2])
        coords = self.wrd2pix(points)
        img_bone = np.ones(self.res) if isinstance(color, int) else np.ones([*self.res, len(color)])
        cv2.fillPoly(img_bone, [coords], color=color)
        if return_coords:
            assert not return_wrd
            return img_bone, coords
        elif return_wrd:
            return img_bone, points
        else:
            return img_bone
        
    def vis_pos(self, pos_seq, img=None, color=(0.5, 0.5, 0.5), thickness=1, aa=False):
        '''
        pos_seq: [N, 3]
            - dim0: N steps
            - dim1: [x, y, rot]
        '''
        img_pos = self.empty_image(num_dim=3 if isinstance(color, tuple) else 2) if img is None else img.copy()
        coords = self.wrd2pix(pos_seq[:, :2])
        for i in range(len(pos_seq) - 1):
            if aa:
                cv2.line(img_pos, tuple(coords[i]), tuple(coords[i+1]), color=color, thickness=thickness, lineType=cv2.LINE_AA)
            else:
                cv2.line(img_pos, tuple(coords[i]), tuple(coords[i+1]), color=color, thickness=thickness)
        
        return img_pos

    def vis_action(self, action, knife_cfg, **kwargs):
        '''
        action: [N + 1, 3]
            - dim0: a_v x N + a_p
            - dim1: [rot_v, rot_k, 0] or [x, y, rot_k]. normalized!
        '''
        pos = normalized_action2pos(action, knife_cfg)
        return self.vis_pos(pos, **kwargs)
        
    def vis_knife(self, pos, img=None, color=(0.5, 0.5, 0.5), thickness=2):
        img_knife = self.empty_image(num_dim=3 if isinstance(color, tuple) else 2) if img is None else img.copy()
        p0 = pos[:2]
        p1 = p0 + np.array([-np.sin(pos[2]), np.cos(pos[2])]) * self.knife_width
        cv2.arrowedLine(img_knife, tuple(self.wrd2pix(p1)), tuple(self.wrd2pix(p0)), color, thickness=thickness)
        return img_knife

    def vis_knife_gif(self, pos_seq, **kwargs):
        gif_images = list()
        for pos in pos_seq:
            gif_images.append(self.vis_knife(pos, **kwargs))
        return gif_images

    
    def get_max_vol(self, bone_cfg, base_eval_env, init_pos):
        bone_info_path = os.path.join(bone_cfg.mesh_root, 'info', f'{bone_cfg.name}.pkl')
        bone_info = pickle.load(open(bone_info_path, 'rb'))
        points = np.stack([bone_info['normalized_x'], bone_info['normalized_y']], axis=1)
        points *= np.array([bone_cfg.scale[1], bone_cfg.scale[0]])
        theta = bone_cfg.euler[2] / 180 * np.pi
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        wrd_pts = (rot_mat @ points.T).T + np.array(bone_cfg.pos[:2])

        bone_polygon = Polygon(wrd_pts)
        cut_box_polygon = Polygon([[0.4, base_eval_env.min_height], [0.4, 0.2], [init_pos[0], 0.2], [init_pos[0], base_eval_env.min_height]])
        base_box_polygon = Polygon([[0.6, base_eval_env.min_height], [0.6, 0.2], [init_pos[0], 0.2], [init_pos[0], base_eval_env.min_height]])

        max_cut_polygon = cut_box_polygon - bone_polygon
        remain_polygon = base_box_polygon - bone_polygon
        if not isinstance(remain_polygon, Polygon):
            for p_idx in range(1, len(remain_polygon.geoms)):
                p = remain_polygon.geoms[p_idx]
                assert(p.area < remain_polygon.geoms[0].area)
                max_cut_polygon = max_cut_polygon.union(p)
        max_vol = max_cut_polygon.area

        return max_vol