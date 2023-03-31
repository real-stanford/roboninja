import os
import pickle

import numpy as np

from roboninja.cutting_policy.cutting_policy_base import CuttingPolicyBase
from roboninja.utils.visualizer import Visualizer
from roboninja.utils.dynamics import normalized_action2pos, inverse_dynamics
from tqdm import trange


def calc_iou(img1, img2):
    I = np.sum(img1 * img2)
    U = np.sum(img1) + np.sum(img2) - I
    return I / U

class CuttingPolicyNN(CuttingPolicyBase):
    def __init__(self, knife_cfg, bone_cfg, expert_dir, num_train):
        super().__init__()
        self.visualizer = Visualizer()
        self.num_train = num_train
        self.knife_cfg = knife_cfg
        
        bone_cfg.mesh_root = 'data/bone_generation'

        # load bone masks (flipped)
        self.bone_mask_database = list()
        self.expert_pos_database = list()
        for bone_idx in trange(num_train):
            bone_cfg.name = f'bone_{bone_idx}'
            self.bone_mask_database.append(1 - self.visualizer.vis_bone(bone_cfg, color=0))

            data = pickle.load(open(os.path.join(expert_dir, f'expert_{bone_idx}', 'optimization.pkl'), 'rb'))
            normalized_actions = data[-1]['comp_actions']
            self.expert_pos_database.append(normalized_action2pos(normalized_actions, knife_cfg))
            
    
    def get_action(self, step_idx, current_pos):
        dist = np.linalg.norm(current_pos[:2] - self.expert_pos_database[self.nn_idx][:-1, :2], axis=1)
        min_dist_idx = np.argmin(dist)
        action = inverse_dynamics(current_pos, self.expert_pos_database[self.nn_idx][min_dist_idx + 1], self.knife_cfg)
        return action


    def update_state_estimation(self, state_estimation):
        self.state_estimation = state_estimation
        self.best_iou, self.nn_idx = 0, 0
        for idx in range(self.num_train):
            iou = calc_iou(1.0 - state_estimation, self.bone_mask_database[idx])
            if iou > self.best_iou:
                self.best_iou = iou
                self.nn_idx = idx

    def get_render_data(self):
        return {
            'state_estimation': self.state_estimation,
            'nn_idx': self.nn_idx,
            'nn_img': self.bone_mask_database[self.nn_idx]
        }