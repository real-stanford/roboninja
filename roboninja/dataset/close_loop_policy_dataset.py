import copy
import os
import pickle

import numpy as np
import torch
from roboninja.utils.visualizer import Visualizer
from roboninja.utils.dynamics import normalized_action2pos, forward_dynamics_inv, decode_action
from roboninja.utils.geom import binary2sdf
import tqdm
import threadpoolctl


class CloseLoopPolicyDataset(torch.utils.data.Dataset):
    def __init__(self, split, dataset_cfg, bone_cfg, knife_cfg):
        self.idx_list = list(range(dataset_cfg.train_num)) if split == 'train' \
            else list(range(dataset_cfg.train_num, dataset_cfg.train_num + dataset_cfg.test_num))
        self.dataset_cfg = dataset_cfg
        self.bone_cfg = bone_cfg
        self.knife_cfg = knife_cfg
        self.visualizer = Visualizer()
        self.knife_color = 0.5

        # replace bone dir
        if self.dataset_cfg.dir is not None:
            self.bone_cfg.mesh_root = self.dataset_cfg.dir

        self.img_bone_cache = dict()
        for bone_idx in tqdm.tqdm(self.idx_list):
            self.img_bone_cache[bone_idx] = self.get_img_bone(bone_idx)

    def __len__(self):
        return len(self.idx_list) * self.dataset_cfg.seq_len

    def get_img_bone(self, bone_idx):
        if bone_idx in self.img_bone_cache:
            return self.img_bone_cache[bone_idx]
        bone_cfg = copy.deepcopy(self.bone_cfg)
        bone_cfg.name = f'bone_{bone_idx}'
        img_bone = self.visualizer.vis_bone(bone_cfg, color=0, return_coords=False)
        img_bone_input = img_bone if not self.dataset_cfg.sdf else binary2sdf(img_bone)
        return img_bone, img_bone_input

    def __getitem__(self, idx):
        threadpoolctl.threadpool_limits(limits=1)

        bone_idx = self.idx_list[idx // self.dataset_cfg.seq_len]
        step_idx = idx % self.dataset_cfg.seq_len

        # get bone state
        img_bone, img_bone_input = self.get_img_bone(bone_idx)

        # load expert
        data = pickle.load(open(os.path.join(self.dataset_cfg.expert_dir, f'expert_{bone_idx}', 'optimization.pkl'), 'rb'))
        epoch_id = -(np.random.choice(self.dataset_cfg.epoch_range) + 1)
        info = data[epoch_id]
        normalized_actions = info['comp_actions']
        actions = decode_action(normalized_actions, self.knife_cfg)
        wrd_pos_seq = normalized_action2pos(normalized_actions, self.knife_cfg)

        # get raw step info
        current_pos = wrd_pos_seq[step_idx]
        next_pos = wrd_pos_seq[step_idx + 1]
        action = actions[step_idx][:2]

        # data augmentation 1: sample tolerance
        tolerance = np.random.rand() * self.dataset_cfg.tolerance_range
        current_pos[0] -= tolerance
        next_pos[0] -= tolerance
        img_tolerance = np.ones_like(img_bone) * tolerance

        # dsata augnemtation 2: action funnel
        noise_k = np.random.normal(0, self.dataset_cfg.funnel_std_k)
        noise_v = np.random.normal(0, self.dataset_cfg.funnel_std_v)
        action[1] += noise_v
        current_pos = forward_dynamics_inv([action[0] + noise_k, action[1]], next_pos, self.knife_cfg)

        # visualize knife with tolerance
        img_knife = self.visualizer.vis_knife(current_pos, color=self.knife_color)

        img_input = np.stack([img_bone_input, img_knife, img_tolerance], axis=0)

        return_data = {
            'img_input': img_input,       # [3, 256, 256]
            'action': action,             # [2]
        }
        return return_data

    def sample(self):
        threadpoolctl.threadpool_limits(limits=1)
        
        bone_idx = np.random.choice(self.idx_list)
        img_bone, img_bone_input = self.get_img_bone(bone_idx)

        # load expert
        data = pickle.load(open(os.path.join(self.dataset_cfg.expert_dir, f'expert_{bone_idx}', 'optimization.pkl'), 'rb'))
        info = data[-1]
        normalized_actions = info['comp_actions']
        actions = decode_action(normalized_actions, self.knife_cfg)
        wrd_pos_seq = normalized_action2pos(normalized_actions, self.knife_cfg)
        init_pos = wrd_pos_seq[0]

        gif_gt = list()
        for pos in wrd_pos_seq:
            gif_gt.append(self.visualizer.vis_knife(pos, img_bone, self.knife_color))

        return_data = {
            'img_bone': img_bone,               # [256, 256]
            'img_bone_input': img_bone_input,   # [256, 256]
            'actions': actions[:-1],            # [60, 2]
            'init_pos': init_pos,               # [3]
            'gif_gt': gif_gt                    # list of img
        }
        return return_data

