import copy

import numpy as np
import torch
from roboninja.utils.visualizer import Visualizer


class StateEstimationDataset(torch.utils.data.Dataset):
    def __init__(self, split, dataset_cfg, bone_cfg):
        self.idx_list = list(range(dataset_cfg.train_num)) if split == 'train' \
            else list(range(dataset_cfg.train_num, dataset_cfg.train_num + dataset_cfg.test_num))
        self.dataset_cfg = dataset_cfg
        self.bone_cfg = bone_cfg

        self.visualizer = Visualizer()

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        bone_idx = self.idx_list[idx]
        bone_cfg = copy.deepcopy(self.bone_cfg)
        if self.dataset_cfg.dir is not None:
            bone_cfg.mesh_root = self.dataset_cfg.dir
        bone_cfg.name = f'bone_{bone_idx}'
        img_bone, coords = self.visualizer.vis_bone(bone_cfg, color=0, return_coords=True)
        selective_coords = coords[15:-15]

        num_pts = np.random.randint(self.dataset_cfg.max_num_pts)
        selected_idx = np.random.choice(len(selective_coords), num_pts)
        coords = coords[selected_idx]

        img_pts = self.visualizer.empty_image(num_dim=2)
        for i, c in enumerate(coords):
            img_pts[c[1], c[0]] = 0
        
        return_data = {
            'img_bone': img_bone[None],     # [1, 256, 256]
            'img_pts': img_pts[None],       # [1, 256, 256]
        }
        return return_data
