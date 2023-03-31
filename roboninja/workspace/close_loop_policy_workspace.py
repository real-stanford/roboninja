import os
import pathlib
import sys

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import cv2
import hydra
import numpy as np
import threadpoolctl
import torch
import tqdm
import wandb
from roboninja.dataset.close_loop_policy_dataset import CloseLoopPolicyDataset
from roboninja.model.close_loop_policy_model import CloseLoopPolicyModel
from roboninja.utils.visualizer import Visualizer
from roboninja.utils.dynamics import forward_dynamics
from roboninja.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

class CloseLoopPolicyWorkspace(BaseWorkspace):
    include_keys = list()
    exclude_keys = list()

    def __init__(self, cfg: OmegaConf):
        # replace all macros with concrete value
        OmegaConf.resolve(cfg)
        super().__init__(cfg)

        self.model = CloseLoopPolicyModel(**self.cfg.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.cfg.optimizer)
        self.criteria = torch.nn.MSELoss()
        self.epoch = 0

    def run(self):
        threadpoolctl.threadpool_limits(limits=4)
        
        # wandb
        wandb_run = wandb.init(
            config=OmegaConf.to_container(self.cfg, resolve=True),
            **self.cfg.logging
        )

        # set device
        device = torch.device(f'cuda:{self.cfg.gpu}')
        self.model.to(device)

        # set dataset
        dataset, data_loader = dict(), dict()
        for split in ['train', 'test']:
            dataset[split] = CloseLoopPolicyDataset(
                split=split,
                dataset_cfg=self.cfg.dataset,
                bone_cfg=self.cfg.cut_env.bone,
                knife_cfg=self.cfg.cut_env.knife
            )
            data_loader[split] = torch.utils.data.DataLoader(dataset[split], **self.cfg.dataloader)

        # visualizer
        visualizer = Visualizer()
        while self.epoch < self.cfg.max_epoch:
            wandb_log = {'train_loss': 0, 'test_loss': 0}
            for split in ['train', 'test']:
                self.model.train(split == 'train')
                with tqdm.tqdm(data_loader[split], desc=f'{self.epoch}-{split}', leave=True) as tepoch:
                    for batch in tepoch:
                        img_input = batch['img_input'].to(dtype=torch.float32, device=device)
                        aciton_gt = batch['action'].to(dtype=torch.float32, device=device)

                        batch_size = img_input.size(0)
                        action_pred = self.model(img_input)
                        loss = self.criteria(action_pred, aciton_gt) * 10
                        wandb_log[f'{split}_loss'] += loss.item() * batch_size / len(dataset[split])

                        # backward
                        if split == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                self.model.train(False)
                with torch.no_grad():
                    sample = dataset[split].sample()
                    img_bone = sample['img_bone']
                    img_bone_input = sample['img_bone_input']
                    img_bone_input_vis = np.clip(img_bone_input * 5, 0, 1)
                    gif_gt = sample['gif_gt']

                    num_vis = 5
                    all_gif_pred= list()
                    for k in range(num_vis):
                        tolerance = self.cfg.policy.tolerance_range / (num_vis - 1) * k
                        img_tolerance = np.ones_like(img_bone_input) * tolerance
                        gif_pred= list()
                        current_pos = sample['init_pos'].copy()
                        for step_idx in range(self.cfg.dataset.seq_len):
                            img_knife = visualizer.vis_knife(current_pos, color=dataset[split].knife_color)
                            img_input = np.stack([img_bone_input, img_knife, img_tolerance], axis=0)
                            
                            img_vis = visualizer.vis_knife(current_pos, img_bone, color=dataset[split].knife_color)
                            cv2.putText(
                                img_vis,
                                str(tolerance)[:8],
                                (10, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8,
                                thickness=2,
                                color=0
                            )
                            gif_pred.append(img_vis)

                            action_pred = self.model(torch.from_numpy(img_input[None]).to(dtype=torch.float32, device=device)).cpu().numpy()[0]
                            current_pos = forward_dynamics(action_pred, current_pos, self.cfg.cut_env.knife)
                        all_gif_pred.append(gif_pred)

                    vis_gif = list()
                    for step_idx in range(self.cfg.dataset.seq_len):
                        img_list = [x[step_idx] for x in all_gif_pred] + [img_bone_input_vis, gif_gt[step_idx]]
                        vis_gif.append(np.concatenate(img_list, axis=1)[None] * 255)
                    wandb_log[f'{split}-vis'] = wandb.Video(np.stack(vis_gif).astype(np.uint8), fps=10)

            wandb_run.log(wandb_log, step=self.epoch)
            self.save_checkpoint()
            if (self.epoch + 1) % 10 == 0:
                self.save_checkpoint(tag=f'epoch_{self.epoch:04}')
            self.epoch += 1
        

@hydra.main(
    version_base=None,
    config_path='../config', 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = CloseLoopPolicyWorkspace(cfg)
    workspace.run()

if  __name__=='__main__':
    main()
