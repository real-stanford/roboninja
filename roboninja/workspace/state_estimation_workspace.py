import os
import pathlib
import sys

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import tqdm

import hydra
import numpy as np
import torch
import wandb
from roboninja.workspace.base_workspace import BaseWorkspace
from roboninja.dataset.state_estimation_dataset import StateEstimationDataset
from roboninja.model.state_estimation_model import StateEstimationModel
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


class StateEstimationWorkspace(BaseWorkspace):
    include_keys = list()
    exclude_keys = list()

    def __init__(self, cfg: OmegaConf):
        # replace all macros with concrete value
        OmegaConf.resolve(cfg)
        super().__init__(cfg)

        self.model = StateEstimationModel(**self.cfg.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.cfg.optimizer)
        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.epoch = 0

    def run(self):
        # wandb
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(self.cfg, resolve=True),
            **self.cfg.logging
        )

        # set device
        device = torch.device(f'cuda:{self.cfg.gpu}')
        self.model.to(device)

        # set dataset
        dataset, data_loader = dict(), dict()
        for split in ['train', 'test']:
            dataset[split] = StateEstimationDataset(
                split=split,
                dataset_cfg=self.cfg.dataset,
                bone_cfg=self.cfg.cut_env.bone
            )
            data_loader[split] = torch.utils.data.DataLoader(dataset[split], **self.cfg.dataloader)

        while self.epoch < self.cfg.max_epoch:
            wandb_log = {'train_loss': 0, 'test_loss': 0, 'train_accuracy': 0, 'test_accuracy': 0}
            for split in ['train', 'test']:
                self.model.train(split == 'train')
                with tqdm.tqdm(data_loader[split], desc=f'{self.epoch}-{split}', leave=True) as tepoch:
                    first_batch = True
                    for batch in tepoch:
                        img_pts = batch['img_pts'].to(dtype=torch.float32, device=device)
                        img_bone = batch['img_bone'].to(dtype=torch.float32, device=device)

                        batch_size = img_pts.size(0)
                        logits = self.model(img_pts)
                        pred = torch.sigmoid(logits) > 0.5
                        loss = self.criteria(logits, img_bone) * 10

                        acc = (pred == img_bone).float().mean() 
                        wandb_log[f'{split}_loss'] += loss.item() * batch_size / len(dataset[split])
                        wandb_log[f'{split}_accuracy'] += acc * batch_size / len(dataset[split])

                        # backward
                        if split == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        
                        # regular visualization
                        if first_batch:
                            vis_img_pts = batch['img_pts'].numpy()[:, 0]
                            vis_img_bone = batch['img_bone'].numpy()[:, 0]
                            vis_pred = pred.cpu().numpy()[:, 0]
                            vis_images = np.concatenate([vis_img_pts, vis_img_bone, vis_pred], axis=1) # [B, 3N, N]
                            wandb_log[f'{split}-vis'] = wandb.Image(np.concatenate(vis_images, axis=1)[None], caption=f'{split}-visualization')
                        first_batch = False

            wandb_run.log(wandb_log, step=self.epoch)
            self.save_checkpoint()
            self.epoch += 1
        

@hydra.main(
    version_base=None,
    config_path='../config', 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = StateEstimationWorkspace(cfg)
    workspace.run()

if  __name__=='__main__':
    main()
