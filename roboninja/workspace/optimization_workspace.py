import os
import pathlib
import sys

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import pickle
import time

import hydra
import numpy as np
import wandb
from colorama import Fore, Style
from cut_simulation.configs.macros import *
from cut_simulation.optimizer.optim import SGD, Adam, Momentum
from roboninja.env.tc_env import get_cut_env
from roboninja.utils.visualizer import Visualizer
from roboninja.utils.dynamics import normalized_action2pos
from roboninja.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)
OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum,
    'SGD': SGD
}

class OptimizationWorkspace(BaseWorkspace):
    include_keys = list()
    exclude_keys = list()

    def __init__(self, cfg: OmegaConf):
        # replace all macros with concrete value
        OmegaConf.resolve(cfg)
        super().__init__(cfg)

    def forward(self, taichi_env, init_state, comp_actions, knife, grad_enabled, render, **kwargs):
        start_timestep = time.time()
        taichi_env.set_state(init_state, grad_enabled=grad_enabled)

        actions_v = comp_actions[:-1]
        action_p = comp_actions[-1]
        horizon_action = len(actions_v)
        taichi_env.apply_agent_action_p(action_p)
        render_images = list()

        if render: render_images.append(taichi_env.render())
        for i in range(horizon_action):
            taichi_env.step(actions_v[i])
            if render: render_images.append(taichi_env.render())
        loss = taichi_env.get_loss()

        forward_output = {
            'loss': loss,
            'forward_duration': time.time() - start_timestep,
            'render_images': render_images
        }

        return forward_output

    def backward(self, taichi_env, comp_actions, **kwargs):
        start_timestep = time.time()
        taichi_env.reset_grad()
        taichi_env.get_loss_grad()

        actions_v = comp_actions[:-1]
        action_p = comp_actions[-1]
        horizon_action = len(actions_v)

        for i in range(horizon_action-1, -1, -1):
            taichi_env.step_grad(actions_v[i])
        grad = taichi_env.agent.get_grad(len(actions_v))
        grad[-1][1:] = 0

        backward_output = {
            'grad': grad,
            'backward_duration': time.time() - start_timestep,
        }

        return backward_output
    
    def log(self, taichi_env, knife, iter, comp_actions, loss, grad, forward_duration, backward_duration, wandb_run, img_traj, gif_traj, render_images, **kwargs):
        horizon_action = len(comp_actions) - 1
        final_rot = knife.theta_k[horizon_action]

        loss_print = f'{Fore.CYAN}loss={Fore.WHITE}{loss["loss"]:.4f}'
        for loss_type in taichi_env.loss.weights.keys():
            weighted_loss = loss[f'{loss_type}_loss'] * taichi_env.loss.weights[loss_type]
            loss_print += f' {Fore.CYAN}{loss_type}={Fore.WHITE}{weighted_loss:.3f}'

        print(
            f'{Fore.WHITE}>{self.cfg.cut_env.bone.name}-{iter:3d}  ' +
            f'{Fore.RED}{forward_duration:.1f}+{backward_duration:.1f} ' +
            f'{Fore.GREEN}rot={Fore.YELLOW}{final_rot:.3f} ' +
            loss_print + f'{Style.RESET_ALL}'
        )

        if wandb_run is not None:
            log_info = {
                'action/final_rot': final_rot,
                'grad/theta_k': np.mean(grad[:-1, 0]),
                'grad/theta_v': np.mean(grad[:-1, 1]),
                'grad/init_x': np.mean(grad[-1, 0]),
                'trajectory': wandb.Image(img_traj),
                'trajectory_video': wandb.Video((np.stack(gif_traj).transpose([0,3,1,2]) * 255).astype(np.uint8), fps=10),
                'misc/x_bnd': taichi_env.loss.x_bnd[None],
            }
            for loss_type in loss.keys():
                log_info[f'loss/{loss_type}'] = loss[loss_type]
            if len(render_images) > 0:
                log_info['video'] = wandb.Video(np.stack(render_images).transpose([0,3,1,2]), fps=10)
            wandb_run.log(log_info)


    def run(self):
        # set some constant
        horizon_action = self.cfg.horizon_action
        render_gap = self.cfg.render_gap

        if render_gap == -1:
           self.cfg.cut_env.render = None

        output_dir = self.cfg.output_dir if 'output_dir' in self.cfg else self.output_dir
        save_path = os.path.join(output_dir, 'optimization.pkl')
        if os.path.exists(save_path):
            return

        taichi_env = get_cut_env(self.cfg.cut_env)
        knife = taichi_env.agent.effectors[0]

        # wandb
        wandb_run = wandb.init(
            config=OmegaConf.to_container(self.cfg, resolve=True),
            **self.cfg.logging
        ) if self.cfg.log_wandb else None

        # set visualizer
        visualizer = Visualizer()
        img_bone = visualizer.vis_bone(bone_cfg=self.cfg.cut_env.bone)

        # set init actions
        init_action_p = np.array(self.cfg.init_action_p)
        init_action_v = np.array([self.cfg.init_action_v] * horizon_action)
        init_action_v[:int(horizon_action * 0.4), 1] = -0.3
        init_actions = np.concatenate([init_action_v, init_action_p[None]], axis=0)

        # set learning_rate and optimizer
        lr_action_p = np.ones_like(init_action_p) * self.cfg.optim.lr_action_p
        lr_action_v = np.ones_like(init_action_v) * self.cfg.optim.lr_action_v
        lr = np.concatenate([lr_action_v, lr_action_p[None]], axis=0)
        optim = OPTIMS[self.cfg.optim.type](init_actions.copy(), lr, self.cfg.optim)

        init_state = taichi_env.get_state()['state']
        current_actions = init_actions

        save_info = list()

        for iteration in range(self.cfg.n_iters):
            render = render_gap != -1 and (iteration == 0 or (iteration + 1) % render_gap == 0)
            kwargs = {
                'iter': iteration,
                'taichi_env': taichi_env,
                'init_state': init_state,
                'comp_actions': current_actions,
                'knife': knife,
                'grad_enabled': True,
                'render': render,
                'wandb_run': wandb_run
            }
            # forward pass
            forward_output = self.forward(**kwargs)
            kwargs = kwargs | forward_output

            # backward_pass
            backward_output = self.backward(taichi_env, current_actions)
            kwargs = kwargs | backward_output

            # visualize trajectory
            pos_seq = normalized_action2pos(current_actions, self.cfg.cut_env.knife)
            img_traj = visualizer.vis_pos(pos_seq, img_bone)
            gif_traj = visualizer.vis_knife_gif(pos_seq, img=img_bone)
            kwargs['img_traj'] = img_traj
            kwargs['gif_traj'] = gif_traj

            # log result
            self.log(**kwargs)
            
            # step optimization
            grad = kwargs['grad']
            assert not np.isnan(np.mean(grad))
            current_actions = optim.step(grad)
            
            # select info
            cur_save_info = dict()
            for key in self.cfg.save_info_keys:
                cur_save_info[key] = copy.deepcopy(kwargs[key])
            save_info.append(cur_save_info)

        # save results
        pickle.dump(save_info, open(save_path, 'wb'))


@hydra.main(
    version_base=None,
    config_path='../config', 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = OptimizationWorkspace(cfg)
    workspace.run()

if  __name__=='__main__':
    main()
