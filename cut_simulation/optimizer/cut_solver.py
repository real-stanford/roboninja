from time import time
import numpy as np
import taichi as ti
import os
from cut_simulation.engine.taichi_env import TaichiEnv
from cut_simulation.optimizer.optim import SGD, Adam, Momentum, Optimizer
from colorama import Fore, Style
from tqdm import trange
import pickle
import numpy as np
from cut_simulation.configs.macros import *
import cv2
from cut_simulation.utils.visualize import mkdir

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum,
    'SGD': SGD
}


class CutSolver:
    def __init__(self, taichi_env: TaichiEnv, cfg, exp_name, render_gap=10):

        self.cfg = cfg
        self.horizon_action = cfg.horizon_action
        self.taichi_env = taichi_env
        self.knife = taichi_env.agent.effectors[0]
        self.bone = taichi_env.statics[2]

        self.exp_name = exp_name
        self.render_gap = render_gap

        self.res = (256, 256)
        self.bnd = np.array([[0.38, 0.62], [0.0, 0.24]])
        x = np.linspace(self.bnd[0, 0], self.bnd[0, 1], self.res[0])
        y = np.linspace(self.bnd[1, 0], self.bnd[1, 1], self.res[1])
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)
        zv = np.ones_like(xv) * 0.5
        particles = np.stack([xv, yv, zv], axis=1)
        n_particle = len(particles)
        sdf = np.zeros([n_particle])
        self.bone.check_collision(n_particle, particles, sdf)
        sdf_img = sdf.reshape(self.res)
        self.vis_background = np.zeros_like(sdf_img)
        self.vis_background[sdf_img > 0] = 1.0

        self.last_iter = 30


    def vis_trajectory(self, pos_global):
        self.knife.read_pos_global(self.horizon_action + 1, pos_global)
        vis_img = self.vis_background.copy()
        coords = (pos_global[:, :2] - self.bnd[:, 0]) / ( self.bnd[:, 1] -  self.bnd[:, 0]) * (np.array(self.res) - 1)
        coords = coords.astype(int)
        for i in range(self.horizon_action):
            cv2.line(vis_img, coords[i], coords[i+1], 0.5, 1)
        vis_img = np.flip(vis_img, [0, 1])
        return vis_img


    def solve(self, exp_info='', log_wandb=True):    
        # Set wandb
        if log_wandb:
            import wandb
            wandb.init(
                project='cut_expert',
                name=self.exp_name
            )

        taichi_env = self.taichi_env
        knife = self.knife
        init_position = np.array([-0.8, 0.215, 0.5])
        init_actions = np.concatenate([np.zeros([self.horizon_action, 3]), init_position[None]], axis=0)

        # initialize ...
        lr = np.ones_like(init_actions) * self.cfg.optim.lr

        lr[-1] *= 4 # TODO hacky!!! position has a larger lr
        
        optim = OPTIMS[self.cfg.optim.type](init_actions, lr, self.cfg.optim)
        taichi_env_state = taichi_env.get_state()

        def forward_backward(iter, sim_state, comp_actions, horizon_action, render=False):
            taichi_env.set_state(sim_state, grad_enabled=True)

            # forward pass
            t1 = time()

            actions_v = comp_actions[:-1]
            actions_p = comp_actions[-1]
            
            taichi_env.apply_agent_action_p(actions_p)

            images = list()
            if render: images.append(taichi_env.render())
            for i in range(horizon_action):
                taichi_env.step(actions_v[i])
                if render: images.append(taichi_env.render())
        
            pos_global = np.zeros((self.horizon_action + 1, 3), dtype=DTYPE_NP)
            trajectory_img = self.vis_trajectory(pos_global)

            loss_info = taichi_env.get_loss()
            loss = loss_info['loss']

            t2 = time()

            # backward pass
            taichi_env.reset_grad()
            taichi_env.get_loss_grad()

            for i in range(horizon_action-1, -1, -1):
                taichi_env.step_grad(actions_v[i])
            taichi_env.apply_agent_action_p_grad(actions_p)
            t3 = time()
            grad = taichi_env.agent.get_grad(len(actions_v))
            grad[-1][1:] = 0
            assert np.sum(np.abs(grad[:-1, 2])) < 1e-20

            final_rot = knife.theta_k[self.horizon_action]
            start_pos = pos_global[0][0]
            final_pos = pos_global[self.horizon_action][0]

            loss_print = f'{Fore.CYAN}loss={Fore.WHITE}{loss:.4f}'
            for loss_type in ['cut', 'collision', 'move', 'rotation', 'force']:
                weighted_loss = loss_info[f'{loss_type}_loss'] * taichi_env.loss.weights[loss_type]
                loss_print += f' {Fore.CYAN}{loss_type}={Fore.WHITE}{weighted_loss:.3f}'

            print(
                f'{Fore.WHITE}>{exp_info}{iter:3d}  ' +
                f'{Fore.RED}{t2-t1:.1f}+{t3-t2:.1f} ' +
                f'{Fore.GREEN}rot={Fore.YELLOW}{final_rot:.3f} ' +
                f'{Fore.GREEN}pos={Fore.YELLOW}({start_pos:.3f}, {final_pos:.3f}) ' +
                loss_print + f'{Style.RESET_ALL}'
            )

            if log_wandb:
                log_info = {
                    'action/final_rot': final_rot,
                    'action/start_pos': start_pos,
                    'action/final_pos': final_pos,
                    'grad/theta_k': np.mean(grad[:-1, 0]),
                    'grad/theta_v': np.mean(grad[:-1, 1]),
                    'grad/init_x': np.mean(grad[-1, 0]),
                    'trajectory': wandb.Image(trajectory_img),
                    'misc/x_bnd': taichi_env.loss.x_bnd[None],
                    'misc/x_max': np.max(pos_global[20:40, 0])
                }
                for loss_type in loss_info.keys():
                    log_info[f'loss/{loss_type}'] = loss_info[loss_type]
                if render:
                    log_info['video'] = wandb.Video(np.stack(images).transpose([0,3,1,2]), fps=10)
                wandb.log(log_info)

            pickle.dump([current_actions, loss_info, grad, pos_global, trajectory_img], open(f'log/{self.exp_name}/info-{iter}.pkl', 'wb'))

            # curriculum training
            # if np.max(pos_global[20:40, 0]) > taichi_env.loss.x_bnd[None] - 0.01 and iter - self.last_iter > 20:
            #     taichi_env.loss.x_bnd[None] += 0.01
            #     self.last_iter = iter

            return loss, grad

        best_action = None
        best_loss = 1e10

        current_actions = init_actions
        mkdir(f'log/{self.exp_name}')
        for iteration in range(self.cfg.n_iters):
            render = self.render_gap != -1 and (iteration == 0 or (iteration + 1) % self.render_gap == 0)
            loss, grad = forward_backward(iteration, taichi_env_state['state'], current_actions, self.horizon_action, render)

            if np.isnan(np.mean(grad)):
                print('grad is nan. terminate!')
                return
                
            if loss < best_loss:
                best_loss = loss
                best_action = current_actions.copy()

            current_actions = optim.step(grad)
            
            current_actions[-1, 0] = min(-0.5, current_actions[-1, 0]) # TODO
            if iteration > 100:
                # push the start position to -0.5 (normalized value)
                current_actions[-1, 0] += (-0.5 - current_actions[-1, 0]) * 0.2


        taichi_env.set_state(**taichi_env_state)
        return best_action