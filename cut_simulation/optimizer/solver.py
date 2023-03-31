import numpy as np
import taichi as ti

from cut_simulation.engine.taichi_env import TaichiEnv
from cut_simulation.optimizer.optim import Optimizer, Adam, Momentum, SGD

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum,
    'SGD': SGD
}

class Solver:
    def __init__(self, env, logger=None, cfg=None):
        self.cfg = cfg
        self.env = env
        self.target_file = env.target_file
        self.logger = logger

    def solve(self, callbacks=()):
        taichi_env = self.env.taichi_env
        init_comp_actions = self.init_composite_actions(taichi_env.agent.action_dim, self.env.horizon_action, self.cfg)

        # initialize ...
        optim = OPTIMS[self.cfg.optim.type](init_comp_actions, self.cfg.optim)
        taichi_env_state = taichi_env.get_state()

        def forward_backward(sim_state, comp_actions, horizon, horizon_action):
            if self.logger is not None:
                self.logger.reset()

            taichi_env.set_state(sim_state, grad_enabled=True)

            # forward pass
            from time import time
            t1 = time()
            actions_v = comp_actions[:-1]
            actions_p = comp_actions[-1]
            taichi_env.apply_agent_action_p(actions_p)
            
            for i in range(horizon):
                if i < horizon_action:
                    action = actions_v[i]
                else:
                    action = None
                taichi_env.step(action)

            loss_info = taichi_env.get_loss()
            if self.logger is not None:
                self.logger.step(None, None, loss_info['reward'], None, True, loss_info)
            t2 = time()

            # backward pass
            taichi_env.reset_grad()
            taichi_env.get_loss_grad()

            for i in range(horizon-1, -1, -1):
                if i < horizon_action:
                    action = actions_v[i]
                else:
                    action = None
                taichi_env.step_grad(action)

            taichi_env.apply_agent_action_p_grad(actions_p)

            t3 = time()
            print(f'=======> forward: {t2-t1:.2f}s backward: {t3-t2:.2f}s')
            loss = taichi_env.loss.loss[None]
            return loss, taichi_env.agent.get_grad(len(actions_v))

        best_action = None
        best_loss = 1e10

        comp_actions = init_comp_actions
        for iteration in range(self.cfg.n_iters):
            self.params = comp_actions.copy()
            if iteration % 10 == 0:
                self.render_comp_actions(taichi_env, taichi_env_state, comp_actions, self.env.horizon, self.env.horizon_action, iteration)

            loss, grad = forward_backward(taichi_env_state['state'], comp_actions, self.env.horizon, self.env.horizon_action)
            if loss < best_loss:
                best_loss = loss
                best_action = comp_actions.copy()
            comp_actions = optim.step(grad)
            for callback in callbacks:
                callback(self, optim, loss, grad)


        taichi_env.set_state(**taichi_env_state)
        return best_action

    @staticmethod
    def render_comp_actions(taichi_env, init_state, comp_actions, horizon, horizon_action, iteration):
        taichi_env.set_state(**init_state)
        actions_v = comp_actions[:-1]
        actions_p = comp_actions[-1]
        taichi_env.apply_agent_action_p(actions_p)
        for i in range(horizon):
            if i < horizon_action:
                action = actions_v[i]
            else:
                action = None
            taichi_env.step(action)
            taichi_env.render(iteration=iteration, t=i, save=False)

    @staticmethod
    def init_composite_actions(action_dim, horizon, cfg):
        if cfg.init_sampler == 'uniform':
            comp_actions_v = np.random.uniform(cfg.init_range.v[0], cfg.init_range.v[1], size=(horizon, action_dim))
            comp_actions_p = np.random.uniform(cfg.init_range.p[0], cfg.init_range.p[1], size=(1, action_dim))
        else:
            raise NotImplementedError

        comp_actions = np.vstack([comp_actions_v, comp_actions_p])
        return comp_actions

def solve_action(env, log_dir, logger, cfg):
    import os
    os.makedirs(log_dir, exist_ok=True)
    env.reset()

    solver = Solver(env, logger, cfg)
    action = solver.solve()

    print('Rendering optimized actions...')
    for i, act in enumerate(action):
        env.step(act)
        img = env.render(mode='human')
