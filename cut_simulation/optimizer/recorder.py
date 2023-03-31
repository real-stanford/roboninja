import os
import numpy as np
import taichi as ti
import pickle as pkl

from .optim import Optimizer, Adam, Momentum, SGD

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum,
    'SGD': SGD
}

class Recorder:
    def __init__(self, env):
        self.env = env
        self.target_file = env.target_file

        os.makedirs(os.path.dirname(self.target_file), exist_ok=True)

    def record(self):
        policy = self.env.demo_policy()
        taichi_env = self.env.taichi_env

        # initialize ...
        taichi_env_state = taichi_env.get_state()

        self.render_policy(taichi_env, taichi_env_state, policy, self.env.horizon, self.env.horizon_action)

        end_state = taichi_env.get_state()
        target = {}
        if taichi_env.has_particles:
            target['x'] = end_state['state']['x'],
            target['used'] = end_state['state']['used'],
            target['mat'] = taichi_env.simulator.particles_i.mat.to_numpy()

        if os.path.exists(self.target_file):
            os.remove(self.target_file)
        pkl.dump(target, open(self.target_file, 'wb'))
        print(f'==========> New target generated and dumped to {self.target_file}.')


    @staticmethod
    def render_policy(taichi_env, init_state, policy, horizon, horizon_action):
        taichi_env.set_state(**init_state)
        taichi_env.apply_agent_action_p(policy.get_actions_p())
        
        for i in range(horizon):
            if i < horizon_action:
                action = policy.get_action_v(i)
            else:
                action = None

            taichi_env.step(action)
            save = False
            # save = True
            taichi_env.render(iteration=0, save=save, t=i)
        # while True:
        #     taichi_env.render()


def record_action(env):
    env.reset()

    recorder = Recorder(env)
    recorder.record()
