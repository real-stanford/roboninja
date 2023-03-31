import os
import gym
import torch
import random
import argparse
import numpy as np

from cut_simulation.utils.logger import Logger
from cut_simulation.optimizer.solver import solve_action
from cut_simulation.optimizer.recorder import record_action
from cut_simulation.utils.config import load_config
from cut_simulation.utils.misc import get_src_dir

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--record", action='store_true')

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    cfg = load_config(args.cfg_file)

    log_dir = os.path.join(get_src_dir(), '..', 'logs', args.exp_name)
    logger = Logger(log_dir)
    set_random_seed(cfg.EXP.seed)


    if args.record:
        env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False)
        record_action(env)
    else:
        env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=True)
        if cfg.SOLVER_ALGO == 'action':
            solve_action(env, log_dir, logger, cfg.SOLVER)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    main()
