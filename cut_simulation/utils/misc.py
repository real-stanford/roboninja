import os
import torch
import cut_simulation
import taichi as ti
import numpy as np
from cut_simulation.configs.macros import *

def get_src_dir():
    return os.path.dirname(cut_simulation.__file__)

def get_cfg_path(file):
    return os.path.join(get_src_dir(), 'envs', 'configs', file)

def get_tgt_path(file):
    return os.path.join(get_src_dir(), 'assets', 'targets', file)

def eval_str(x):
    if type(x) is str:
        return eval(x)
    else:
        return x
