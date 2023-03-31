############ material type #############
RIGID = 0
MEAT = 1

CHOPPINGBOARD = 10
KNIFE = 11
SUPPORT = 12
BONE = 13

FRAME = 100
TARGET = 101
EFFECTOR = 102

MAT_PLASTO_ELASTIC = 201
MAT_RIGID = 202

############ material name #############
MAT_NAME = {
    RIGID: 'rigid',
    MEAT: 'meat'
}

############ material class #############
MAT_CLASS = {
    RIGID: MAT_RIGID,
    MEAT: MAT_PLASTO_ELASTIC,
}

############ default color #############
COLOR = {
    MEAT:       (1, 0.953, 0.62, 1.0),
    CHOPPINGBOARD:  (0.9, 0.9, 0.9, 1.0),
    KNIFE:     (0.7, 0.7, 0.7, 1.0),
    SUPPORT:   (0.5, 0.5, 0.5, 0.5),
    BONE:      (0.78, 0.44, 0.33, 1.0),

    FRAME:     (1.0, 0.2, 0.2, 1.0),
    TARGET:    (0.2, 0.9, 0.2, 0.4),
    EFFECTOR:  (1.0, 0.0, 0.0, 1.0),
}


############ properties #############
FRICTION = {
    CHOPPINGBOARD: 0.0,
    KNIFE: 0.0, # before: 0.0
    SUPPORT: 0.5,
    BONE: 0.5
}

MU = {
    RIGID: 416.67,
    MEAT: 2083.33
}

LAMDA = {
    RIGID: 277.78,
    MEAT: 1388.89
}

RHO = {
    RIGID: 1.0,
    MEAT: 1.0
}

YIELD_STRESS = {
    RIGID: 200.0,
    MEAT: 200.0,
}

############ dtype #############
import numpy as np
import torch
import taichi as ti
dprecision = 64
DTYPE_TI = eval(f'ti.f{dprecision}')
DTYPE_NP = eval(f'np.float{dprecision}')
DTYPE_TC = eval(f'torch.float{dprecision}')

############ misc #############
NOWHERE = [-100.0, -100.0, -100.0]


