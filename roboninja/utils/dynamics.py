import numpy as np


def get_init_pos(init_action, knife_cfg):
    action_p_lower = np.array(knife_cfg.effectors[0].params.action_p_lower)
    action_p_upper = np.array(knife_cfg.effectors[0].params.action_p_upper)
    wrd_p = (init_action + 1) / 2 * (action_p_upper - action_p_lower) + action_p_lower
    init_pos = np.zeros(3) #[x, y, rot]
    init_pos[:2] = wrd_p[:2]
    return init_pos

def forward_dynamics(action, current_pos, knife_cfg):
    """
    current_pos + aciton --> next_pos
    """
    step_length = knife_cfg.effectors[0].params.step_length
    next_pos = np.zeros(3) #[x, y, rot]
    next_pos[0] = current_pos[0] + np.sin(action[1]) * step_length
    next_pos[1] = current_pos[1] - np.cos(action[1]) * step_length
    next_pos[2] = action[0]
    return next_pos

def inverse_dynamics(current_pos, next_pos, knife_cfg):
    """
    current_pos + next_pos --> aciton
    """
    step_length = knife_cfg.effectors[0].params.step_length
    action = np.array([next_pos[2], 0, 0])
    action[1] = np.arctan2(
        (next_pos[0] - current_pos[0]) / step_length,
        -(next_pos[1] - current_pos[1]) / step_length,
    )
    if action[1] > np.pi:
        action[1] -= np.pi * 2
    return action


def forward_dynamics_inv(action, next_pos, knife_cfg):
    """
    next_pos + aciton --> current_pos
    Attention: it can only calculate position, without rotation
    """
    step_length = knife_cfg.effectors[0].params.step_length
    current_pos = np.zeros(3) #[x, y, rot]
    current_pos[0] = next_pos[0] - np.sin(action[1]) * step_length
    current_pos[1] = next_pos[1] + np.cos(action[1]) * step_length
    current_pos[2] = action[0] # dummy
    return current_pos
    
def normalized_action2pos(action, knife_cfg):
    """
    action: sequence of raw actions with init. [N + 1, 3]
    return: pos [N + 1, 3]
    """
    action_type = knife_cfg.effectors[0].params.action_type
    action_scale_v = np.array(knife_cfg.effectors[0].params.get(f'action_scale_v_{action_type}'))
    rel_flag = 1 if action_type == 'rel' else 0

    len_action = len(action)
    pos = np.zeros([len_action, 3]) #[x, y, rot]
    pos[0] = get_init_pos(action[-1], knife_cfg)
    cur_action = np.zeros(3)
    for i in range(1, len_action):
        cur_action = rel_flag * cur_action + action_scale_v * action[i - 1]
        pos[i] = forward_dynamics(cur_action, pos[i - 1], knife_cfg)
    return pos

def decode_action(action, knife_cfg):
    """
    action: sequence of raw actions with init. [N + 1, 3]
    return: unnormalized action. [N, 3]
    """
    action_type = knife_cfg.effectors[0].params.action_type
    action_scale_v = np.array(knife_cfg.effectors[0].params.get(f'action_scale_v_{action_type}'))
    rel_flag = 1 if action_type == 'rel' else 0

    len_action = len(action)
    new_action = np.zeros([len_action, 3]) #[x, y, rot]
    cur_action = np.zeros(3)
    for i in range(1, len_action):
        cur_action = rel_flag * cur_action + action_scale_v * action[i - 1]
        new_action[i - 1] = cur_action
    return new_action
