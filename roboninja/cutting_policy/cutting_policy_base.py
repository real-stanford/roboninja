import numpy as np

class CuttingPolicyBase:
    state_estimation: np.ndarray
    state_estimation_sdf: np.ndarray
    
    def __init__(self):
        pass

    def reset(self, bone_idx):
        pass

    def get_action(self, step_idx, current_pos, **kwargs):
        raise NotImplementedError()
    
    def collision(self, step_idx):
        pass
    
    def update_state_estimation(self, state_estimation):
        pass

    def update_collision_map(self, collision_map):
        pass

    def get_render_data(self):
        return None