import numpy as np
import torch

from roboninja.cutting_policy.cutting_policy_base import CuttingPolicyBase
from roboninja.utils.geom import binary2sdf
from roboninja.utils.visualizer import Visualizer


class CuttingPolicyNonAdaptive(CuttingPolicyBase):
    def __init__(self, close_loop_policy_model, device, cfg):
        super().__init__()

        self.close_loop_policy_model = close_loop_policy_model
        self.device = device
        self.cfg = cfg
        self.visualizer = Visualizer()
        
        self.n_back = self.cfg.n_back
            
    def get_action(self, step_idx, current_pos):
        img_knife = self.visualizer.vis_knife(current_pos, color=0.5)
        img_bone_input = self.state_estimation_sdf
        img_tolerance = np.zeros_like(img_bone_input)
        img_input = np.stack([img_bone_input, img_knife, img_tolerance], axis=0)
        action = self.close_loop_policy_model(
                torch.from_numpy(img_input[None]).to(dtype=torch.float32, device=self.device)
            )[0].cpu().numpy()
        return action
    
    def update_state_estimation(self, state_estimation):
        """
        state_estimation: binary mask
        """
        self.state_estimation = state_estimation
        self.state_estimation_sdf = binary2sdf(state_estimation)

    def get_render_data(self):
        return {
            'state_estimation': self.state_estimation,
        }