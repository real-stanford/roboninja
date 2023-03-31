import numpy as np
import torch

from roboninja.cutting_policy.cutting_policy_base import CuttingPolicyBase
from roboninja.utils.geom import binary2sdf
from roboninja.utils.visualizer import Visualizer


class CuttingPolicyAdaptive(CuttingPolicyBase):
    def __init__(self, close_loop_policy_model, device, cfg):
        super().__init__()

        self.close_loop_policy_model = close_loop_policy_model
        self.device = device
        self.cfg = cfg
        self.visualizer = Visualizer()
        
        self.n_back = self.cfg.n_back
        self.min_tolerance = cfg.policy.min_tolerance
        self.max_tolerance = cfg.policy.max_tolerance
        assert self.max_tolerance <= cfg.policy.tolerance_range

        self.increase_delta = cfg.policy.increase_delta
        self.decrease_speed = cfg.policy.decrease_speed

    def reset(self, **kwargs):
        self.tolerance_array = np.ones(self.cfg.seq_len * 2) * self.min_tolerance

    def update_tolerance(self, step_idx):
        for i in range(self.n_back):
            if step_idx >= len(self.tolerance_array):
                return
            self.tolerance_array[step_idx] = np.clip(
                self.tolerance_array[step_idx] + self.increase_delta,
                self.min_tolerance, self.max_tolerance
            )
            step_idx += 1

        for i in range(60):
            if step_idx >= len(self.tolerance_array):
                return
            delta = self.increase_delta - self.decrease_speed * i
            if delta < 0:
                return
            self.tolerance_array[step_idx] = np.clip(
                self.tolerance_array[step_idx] + delta,
                self.min_tolerance, self.max_tolerance
            )
            step_idx += 1
    
    def get_tolerance(self, step_idx=None):
        if step_idx is None:
            return self.tolerance_array
        else:
            return self.tolerance_array[step_idx]
            
    def get_action(self, step_idx, current_pos):
        tolerance = self.get_tolerance(step_idx)
        img_knife = self.visualizer.vis_knife(current_pos, color=0.5)
        img_bone_input = self.state_estimation_sdf
        img_tolerance = np.ones_like(img_bone_input) * tolerance
        img_input = np.stack([img_bone_input, img_knife, img_tolerance], axis=0)
        action = self.close_loop_policy_model(
                torch.from_numpy(img_input[None]).to(dtype=torch.float32, device=self.device)
            )[0].cpu().numpy()
        return action
    
    def collision(self, step_idx):
        self.update_tolerance(step_idx)

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