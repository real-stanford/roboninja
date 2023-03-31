import numpy as np

from roboninja.cutting_policy.cutting_policy_base import CuttingPolicyBase
from roboninja.utils.dynamics import forward_dynamics
from roboninja.utils.geom import binary2sdf
from roboninja.utils.visualizer import Visualizer


class CuttingPolicyGreedy(CuttingPolicyBase):
    def __init__(self, knife_cfg):
        super().__init__()
        self.knife_cfg = knife_cfg
        self.visualizer = Visualizer()
        self.sdf_threshold = 0.02
        self.n_share = 30 # number of shares in max angles
        self.max_angle = np.pi / 2.5
        self.last_current = 0
        self.max_delta_left = 4
        self.max_delta_right = 4

    def reset(self, **kwargs):
        self.last_current = 0
    
    def get_action(self, step_idx, current_pos):
        pix = self.visualizer.wrd2pix(current_pos[:2])
        current = 0 if np.max(pix) < 256 and np.min(self.state_estimation_sdf[pix[1]]) > 0 else -self.n_share
        action = np.array([0, 0, 0])
        while current < self.n_share:
            angle = -(current / self.n_share) * self.max_angle
            action = np.array([0, angle, 0])
            next_pos = forward_dynamics(action, current_pos, self.knife_cfg)
            pix = self.visualizer.wrd2pix(next_pos[:2])
            if np.max(pix) > 255: break
            if self.state_estimation_sdf[pix[1], pix[0]] > self.sdf_threshold:
                break
            current += 1

        current = np.clip(current, self.last_current - self.max_delta_left, self.last_current + self.max_delta_right)

        while current > -self.n_share:
            angle = -(current / self.n_share) * self.max_angle
            action[0] = angle
            flag = True
            N = 10
            for i in range(N):
                knife_tail_wrd = next_pos[:2] + np.array([-np.sin(angle), np.cos(angle)]) * self.visualizer.knife_width * i / (N - 1)
                pix = self.visualizer.wrd2pix(knife_tail_wrd)
                if np.max(pix) <= 255 and self.state_estimation_sdf[pix[1], pix[0]] == 0:
                    flag = False
            if flag: break
            current -= 1

        self.last_current = current

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