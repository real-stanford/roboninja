import numpy as np
from roboninja.env.base_env import BaseEnv
from shapely.geometry import Point, Polygon
from roboninja.utils.visualizer import Visualizer

class SimEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.bone_polygon = None
        self.visualizer = Visualizer()

    def reset(self, bone_wrd_pts, **kwargs):
        self.bone_polygon = Polygon(bone_wrd_pts)
        self.cut_mass = 0
        self.cut_mass_array = list()
        self.collision_array = list()
        self.num_collision = 0

    def move(self, wrd_pos, pre_pos, collision_candidates=None, **kwargs) -> bool:
        # cumulate cut mass
        delta_cut_mass = 0.5 * ((wrd_pos[0] - 0.4) + (pre_pos[0] - 0.4)) * \
            (self.clip(pre_pos[1]) - self.clip(wrd_pos[1]))
        self.cut_mass += delta_cut_mass
        self.cut_mass_array.append(delta_cut_mass)

        # detect collision
        p = Point(wrd_pos[0], wrd_pos[1])
        stop_signal = p.distance(self.bone_polygon) < 1e-9
        self.collision_array.append(stop_signal)

        if collision_candidates is None:
            return stop_signal
        else:
            collision_results = list()
            for c in collision_candidates:
                p = Point(c[0], c[1])
                collision_results.append(p.distance(self.bone_polygon) < 1e-9)
            return collision_results

    @property
    def energy(self):
        return None