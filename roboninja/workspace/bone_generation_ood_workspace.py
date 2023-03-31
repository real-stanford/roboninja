import os
import pathlib
import sys

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
from roboninja.utils.misc import mkdir
from roboninja.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf
from scipy import interpolate
from shapely.geometry import Polygon
from tqdm import trange
from trimesh.creation import extrude_polygon

OmegaConf.register_new_resolver("eval", eval)
np.random.seed(0)

def vis_plt(kps_x, kps_y, interpolate_x, interpolate_y, vis_type=['kps', 'curve'], **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if 'kps' in vis_type:
        ax.plot(kps_x, kps_y, 'x')
    if 'curve' in vis_type:
        ax.plot(interpolate_x, interpolate_y)
    ax.set_ylim([-0.1, 0.1])
    ax.set_xlim([-0.02, 0.22])
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img

class BoneGenerationWorkspace(BaseWorkspace):
    include_keys = list()
    exclude_keys = list()

    def __init__(self, cfg: OmegaConf):
        # replace all macros with concrete value
        OmegaConf.resolve(cfg)
        super().__init__(cfg)

    def gen_bone_kps(self, n_kps):
        # generate key points
        N = n_kps + 2
        kps_x = np.arange(0, 0.2/N*(N+1), 0.2/(N-1))
        kps_y = np.random.rand(N) # * 0 + 0.3

        kps_y[0] = kps_y[-1] = 0.3
        kps_x = np.insert(kps_x, 0, -0.006)
        kps_y = np.insert(kps_y, 0, 0)
        kps_x = np.insert(kps_x, 0, -0.01)
        kps_y = np.insert(kps_y, 0, -1)

        kps_x = np.append(kps_x, 0.206)
        kps_y = np.append(kps_y, 0)
        kps_x = np.append(kps_x, 0.21)
        kps_y = np.append(kps_y, -1)

        # normalize.   x: [0, 0.2]   y: [-0.07, 0.07]
        kps_x -= np.min(kps_x)
        kps_x /= np.max(kps_x) / 0.2
        kps_y *= 0.07

        # interpolate
        M = 1000
        x_range = kps_x[-1] - kps_x[0]
        tck = interpolate.splrep(kps_x, kps_y, s=0)
        x = np.arange(0, x_range/(M-1)*M, x_range/(M-1)) + kps_x[0]
        y = interpolate.splev(x, tck)

        return kps_x, kps_y, x, y

    def gen_bone_triangle(self):
        l_b = np.random.rand() * 0.05 + 0.05
        l_h = np.random.rand() * 0.05 + 0.05

        margin = 0.03
        x_st = np.random.rand() * (0.2 - l_b - 2 * margin) + margin
        x_fi = x_st + l_b

        x_mid = (x_st + x_fi) / 2
        y_c = 0.01
        y_0 = -l_h / 3 + y_c
        y_1 = l_h / 3 * 2 + y_c

        x = np.array([x_st, x_mid, x_fi])
        y = np.array([y_0, y_1, y_0])
        return None, None, x, y


    def gen_bone_rectangle(self):
        l_b = np.random.rand() * 0.05 + 0.05
        l_h = np.random.rand() * 0.05 + 0.05

        margin = 0.03
        x_st = np.random.rand() * (0.2 - l_b - 2 * margin) + margin
        x_fi = x_st + l_b

        y_c = 0.01
        y_0 = -l_h / 2 + y_c
        y_1 = l_h / 2 + y_c

        x = np.array([x_st, x_st, x_fi, x_fi])
        y = np.array([y_0, y_1, y_1, y_0])
        return None, None, x, y

    def gen_bone_ellipse(self):
        a = (np.random.rand() * 0.05 + 0.1) / 2
        b = (np.random.rand() * 0.05 + 0.1) / 2

        margin = 0.02
        x_c = np.random.rand() * (0.2 - 2 * a - 2 * margin) + margin + a
        y_c = 0.01
        
        thetas = np.arange(0, 1, 1.0/1000) * np.pi * 2
        x = a * np.cos(thetas) + x_c
        y = b * np.sin(thetas) + y_c

        return None, None, x, y


    def gen_bone(self, bone_type):
        if bone_type == '2kps':
            kps_x, kps_y, x, y = self.gen_bone_kps(2)
        elif bone_type == '3kps':
            kps_x, kps_y, x, y = self.gen_bone_kps(3)
        elif bone_type == '4kps':
            kps_x, kps_y, x, y = self.gen_bone_kps(4)
        elif bone_type == 'triangle':
            kps_x, kps_y, x, y = self.gen_bone_triangle()
        elif bone_type == 'rectangle':
            kps_x, kps_y, x, y = self.gen_bone_rectangle()
        elif bone_type == 'ellipse':
            kps_x, kps_y, x, y = self.gen_bone_ellipse()
        
        # extrude
        points = np.stack([x, y], axis=1)
        polygon = Polygon(points)
        mesh = extrude_polygon(polygon, self.cfg.bone_thichness)

        bone_info = {
            'kps_x': kps_x,
            'kps_y': kps_y,
            'interpolate_x': x,
            'interpolate_y': y,
            'normalized_x': (x - 0.1) / 0.2,
            'normalized_y': y / 0.2,
        }

        # normalize and scale to [-0.5, 0.5]
        mesh.vertices -= np.array([0.1, 0, self.cfg.bone_thichness / 2])
        mesh.vertices /= 0.2

        return bone_info, mesh

@hydra.main(
    version_base=None,
    config_path='../config', 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = BoneGenerationWorkspace(cfg)
    raw_dir = os.path.join(workspace.output_dir, 'raw')
    processed_dir = os.path.join(workspace.output_dir, 'processed')
    info_dir = os.path.join(workspace.output_dir, 'info')
    mkdir(raw_dir, clean=True)
    mkdir(processed_dir, clean=True)
    mkdir(info_dir, clean=True)


    idx = 0
    for bone_type, bone_num in cfg.bone_types:
        for _ in trange(bone_num):
            bone_info, bone_mesh = workspace.gen_bone(bone_type)
            
            pickle.dump(bone_info, open(os.path.join(info_dir, f'bone_{idx}.pkl'), 'wb'))
            bone_mesh.export(os.path.join(raw_dir, f'bone_{idx}.obj'))
            idx += 1

if __name__ == "__main__":
    main()
