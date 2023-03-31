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


    def gen_bone(self):
        # generate key points
        N = self.cfg.n_kps + 2
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

    for idx in trange(cfg.n_bone):
        bone_info, bone_mesh = workspace.gen_bone()
        
        pickle.dump(bone_info, open(os.path.join(info_dir, f'bone_{idx}.pkl'), 'wb'))
        bone_mesh.export(os.path.join(raw_dir, f'bone_{idx}.obj'))

if __name__ == "__main__":
    main()
