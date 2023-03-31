import os
import pathlib
import sys

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pickle

import hydra
import numpy as np
from shapely.geometry import Polygon
import rich
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from roboninja.workspace.base_workspace import BaseWorkspace
from roboninja.utils.visualizer import Visualizer
from roboninja.utils.dynamics import get_init_pos
from roboninja.env import BaseEnv

OmegaConf.register_new_resolver("eval", eval, replace=True)

class SummarizeWorkspace(BaseWorkspace):
    include_keys = list()
    exclude_keys = list()

    def __init__(self, cfg: OmegaConf):
        # replace all macros with concrete value
        OmegaConf.resolve(cfg)
        super().__init__(cfg)

    def run(self):
        table = Table(title=self.cfg.title, box=rich.box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Approach", justify="left")
        table.add_column("Complete Rate", justify="right")
        table.add_column("Cut Mass", justify="right")
        table.add_column("Collision Ratio", justify="right")
        table.add_column("Avg Energy", justify="right")
        table.add_column("Max Energy", justify="right")

        base_eval_env = BaseEnv()
        bone_cfg = self.cfg.cut_env.bone
        knife_cfg = self.cfg.cut_env.knife
        init_pos = get_init_pos(np.array(self.cfg.init_action_p), knife_cfg) # [0.485, 0.215, 0.]
        visualizer = Visualizer()

        for approach, path in self.cfg.approach:
            print('==>loading', approach, path)
            comp_rate = list()
            cut_vol = list()
            cls_ratio = list()
            avg_eng = list()
            max_eng = list()

            for return_data_path in pathlib.Path(f'data/eval/{path}/storage').glob('*.pkl'):
                bone_idx = int(return_data_path.stem)
                bone_cfg.name = f'bone_{bone_idx}'
                bone_cfg.mesh_root = 'data/bone_generation' if bone_idx >= 300 else 'data/bone_generation_ood'

                max_vol = visualizer.get_max_vol(
                    bone_cfg=bone_cfg,
                    base_eval_env=base_eval_env,
                    init_pos=init_pos
                )
                
                return_data = pickle.load(open(return_data_path.resolve(), 'rb'))

                energy_threshold = 3

                complete_flag = (return_data['num_collision'] <= 10) and (np.max(return_data['energy']['energy_array']) < energy_threshold)                
                comp_rate.append(complete_flag)
                assert return_data['cut_mass'] <= max_vol
                all_stop_idx = np.where(return_data['energy']['energy_array'] >= energy_threshold)[0]
                stop_idx = all_stop_idx[0] if len(all_stop_idx) > 0 else -1

                cut_vol.append(np.sum(return_data['cut_mass_array'][:stop_idx]) / max_vol)
                cls_ratio.append(np.mean(return_data['collision_array'][:stop_idx]))
                avg_eng.append(np.mean(return_data['energy']['energy_array'][:stop_idx]))
                max_eng.append(np.max(return_data['energy']['energy_array'][:stop_idx]))
            table.add_row(
                str(approach),
                f'{np.mean(comp_rate):.5f}',
                f'{np.mean(cut_vol):.5f}',
                f'{np.mean(cls_ratio):.5f}',
                f'{np.mean(avg_eng):.5f}',
                f'{np.mean(max_eng):.5f}'
            )
        console = Console()
        console.print(table)
                
@hydra.main(
    version_base=None,
    config_path='../config', 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = SummarizeWorkspace(cfg)
    workspace.run()
 
if  __name__=='__main__':
    main()
