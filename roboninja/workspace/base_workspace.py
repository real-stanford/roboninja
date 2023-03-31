import os
import pathlib

import dill
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


class BaseWorkspace:
    include_keys = list()
    exclude_keys = list()

    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        
    @property
    def output_dir(self):
        return HydraConfig.get().runtime.output_dir

    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = self.exclude_keys
        if include_keys is None:
            include_keys = self.include_keys

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)

        torch.save(payload, path.open('wb'), pickle_module=dill)
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self._load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload

    def _load_payload(self, payload, exclude_keys, include_keys):
        if exclude_keys is None:
            exclude_keys = self.exclude_keys
        if include_keys is None:
            include_keys = self.include_keys

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill, **kwargs)
        instance = cls(payload['cfg'])
        instance._load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.
        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
