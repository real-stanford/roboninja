from cut_simulation.utils.misc import get_src_dir

def make_cls_config(self, cfg=None):
    _cfg = self.default_config()
    if cfg is not None:
        if isinstance(cfg, str):
            _cfg.merge_from_file(cfg)
        else:
            _cfg.merge_from_other_cfg(cfg)
    return _cfg

def merge_dict(a, b):
    if b is None:
        return a
    import copy
    a = copy.deepcopy(a)
    for key in a:
        if key in b:
            if not isinstance(b[key], dict):
                a[key] = b[key]
            else:
                assert not isinstance(a[key], list)
                a[key] = merge_dict(a[key], b[key])
    for key in b:
        if key not in a:
            raise ValueError("Key is not in dict A!")
    return a

def merge_lists(a, b):
    outs = []
    assert isinstance(a, list) and isinstance(b, list)
    for i in range(len(a)):
        assert isinstance(a[i], dict)
        x = a[i]
        if i < len(b):
            x = merge_dict(a[i], b[i])
        outs.append(x)
    return outs