"""Config loading: company YAML/JSON merged over configs/_defaults.yaml."""
import json, os, yaml

def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)

def load_config(path):
    here = os.path.dirname(os.path.abspath(path))
    dflt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "_defaults.yaml")
    cfg = _merge(_read(dflt), _read(path))
    cfg["_dir"] = here
    for need in ("company", "reporting_currency", "tree_file", "sources"):
        if need not in cfg:
            raise ValueError(f"config missing required key: {need}")
    return cfg

def resolve(cfg, p, root=None):
    """Resolve a path from config: absolute, else relative to repo root, else config dir."""
    if p is None or os.path.isabs(p):
        return p
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for base in (os.getcwd(), root, cfg.get("_dir", ".")):
        c = os.path.join(base, p)
        if os.path.exists(c):
            return c
    return os.path.join(root, p)
