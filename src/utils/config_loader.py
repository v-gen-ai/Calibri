import importlib.util
import sys
from pathlib import Path

def load_config_from_py(path: str):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    cfg = mod.get_config()
    return cfg
