import random
import os
import numpy as np
import torch
import json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(cfg, logdir):
    try:
        resolved_json = cfg.to_json_best_effort(indent=2)
    except Exception:
        # fallback, если вдруг встретились нестандартные типы
        resolved_json = json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2)
    with open(os.path.join(logdir, "config.json"), "w", encoding="utf-8") as f:
        f.write(resolved_json)
