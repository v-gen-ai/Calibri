import random
import os
import numpy as np
import torch
import json
from PIL import Image


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

def to_pil_list(images):
    # вход: либо torch.Tensor [N,C,H,W] 0..1 или 0..255, либо список PIL/np

    pil_images = []
    if isinstance(images, torch.Tensor):
        arr = images
        # приведение к uint8
        if arr.dtype.is_floating_point:
            arr = (arr * 255.0).round().clamp(0, 255)
        arr = arr.to(torch.uint8).cpu().numpy()  # NCHW
        arr = arr.transpose(0, 2, 3, 1)         # NHWC
        for im in arr:
            pil_images.append(Image.fromarray(im))
        return pil_images

    # список PIL или np.ndarray
    for im in images:
        if isinstance(im, Image.Image):
            pil_images.append(im)
        else:
            # предполагаем np.ndarray NHWC uint8
            pil_images.append(Image.fromarray(im))
    return pil_images

def mean_score(scores, mode="mean") -> float:
    res_scores = {}
    for name, score_seq in scores.items():
        if score_seq is None:
            res_scores[name] = -np.inf
        if torch.is_tensor(score_seq):
            if score_seq.numel() == 0:
                res_scores[name] = -np.inf
            if mode == "mean":
                res_scores[name] = float(score_seq.float().mean().item())
            elif mode == "sum":
                res_scores[name] = float(score_seq.float().sum().item())
        else:
            if len(score_seq) == 0:
                res_scores[name] = -np.inf
            if mode == "mean":
                res_scores[name] = float(np.mean([float(s) for s in score_seq]))
            elif mode == "sum":
                res_scores[name] = float(np.sum([float(s) for s in score_seq]))
    return res_scores

def call_reward(fn, images, prompts, metadata=None):
    try:
        out = fn(images, prompts, metadata)
    except TypeError:
        out = fn(images, prompts)
    return out[0]