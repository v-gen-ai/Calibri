import os
import sys
import re
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from absl import app, flags
from ml_collections import config_flags

from src.models.flux_sg import SGFluxPipeline
from src.utils.utils import set_seed
from src.data.prompts import make_loader
from src.optim.cmaes import CMAESTrainer
import src.metrics.rewards as rewards

CONFIG = config_flags.DEFINE_config_file("config", default="configs/calibri.py:cmaes_hpsv3_flux_gates",
                                         help_string="Training configuration to reuse for eval")

def main(_):

    log_dir = "./logs/cmaes_hpsv3_1024_2025-10-10_04:27:45"

    cfg = CONFIG.value
    if cfg.experiment.seed is not None:
        set_seed(cfg.experiment.seed)

    infer_dtype = torch.float32
    if getattr(cfg.model, "dtype", "fp32") == "fp16":
        infer_dtype = torch.float16
    elif getattr(cfg.model, "dtype", "fp32") == "bf16":
        infer_dtype = torch.bfloat16

    pipeline = SGFluxPipeline(
        device=cfg.device,
        dtype=infer_dtype,
        model_name=cfg.model.model_name,
        num_models=cfg.scaleguidance.num_models
    )
    pipeline.pipeline.set_progress_bar_config(disable=True)

    scoredict = getattr(cfg, "reward_fn_eval", None) or cfg.reward_fn

    eval_reward_fn = rewards.multi_score(cfg.device, scoredict)

    val_loader = make_loader(
        cfg.data.val_dataset, 
        8, 
        0, 
        False, 
        False, 
        limit=cfg.data.limit_val, 
        # cut_cnt=2
    )

    trainer = CMAESTrainer(cfg, pipeline, reward_fn=None, eval_reward_fn=eval_reward_fn,
                           writer=None, train_loader=None, val_loader=val_loader)

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints dir: {ckpt_dir}")

    ckpts = []
    for entry in os.listdir(ckpt_dir):
        step_dir_path = os.path.join(ckpt_dir, entry)
        
        if os.path.isdir(step_dir_path):
            for fn in os.listdir(step_dir_path):
                if fn.endswith(".json"):
                    path = os.path.join(step_dir_path, fn)
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)

                    m = re.search(r"step_?(\d+)", entry)
                    if m:
                        step = int(m.group(1))
                    else:
                        step = int(payload.get("step", -1))
                        
                    sol = np.asarray(payload["solution"], dtype=np.float64)
                    ckpts.append((step, path, sol))

    ckpts.sort(key=lambda x: x[0])
    
    ckpts = ckpts[6:]
    print(f"{len(ckpts)} steps for val:", [t[0] for t in ckpts])

    out_dir = os.path.join(log_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    
    for step, path, sol in tqdm(ckpts):
        save_dir = os.path.join(out_dir, f"step_{step}")
        vals = trainer._eval_validation(sol, seed=cfg.experiment.seed,
                                        save_dir=save_dir,
                                        save_images=True)

if __name__ == "__main__":
    app.run(main)
