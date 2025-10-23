import os
import sys
import re
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from absl import app, flags
from ml_collections import config_flags

# project imports
from src.utils.utils import set_seed, call_reward, mean_score
import src.rewards as rewards

CONFIG = config_flags.DEFINE_config_file("config", default="configs/scaleguidance.py:cmaes_image_reward",
                                         help_string="Training configuration to reuse for eval")


def natural_sort_key(p: Path):
    # Natural sort by splitting digits and text
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def list_step_dirs(out_dir: Path):
    step_dirs = []
    for p in out_dir.iterdir():
        if p.is_dir():
            m = re.match(r"step_(\d+)$", p.name)
            if m:
                step_dirs.append((int(m.group(1)), p))
    step_dirs.sort(key=lambda x: x[0])
    return step_dirs


def read_val_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def main(_):

    log_dir = "/home/jovyan/sobolev/ii/danil/scaleguidance/evolve/logs/cmaes_hpsv3_1024_2025-10-10_04:27:45"
    out_dir = os.path.join(log_dir, "eval")
    out_path = Path(out_dir)
    if not out_path.exists():
        raise FileNotFoundError(f"Not found: {out_dir}")

    cfg = CONFIG.value
    if getattr(cfg.experiment, "seed", None) is not None:
        set_seed(cfg.experiment.seed)

    # готовим eval_reward_fn: либо из конфига, либо перезаписываем под одну метрику
    scoredict = cfg.reward_fn_eval
    eval_reward_fn = rewards.multi_score(cfg.device, scoredict)

    val_prompts = read_val_prompts(cfg.data.val_dataset)
    num_prompts = len(val_prompts)

    step_dirs = list_step_dirs(out_path)

    rows = []
    batch_size = cfg.data.batch_size
    for step, step_dir in tqdm(step_dirs, desc="Evaluating steps"):
        images = []
        # print(len(sorted(list(step_dir.iterdir()))))
        for image_path in sorted(list(step_dir.iterdir())):
            img = Image.open(image_path)
            images.append(img)
        
        total_scores = None
        
        for i in tqdm(range(0, num_prompts, batch_size), desc="Evaluating in batches"):
            prompts_batch = val_prompts[i:i+batch_size]
            images_batch = images[i:i+batch_size]
            scores = call_reward(eval_reward_fn, images_batch, prompts_batch)
            sum_scores = mean_score(scores, mode="sum")
            if total_scores is None:
                total_scores = sum_scores
            else:
                for name, score in sum_scores.items():
                    total_scores[name] += score
        
        metrics_step = {name: score / len(val_prompts) for name, score in total_scores.items()}

        row = {"step": step}
        row.update(metrics_step)
        rows.append(row)

        steps = [r["step"] for r in rows]
        for metric_key in metrics_step.keys():
            ys = [float(r[metric_key]) for r in rows]
            plt.figure(figsize=(7, 4))
            plt.plot(steps, ys, marker="o")
            plt.xlabel("step")
            plt.ylabel(metric_key)
            plt.title(f"val {metric_key} over checkpoints")
            plt.grid(True, alpha=0.5)
            png_path = out_path / f"val_{metric_key}.png"
            plt.tight_layout()
            plt.savefig(png_path.as_posix(), dpi=150)
            plt.close()
            print(f"Saved: {png_path}")

            csv_path = out_path / f"val_{metric_key}.csv"
            fieldnames = ["step", metric_key]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow({"step": r["step"], metric_key: r[metric_key]})
            print(f"Saved: {csv_path}")

if __name__ == "__main__":
    app.run(main)
