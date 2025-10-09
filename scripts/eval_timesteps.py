import sys
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from absl import app, flags
from ml_collections import config_flags
import torch

import src.rewards
from src.models.flux_sg import SGFluxPipeline
from src.utils.utils import set_seed, save_config
from src.utils.logging_tb import create_writer
from src.data.prompts import make_loader
from src.optim.cmaes import CMAESTrainer

_CONFIG = config_flags.DEFINE_config_file("config", "configs/base.py", "Training configuration.")


def plot_and_save(timesteps, total_rewards, save_path="/home/jovyan/sobolev/ii/danil/scaleguidance/evolve/logs/cmaes_imgr_2025-10-03_03:05:29/nfe_analysis.png"):
    # Распаковка
    score_orig = [x[0] for x in total_rewards]
    score_top = [x[1] for x in total_rewards]

    plt.figure(figsize=(8, 6))
    plt.plot(timesteps[:len(total_rewards)], score_orig, marker='o', label="Orig score")
    plt.plot(timesteps[:len(total_rewards)], score_top, marker='s', label="Top score")
    plt.xlabel("Timesteps")
    plt.ylabel("Score")
    plt.title("Total rewards per timestep")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(_):
    print("started")

    cfg = _CONFIG.value

    if cfg.experiment.seed is not None:
        set_seed(cfg.experiment.seed)

    inference_dtype = torch.float32
    if cfg.model.dtype == "fp16":
        inference_dtype = torch.float16
    elif cfg.model.dtype == "bf16":
        inference_dtype = torch.bfloat16

    pipeline = SGFluxPipeline(
        device=cfg.device,
        dtype=inference_dtype,
        model_name=cfg.model.model_name,
        num_models=cfg.scaleguidance.num_models
    )
    pipeline.pipeline.set_progress_bar_config(disable=True)

    reward_fn = getattr(src.rewards, 'multi_score')(cfg.device, cfg.reward_fn)
    eval_reward_fn = getattr(src.rewards, 'multi_score')(cfg.device, cfg.reward_fn_eval)

    train_loader = make_loader(
        cfg.data.train_dataset, 
        cfg.optimize.bucket_size, 
        cfg.data.num_workers, 
        cfg.data.shuffle, 
        cfg.data.drop_last, 
        limit=cfg.data.limit_train, 
        infinite=True
    )
    val_loader = make_loader(
        cfg.data.val_dataset, 
        cfg.data.batch_size, 
        0, 
        False, 
        False, 
        limit=cfg.data.limit_val, 
        # cut_cnt=2
    )

    total_rewards = []
    timesteps = [5, 10, 15, 20, 25, 30, 40, 50]

    for n_infer in tqdm(timesteps):
        trainer = CMAESTrainer(cfg, pipeline, reward_fn, eval_reward_fn, None, train_loader, val_loader)
        trainer.gen_params["num_inference_steps"] = n_infer

        ### orig
        with open("/home/jovyan/sobolev/ii/danil/scaleguidance/evolve/logs/cmaes_imgr_2025-10-03_03:05:29/checkpoints/cmaes_step_0000.json") as file:
            checkpoint = json.load(file)
        scales = checkpoint["solution"]
        score_orig = trainer._eval_validation(scales)["avg"]

        ### 60 steps
        with open("/home/jovyan/sobolev/ii/danil/scaleguidance/evolve/logs/cmaes_imgr_2025-10-03_03:05:29/checkpoints/cmaes_step_0060.json") as file:
            checkpoint = json.load(file)
        scales = checkpoint["solution"]
        score_top = trainer._eval_validation(scales)["avg"]

        total_rewards.append((score_orig, score_top))
        plot_and_save(timesteps, total_rewards)


if __name__ == "__main__":
    app.run(main)
