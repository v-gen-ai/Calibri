import sys
import os
import json
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from absl import app, flags
from ml_collections import config_flags
import torch
from accelerate import Accelerator

import src.rewards
from src.models.flux_sg import SGFluxPipeline
from src.utils.utils import set_seed, save_config
from src.utils.logging_tb import create_writer, NullWriter
from src.data.prompts import make_loader
from src.optim.cmaes import CMAESTrainer

_CONFIG = config_flags.DEFINE_config_file("config", "configs/base.py", "Training configuration.")


def main(_):

    cfg = _CONFIG.value

    accelerator = Accelerator()  # NEW
    if cfg.experiment.seed is not None:
        set_seed(cfg.experiment.seed)

    inference_dtype = torch.float32
    if cfg.model.dtype == "fp16":
        inference_dtype = torch.float16
    elif cfg.model.dtype == "bf16":
        inference_dtype = torch.bfloat16

    device = accelerator.device
    pipeline = SGFluxPipeline(
        device=device,
        dtype=inference_dtype,
        model_name=cfg.model.model_name,
        num_models=cfg.scaleguidance.num_models
    )
    pipeline.pipeline.set_progress_bar_config(disable=True)

    reward_fn = getattr(src.rewards, 'multi_score')(cfg.device, cfg.reward_fn)
    eval_reward_fn = getattr(src.rewards, 'multi_score')(cfg.device, cfg.reward_fn_eval)

    os.makedirs(cfg.experiment.log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    final_logdir = os.path.join(cfg.experiment.log_dir, f"{cfg.experiment.name}_{current_time}")
    os.makedirs(final_logdir, exist_ok=True)
    save_config(cfg, final_logdir)
    writer = create_writer(final_logdir)

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
        cut_cnt=1
    )

    trainer = CMAESTrainer(
        cfg, pipeline, reward_fn, eval_reward_fn, writer,
        train_loader, val_loader, logdir=final_logdir, accelerator=accelerator
    )
    best_solution, best_train, best_val = trainer.train()
    if accelerator.is_main_process:
        writer.close()

if __name__ == "__main__":
    app.run(main)
