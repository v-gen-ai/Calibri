import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from absl import app, flags
from ml_collections import config_flags

from src.models.flux_autoguidance import AutoGuidanceFluxPipeline
from src.utils.config_loader import load_config_from_py
from src.utils.logging_tb import create_writer
from src.data.prompts import make_loader
from src.optim.cmaes import CMAESAlphaTrainer

_CONFIG = config_flags.DEFINE_config_file("config", "configs/base.py", "Training configuration.")


def main(_):

    cfg = _CONFIG.value
    print(cfg)
    # if args.log_dir:
    #     cfg.experiment.log_dir = args.log_dir
    # cfg.data.train_prompts = args.train_prompts
    # cfg.data.val_prompts = args.val_prompts

    # torch.manual_seed(cfg.experiment.seed)

    # pipeline = AutoGuidanceFluxPipeline(
    #     cfg.model.main_model,
    #     cfg.model.guidance_model,
    #     torch_dtype=cfg.model.torch_dtype,
    #     device_map=cfg.model.device_map,
    #     low_cpu_mem_usage=cfg.model.low_cpu_mem_usage
    # )

    # os.makedirs(cfg.experiment.log_dir, exist_ok=True)
    # writer = create_writer(cfg.experiment.log_dir, cfg.experiment.name)

    # train_loader = make_loader(
    #     cfg.data.train_prompts, cfg.data.batch_size, cfg.data.num_workers, cfg.data.shuffle, cfg.data.drop_last, limit=cfg.data.limit_train
    # )
    # val_loader = make_loader(
    #     cfg.data.val_prompts, cfg.data.batch_size, 0, False, False, limit=cfg.data.limit_val
    # )

    # trainer = CMAESAlphaTrainer(cfg, pipeline, writer, train_loader, val_loader)
    # best_solution, best_train, best_val = trainer.train()

    # writer.close()

if __name__ == "__main__":
    app.run(main)
