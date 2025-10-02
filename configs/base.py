from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()
    cfg.device = "cuda"

    cfg.experiment = ConfigDict()
    cfg.experiment.name = "flux_autoguidance_cmaes"
    cfg.experiment.seed = 42
    cfg.experiment.log_dir = "logs"
    cfg.experiment.save_images = False  # можно включить примеры
    cfg.experiment.test_dataset = "data/test_log.txt"  # dataset to save images on
    cfg.experiment.save_json = True
    cfg.experiment.eval_orig_model = True

    cfg.model = ConfigDict()
    cfg.model.model_name = "black-forest-labs/FLUX.1-dev"
    cfg.model.dtype = "bf16"

    cfg.gen = ConfigDict()
    cfg.gen.num_inference_steps = 15
    cfg.gen.guidance_scale = 3.5
    cfg.gen.image_size = 512

    cfg.scaleguidance = ConfigDict()

    cfg.optimize = ConfigDict()
    cfg.optimize.initial_sigma = 0.25
    cfg.optimize.max_generations = 50
    cfg.optimize.population_size = None  # пусть CMA-ES подберёт
    cfg.optimize.val_every_steps = 3
    cfg.optimize.early_stopping_patience = 10
    cfg.optimize.overfitting_threshold = 0.15
    cfg.optimize.blocks_bound_low = -1.0
    cfg.optimize.blocks_bound_high = 2.0
    cfg.optimize.models_bound_low = -10.0
    cfg.optimize.models_bound_high = 10.0
    cfg.optimize.bucket_size = 64

    cfg.data = ConfigDict()
    cfg.data.batch_size = 8
    cfg.data.num_workers = 2
    cfg.data.shuffle = True
    cfg.data.drop_last = True
    cfg.data.limit_train = -1
    cfg.data.limit_val = -1

    return cfg
