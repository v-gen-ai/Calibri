from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.experiment = ConfigDict()
    cfg.experiment.name = "flux_autoguidance_cmaes"
    cfg.experiment.seed = 42
    cfg.experiment.log_dir = "runs/exp"
    cfg.experiment.save_images = False  # можно включить примеры
    cfg.experiment.save_json = True

    cfg.model = ConfigDict()
    cfg.model.main_model = "black-forest-labs/FLUX.1-dev"
    cfg.model.guidance_model = "black-forest-labs/FLUX.1-dev"
    cfg.model.torch_dtype = "float16"
    cfg.model.device_map = "balanced"
    cfg.model.low_cpu_mem_usage = True

    cfg.gen = ConfigDict()
    cfg.gen.num_inference_steps = 12
    cfg.gen.guidance_scale = 3.5
    cfg.gen.image_size = 512

    cfg.autoguidance = ConfigDict()
    cfg.autoguidance.initial_guidance_weight = 0.3

    cfg.optimize = ConfigDict()
    cfg.optimize.initial_sigma = 0.25
    cfg.optimize.max_generations = 50
    cfg.optimize.population_size = None  # пусть CMA-ES подберёт
    cfg.optimize.val_every_steps = 3
    cfg.optimize.early_stopping_patience = 10
    cfg.optimize.overfitting_threshold = 0.15

    cfg.data = ConfigDict()
    cfg.data.train_prompts = ""  # путь к файлу
    cfg.data.val_prompts = ""    # путь к файлу
    cfg.data.batch_size = 8
    cfg.data.num_workers = 2
    cfg.data.shuffle = True
    cfg.data.drop_last = False
    cfg.data.limit_train = -1  # -1 без ограничения
    cfg.data.limit_val = -1

    return cfg
