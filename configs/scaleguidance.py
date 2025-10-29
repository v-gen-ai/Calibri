import configs.base as base


def cmaes_image_reward():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_imgr"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"  # t2i-compbench
    cfg.optimize.population_size = None
    cfg.reward_fn = {
        "imagereward": 1.0,
    }
    cfg.reward_fn_eval = {
        # "imagereward": 1.0,
        # "hpsv3_remote": 1.0,
    }
    return cfg

def cmaes_image_reward_val():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_imgr_val"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"  # t2i-compbench
    cfg.optimize.population_size = None
    cfg.reward_fn = {
        "imagereward": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        # "hpsv3": 1.0,
        # "qalign": 1.0
    }
    return cfg

def cmaes_image_reward_2models():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_imgr_2models"
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"  # t2i-compbench
    cfg.optimize.population_size = None
    cfg.reward_fn = {
        "imagereward": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
    }
    return cfg


def cmaes_hpsv3():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"  # t2i-compbench
    cfg.optimize.population_size = None
    cfg.data.save_eval_imgs = True
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        # "imagereward": 1.0,
        "hpsv3_remote": 1.0,
    }
    return cfg


def cmaes_hpsv3_val():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"  # t2i-compbench
    cfg.optimize.population_size = None
    cfg.data.save_eval_imgs = True
    cfg.reward_fn = {
        # "hpsv3_remote": 1.0,
    }
    cfg.gen.image_size = 1024
    cfg.device = "cuda:0"
    cfg.data.batch_size = 2
    cfg.reward_fn_eval = {
        # "imagereward": 1.0,
        # "hpsv3_remote": 1.0,
        "hpsv3": 1.0
        # "qalign": 1.0,
        # "unifiedreward_qwen": 1.0,
        # "pickscore": 1.0,
        # "jpeg_compressibility": 1.0
    }
    return cfg


def cmaes_hpsv3_2models():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_2models"
    cfg.experiment.resume_state = None
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"  # t2i-compbench
    cfg.optimize.population_size = None
    cfg.data.save_eval_imgs = True
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0
    }
    return cfg

def cmaes_hpsv3_2models_swanlab():
    cfg = cmaes_hpsv3_2models()
    cfg.experiment.name = "cmaes_hpsv3_2models_swanlab"
    cfg.experiment.use_swanlab = True
    cfg.experiment.swanlab_project = "ScaleGuidance-FLUX"
    cfg.experiment.swanlab_experiment = "flux_hpsv3_2models_optimization"
    cfg.experiment.swanlab_api_key = "grt2RQLOhGXjNZvY4eMb3"  # Your API key
    return cfg

def cmaes_hpsv3_2models_val():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_2models"
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 5
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"  # t2i-compbench
    cfg.optimize.population_size = None
    cfg.data.save_eval_imgs = True
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        # "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        # "qalign_remote": 1.0
    }
    return cfg


def cmaes_hpsv3_2models_sd3_medium():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_2models_sd3-medium"
    cfg.experiment.resume_state = None
    cfg.model.model_name = "stabilityai/stable-diffusion-3.5-medium"
    cfg.gen.guidance_scale = 0.0
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 5
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"  # t2i-compbench
    cfg.optimize.population_size = None
    cfg.data.save_eval_imgs = True
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0
    }
    return cfg


def cmaes_qwen_hpsv3():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_qwen_hpsv3"
    cfg.model.model_name = "Qwen/Qwen-Image"
    cfg.model.dtype = "bf16"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 5
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None
    cfg.data.save_eval_imgs = True
    cfg.gen.guidance_scale = 4.0  # CFG scale for Qwen
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "hpsv3_remote": 1.0,
    }
    return cfg

def cmaes_qwen_hpsv3_swanlab():
    cfg = cmaes_qwen_hpsv3()
    cfg.experiment.name = "cmaes_qwen_hpsv3_swanlab"
    cfg.experiment.use_swanlab = True
    cfg.experiment.swanlab_project = "ScaleGuidance-Qwen"
    cfg.experiment.swanlab_experiment = "qwen_hpsv3_optimization"
    cfg.experiment.swanlab_api_key = "grt2RQLOhGXjNZvY4eMb3"  # Your API key
    # Полная валидация как в оригинальном пайплайне
    cfg.data.batch_size = 1 # Стандартный batch size
    cfg.gen.num_inference_steps = 15  # Стандартное количество шагов
    cfg.data.limit_val = None  # Полная валидация (все изображения)
    return cfg

def cmaes_qwen_hpsv3_fast():
    """Быстрая конфигурация без SwanLab для тестирования"""
    cfg = cmaes_qwen_hpsv3()
    cfg.experiment.name = "cmaes_qwen_hpsv3_fast"
    cfg.experiment.use_swanlab = False  # Отключаем SwanLab
    # Ускоряем валидацию
    cfg.data.limit_val = 5  # Только 5 батчей (20 изображений)
    cfg.gen.num_inference_steps = 15  # Только 5 шагов генерации
    cfg.optimize.val_every_steps = 10  # Валидация каждые 10 поколений
    cfg.experiment.eval_orig_model = False  # Пропустить baseline валидацию
    return cfg

def cmaes_qwen_hpsv3_2models():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_qwen_hpsv3_2models"
    cfg.model.model_name = "Qwen/Qwen-Image"
    cfg.model.dtype = "bf16"
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 5
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None
    cfg.data.save_eval_imgs = True
    cfg.gen.guidance_scale = 4.0  # CFG scale for Qwen
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "hpsv3_remote": 1.0,
    }
    return cfg

def get_config(name):
    return globals()[name]()
