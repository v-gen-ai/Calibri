import torch
import configs.base as base

def cmaes_hpsv3_flux_gates():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_flux_gates"
    cfg.model.model_name = "black-forest-labs/FLUX.1-dev"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.gen.image_size = 512
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0,
        # "pickscore": 1.0,
    }
    return cfg

def cmaes_hpsv3_flux_block():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_flux_block"
    cfg.model.model_name = "flux_block"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.gen.image_size = 512
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0,
        # "pickscore": 1.0,
    }
    return cfg

def cmaes_hpsv3_flux_layer():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_flux_layer"
    cfg.model.model_name = "flux_mlp_attn"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.gen.image_size = 512
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0,
        # "pickscore": 1.0,
    }
    return cfg

def cmaes_pickscore_flux_layer():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_pickscore_flux_layer"
    cfg.model.model_name = "flux_mlp_attn"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.gen.image_size = 512
    cfg.reward_fn = {
        "pickscore": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0,
        "pickscore": 1.0,
    }
    return cfg

def cmaes_pickscore_flux_gates():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_pickscore_flux_gates"
    cfg.model.model_name = "black-forest-labs/FLUX.1-dev"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.gen.image_size = 512
    cfg.reward_fn = {
        "pickscore": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0,
        "pickscore": 1.0,
    }
    return cfg

def cmaes_pickscore_flux_block():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_pickscore_flux_block"
    cfg.model.model_name = "flux_block"
    cfg.scaleguidance.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.gen.image_size = 512
    cfg.reward_fn = {
        "pickscore": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0,
        "pickscore": 1.0,
    }
    return cfg

def cmaes_hpsv3_2models_sd3_medium_cfg():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_2models_sd3-medium_cfg"
    cfg.model.model_name = "stabilityai/stable-diffusion-3.5-medium"
    cfg.gen.guidance_scale = 7.0
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.gen.num_inference_steps = 15
    cfg.gen.num_inference_steps_val = 15
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.gen.image_size = 512
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0
    }
    return cfg

def cmaes_hpsv3_2models_sd3_medium_cfg_flowgrpo():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_2models_sd3-medium_cfg_flowgrpo"
    cfg.model.model_name = "jieliu/SD3.5M-FlowGRPO-GenEval"
    cfg.gen.guidance_scale = 7.0
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.gen.num_inference_steps = 15
    cfg.gen.num_inference_steps_val = 15
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.reward_fn = {
        "hpsv3_remote": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0
    }
    return cfg

def cmaes_pickscore_2models_sd3_medium_cfg():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_pickscore_2models_sd3-medium_cfg"
    cfg.experiment.resume_state = None
    cfg.model.model_name = "stabilityai/stable-diffusion-3.5-medium"
    cfg.gen.guidance_scale = 7.0
    cfg.gen.num_inference_steps = 15
    cfg.gen.num_inference_steps_val = 15
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.gen.image_size = 512
    cfg.reward_fn = {
        "pickscore": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0,
        "pickscore": 1.0,
    }
    return cfg

def cmaes_pickscore_2models_sd3_medium_cfg_flowgrpo():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_pickscore_2models_sd3-medium_cfg_flowgrpo"
    cfg.model.model_name = "jieliu/SD3.5M-FlowGRPO-PickScore"
    cfg.gen.guidance_scale = 7.0
    cfg.scaleguidance.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.gen.num_inference_steps = 15
    cfg.gen.num_inference_steps_val = 15
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = None  # auto
    cfg.data.save_eval_imgs = True
    cfg.data.batch_size_train = 4
    cfg.data.batch_size_val = 4 * num_gpu
    cfg.gen.image_size = 512
    cfg.reward_fn = {
        "pickscore": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0,
        "pickscore": 1.0,
    }
    return cfg

def cmaes_qwen_clean_hpsv3_2models_cfg():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_qwen_clean_hpsv3_2models_cfg"
    cfg.model.model_name = "qwen"
    cfg.model.dtype = "bf16"
    cfg.scaleguidance.num_models = 2
    cfg.scaleguidance.use_cfg = True
    cfg.scaleguidance.negative_prompt = ""
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = 24  # 3 * 8 = 24 (with 8 gpu) (parallelization is across cma-es candidates)
    cfg.data.save_eval_imgs = True
    cfg.gen.guidance_scale = 4.0
    cfg.gen.image_size = 512
    cfg.data.batch_size_train = 1
    cfg.data.batch_size_val = 1 * num_gpu
    cfg.reward_fn = {
        "hpsv3_remote": 1.0
    }
    cfg.reward_fn_eval = {
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0
    }
    return cfg

def cmaes_qwen_clean_hpsv3_2models_cfg_resume():
    num_gpu = max(1, torch.cuda.device_count())
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_qwen_clean_hpsv3_2models_cfg_resume"
    cfg.experiment.resume_state = "./logs/cmaes_qwen_clean_hpsv3_2models_cfg_2026-01-26_13:14:00/checkpoints/step_5250/cmaes_state_5250.pkl"
    cfg.experiment.eval_orig_model = False
    cfg.model.model_name = "qwen"
    cfg.model.dtype = "bf16"
    cfg.scaleguidance.num_models = 2
    cfg.scaleguidance.use_cfg = True
    cfg.scaleguidance.negative_prompt = ""
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 10
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"
    cfg.data.val_dataset = "data/t2i_compbench_val_random_crop.txt"
    cfg.optimize.population_size = 24  # 3 * 8 = 24 (with 8 gpu) (parallelization is across cma-es candidates)
    cfg.data.save_eval_imgs = True
    cfg.gen.guidance_scale = 4.0
    cfg.gen.image_size = 512
    cfg.data.batch_size_train = 1
    cfg.data.batch_size_val = 1 * num_gpu
    cfg.reward_fn = {
        "hpsv3_remote": 1.0
    }
    cfg.reward_fn_eval = {
        "hpsv3_remote": 1.0,
        "qalign_remote": 1.0
    }
    return cfg

def get_config(name):
    return globals()[name]()
