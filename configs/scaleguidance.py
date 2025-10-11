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


def cmaes_hpsv3_2models():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_hpsv3_2models"
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



def get_config(name):
    return globals()[name]()
