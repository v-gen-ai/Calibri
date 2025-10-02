import configs.base as base


def cmaes_image_reward():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_imagereward"
    cfg.gatescale.num_models = 1
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 1
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val.txt"  # t2i-compbench
    cfg.optimize.population_size = 2
    cfg.reward_fn = {
        "imagereward": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
    }
    return cfg

def cmaes_image_reward_2models():
    cfg = base.get_config()
    cfg.experiment.name = "cmaes_imagereward"
    cfg.gatescale.num_models = 2
    cfg.optimize.max_generations = -1
    cfg.optimize.val_every_steps = 1
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val.txt"  # t2i-compbench
    cfg.optimize.population_size = 2
    cfg.reward_fn = {
        "imagereward": 1.0,
    }
    cfg.reward_fn_eval = {
        "imagereward": 1.0,
    }
    return cfg

def get_config(name):
    return globals()[name]()
