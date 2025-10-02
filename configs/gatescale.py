import configs.base as base


def cmaes_image_reward():
    cfg = base.get_config()
    cfg.experiment.name = "flux_autoguidance_cmaes_image_reward"
    cfg.gatescale.num_models = 1
    cfg.optimize.max_generations = 50
    cfg.optimize.val_every_steps = 1
    cfg.data.batch_size = 8
    cfg.data.train_dataset = "data/t2i_compbench_train.txt"  # t2i-compbench
    cfg.data.val_dataset = "data/t2i_compbench_val.txt"  # t2i-compbench
    cfg.reward_fn = {
        "imagereward": 1.0,
    }
    return cfg

def get_config(name):
    return globals()[name]()
