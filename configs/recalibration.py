import configs.base as base


def cmaes_image_reward():
    cfg = base.get_config()
    cfg.experiment.name = "flux_autoguidance_cmaes_image_reward"
    cfg.optimize.max_generations = 10
    cfg.optimize.val_every_steps = 2
    cfg.data.batch_size = 4
    cfg.reward_fn = "image_reward"  # to-implement dynamic import with multi-score (from flow-grpo)
    cfg.data.train_dataset = ""
    cfg.data.val_dataset = ""
    return cfg

def get_config(name):
    return globals()[name]()
