import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def create_writer(log_dir, run_name):
    return SummaryWriter(log_dir=f"{log_dir}/{run_name}")

def log_scalars(writer: SummaryWriter, scalars: dict, step: int, prefix: str = ""):
    for k, v in scalars.items():
        writer.add_scalar(f"{prefix}{k}", v, step)

def log_hist_alphas(writer: SummaryWriter, alpha_dict: dict, step: int, prefix: str):
    # alpha_dict: {'double_alpha_attn': np.array([...]), ...}
    for name, arr in alpha_dict.items():
        arr = np.asarray(arr)
        writer.add_histogram(f"{prefix}{name}", arr, step)

def log_images(writer: SummaryWriter, tag: str, pil_images, step: int, max_images=4):
    # склеивание не делаем, логируем первые N изображений поштучно
    for i, im in enumerate(pil_images[:max_images]):
        writer.add_image(f"{tag}/{i}", torch.from_numpy(np.array(im)).permute(2,0,1), step)
