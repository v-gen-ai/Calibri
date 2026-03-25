import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Import SwanLab logging if available
try:
    from .swanlab_logging import SwanLabWriter, DualWriter, create_swanlab_writer, create_dual_writer
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


class NullWriter:
    def add_scalar(self, *args, **kwargs): 
        pass
    def add_image(self, *args, **kwargs): 
        pass
    def close(self): 
        pass


def create_writer(log_dir, use_swanlab=False, project_name="ScaleGuidance", 
                 experiment_name=None, api_key=None):
    """Create writer with optional SwanLab support"""
    if use_swanlab and SWANLAB_AVAILABLE:
        return create_dual_writer(log_dir, project_name, experiment_name, api_key)
    else:
        return SummaryWriter(log_dir=log_dir)

def log_scalars(writer, scalars: dict, step: int, prefix: str = ""):
    for k, v in scalars.items():
        writer.add_scalar(f"{prefix}{k}", v, step)

def log_hist_alphas(writer, alpha_dict: dict, step: int, prefix: str):
    # alpha_dict: {'double_alpha_attn': np.array([...]), ...}
    for name, arr in alpha_dict.items():
        arr = np.asarray(arr)
        writer.add_histogram(f"{prefix}{name}", arr, step)

def log_model_scales(writer, alpha_dict: dict, step: int, prefix: str):
    def _save_fig_to_tb(fig, tag):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        writer.add_image(tag, torch.from_numpy(np.array(img)).permute(2, 0, 1), step)

    if "models_scales" in alpha_dict:
        ms = np.asarray(alpha_dict["models_scales"])
        fig_ms, ax = plt.subplots(figsize=(8, 5))
        ax.bar(np.arange(len(ms)), ms, color='blue')
        ax.set_xlabel('Model Index'); ax.set_ylabel('Scale Value')
        ax.set_title('Model Scales'); ax.grid(True)
        _save_fig_to_tb(fig_ms, f"{prefix}models_scales")

def log_images(writer, tag: str, pil_images, step: int, max_images=8):
    for i, im in enumerate(pil_images[:max_images]):
        writer.add_image(f"{tag}/{i}", torch.from_numpy(np.array(im)).permute(2,0,1), step)

def log_scatter(writer, alpha_dict: dict, step: int, prefix: str):

    is_multi = isinstance(alpha_dict.get("double_attn"), (list, tuple)) \
               and isinstance(alpha_dict.get("double_mlp"), (list, tuple)) \
               and isinstance(alpha_dict.get("single"), (list, tuple))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    def _save_fig_to_tb(fig, tag):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        writer.add_image(tag, torch.from_numpy(np.array(img)).permute(2, 0, 1), step)

    if is_multi:
        attn_by_model = [np.asarray(x) for x in alpha_dict["double_attn"]]
        mlp_by_model  = [np.asarray(x) for x in alpha_dict["double_mlp"]]
        single_by_model = [np.asarray(x) for x in alpha_dict["single"]]
        n_models = max(len(attn_by_model), len(mlp_by_model), len(single_by_model))

        num_transformer_blocks = max((len(x) for x in attn_by_model), default=0)
        num_single_blocks = max((len(x) for x in single_by_model), default=0)

        fig_attn, ax1 = plt.subplots(figsize=(8, 5))
        for m_idx in range(n_models):
            if m_idx < len(attn_by_model):
                a_arr = attn_by_model[m_idx]
                ax1.scatter(range(len(a_arr)), a_arr, alpha=0.6,
                            color=colors[m_idx % len(colors)], marker='o', label=f"Model {m_idx}")
        ax1.set_xlabel('Block Index'); ax1.set_ylabel('W_attn Value')
        ax1.set_title('Transformer Blocks: Attention Gate Coefficients')
        if num_transformer_blocks > 0:
            ax1.set_xticks(list(range(num_transformer_blocks)))
        ax1.grid(True); ax1.legend()
        _save_fig_to_tb(fig_attn, f"{prefix}double_attn")

        # Отдельный график: MLP
        fig_mlp, ax2 = plt.subplots(figsize=(8, 5))
        for m_idx in range(n_models):
            if m_idx < len(mlp_by_model):
                m_arr = mlp_by_model[m_idx]
                ax2.scatter(range(len(m_arr)), m_arr, alpha=0.6,
                            color=colors[m_idx % len(colors)], marker='o', label=f"Model {m_idx}")
        ax2.set_xlabel('Block Index'); ax2.set_ylabel('W_mlp Value')
        ax2.set_title('Transformer Blocks: MLP Gate Coefficients')
        if num_transformer_blocks > 0:
            ax2.set_xticks(list(range(num_transformer_blocks)))
        ax2.grid(True); ax2.legend()
        _save_fig_to_tb(fig_mlp, f"{prefix}double_mlp")

        fig_single, ax3 = plt.subplots(figsize=(12, 5))
        for m_idx in range(n_models):
            if m_idx < len(single_by_model):
                s_arr = single_by_model[m_idx]
                ax3.scatter(range(len(s_arr)), s_arr, alpha=0.6,
                            color=colors[m_idx % len(colors)], marker='o', label=f"Model {m_idx}")
        ax3.set_xlabel('Block Index'); ax3.set_ylabel('W Value')
        ax3.set_title('Single Blocks: Gate Coefficients')
        if num_single_blocks > 0:
            ax3.set_xticks(list(range(num_single_blocks)))
        ax3.grid(True); ax3.legend()
        _save_fig_to_tb(fig_single, f"{prefix}single")

        if "models_scales" in alpha_dict:
            ms = np.asarray(alpha_dict["models_scales"])
            fig_ms, ax = plt.subplots(figsize=(8, 5))
            ax.bar(np.arange(len(ms)), ms, color='blue')
            ax.set_xlabel('Model Index'); ax.set_ylabel('Scale Value')
            ax.set_title('Model Scales'); ax.grid(True)
            _save_fig_to_tb(fig_ms, f"{prefix}models_scales")

    else:
        for name, arr in alpha_dict.items():
            arr = np.asarray(arr)
            fig, ax = plt.subplots(figsize=(8, 5))
            if name == "models_scales":
                ax.bar(np.arange(len(arr)), arr, color='blue')
                ax.set_xlabel("Index"); ax.set_ylabel("Value")
                ax.set_title(f"Bar plot for {name}")
            else:
                ax.scatter(np.arange(len(arr)), arr, c='red')
                ax.set_xlabel("Block Index"); ax.set_ylabel("Value")
                ax.set_title(f"Scatter plot for {name}")
            ax.grid(True)
            _save_fig_to_tb(fig, f"{prefix}{name}")
