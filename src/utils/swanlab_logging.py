import os
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from typing import Dict, List, Optional, Union
import json

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("SwanLab not available. Install with: pip install swanlab")

from .logging_tb import SummaryWriter, NullWriter


class SwanLabWriter:
    """SwanLab writer wrapper"""
    
    def __init__(self, project_name: str = "ScaleGuidance", experiment_name: str = None, 
                 api_key: str = None, log_dir: str = None):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.api_key = api_key
        self.log_dir = log_dir
        self.initialized = False
        
        if SWANLAB_AVAILABLE:
            self._init_swanlab()
        else:
            print("Warning: SwanLab not available, using NullWriter")
    
    def _init_swanlab(self):
        """Initialize SwanLab"""
        try:
            # Set API key if provided
            if self.api_key:
                os.environ['SWANLAB_API_KEY'] = self.api_key
            
            # Initialize SwanLab
            swanlab.init(
                project=self.project_name,
                experiment_name=self.experiment_name,
                config={
                    "model": "ScaleGuidance",
                    "framework": "PyTorch + CMA-ES"
                }
            )
            self.initialized = True
            print(f"✓ SwanLab initialized: {self.project_name}/{self.experiment_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize SwanLab: {e}")
            self.initialized = False
    
    def add_scalar(self, tag: str, value: float, step: int = None, **kwargs):
        """Log scalar value"""
        if self.initialized:
            try:
                swanlab.log({tag: value}, step=step)
            except Exception as e:
                print(f"Warning: Failed to log scalar {tag}: {e}")
    
    def add_image(self, tag: str, img_tensor: torch.Tensor, step: int = None, **kwargs):
        """Log image"""
        if self.initialized:
            try:
                # Convert tensor to PIL Image
                if isinstance(img_tensor, torch.Tensor):
                    if img_tensor.dim() == 4:  # Batch of images
                        img_tensor = img_tensor[0]  # Take first image
                    if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # CHW format
                        img_tensor = img_tensor.permute(1, 2, 0)  # Convert to HWC
                    img_tensor = (img_tensor * 255).clamp(0, 255).to(torch.uint8)
                    img_array = img_tensor.cpu().numpy()
                    img = Image.fromarray(img_array)
                else:
                    img = img_tensor
                
                swanlab.log({tag: swanlab.Image(img)}, step=step)
            except Exception as e:
                print(f"Warning: Failed to log image {tag}: {e}")
    
    def add_images(self, tag: str, img_tensor: torch.Tensor, step: int = None, max_images: int = 8, **kwargs):
        """Log multiple images"""
        if self.initialized:
            try:
                if img_tensor.dim() == 4:  # Batch of images
                    for i, img in enumerate(img_tensor[:max_images]):
                        self.add_image(f"{tag}/{i}", img, step)
                else:
                    self.add_image(tag, img_tensor, step)
            except Exception as e:
                print(f"Warning: Failed to log images {tag}: {e}")
    
    def add_histogram(self, tag: str, values: np.ndarray, step: int = None, **kwargs):
        """Log histogram"""
        if self.initialized:
            try:
                swanlab.log({tag: swanlab.Histogram(values)}, step=step)
            except Exception as e:
                print(f"Warning: Failed to log histogram {tag}: {e}")
    
    def add_figure(self, tag: str, figure: plt.Figure, step: int = None, **kwargs):
        """Log matplotlib figure"""
        if self.initialized:
            try:
                # Convert figure to image
                buf = io.BytesIO()
                figure.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
                swanlab.log({tag: swanlab.Image(img)}, step=step)
                plt.close(figure)
            except Exception as e:
                print(f"Warning: Failed to log figure {tag}: {e}")
    
    def add_text(self, tag: str, text: str, step: int = None, **kwargs):
        """Log text"""
        if self.initialized:
            try:
                swanlab.log({tag: text}, step=step)
            except Exception as e:
                print(f"Warning: Failed to log text {tag}: {e}")
    
    def add_config(self, config: Dict):
        """Log configuration"""
        if self.initialized:
            try:
                swanlab.config.update(config)
            except Exception as e:
                print(f"Warning: Failed to log config: {e}")
    
    def close(self):
        """Close SwanLab"""
        if self.initialized:
            try:
                swanlab.finish()
            except Exception as e:
                print(f"Warning: Failed to close SwanLab: {e}")


class DualWriter:
    """Writer that logs to both TensorBoard and SwanLab"""
    
    def __init__(self, tb_writer: SummaryWriter, swanlab_writer: SwanLabWriter):
        self.tb_writer = tb_writer
        self.swanlab_writer = swanlab_writer
    
    def add_scalar(self, tag: str, value: float, step: int = None, **kwargs):
        self.tb_writer.add_scalar(tag, value, step, **kwargs)
        self.swanlab_writer.add_scalar(tag, value, step, **kwargs)
    
    def add_image(self, tag: str, img_tensor: torch.Tensor, step: int = None, **kwargs):
        self.tb_writer.add_image(tag, img_tensor, step, **kwargs)
        self.swanlab_writer.add_image(tag, img_tensor, step, **kwargs)
    
    def add_images(self, tag: str, img_tensor: torch.Tensor, step: int = None, max_images: int = 8, **kwargs):
        self.tb_writer.add_images(tag, img_tensor, step, max_images, **kwargs)
        self.swanlab_writer.add_images(tag, img_tensor, step, max_images, **kwargs)
    
    def add_histogram(self, tag: str, values: np.ndarray, step: int = None, **kwargs):
        self.tb_writer.add_histogram(tag, values, step, **kwargs)
        self.swanlab_writer.add_histogram(tag, values, step, **kwargs)
    
    def add_figure(self, tag: str, figure: plt.Figure, step: int = None, **kwargs):
        self.tb_writer.add_figure(tag, figure, step, **kwargs)
        self.swanlab_writer.add_figure(tag, figure, step, **kwargs)
    
    def add_text(self, tag: str, text: str, step: int = None, **kwargs):
        self.tb_writer.add_text(tag, text, step, **kwargs)
        self.swanlab_writer.add_text(tag, text, step, **kwargs)
    
    def add_config(self, config: Dict):
        self.swanlab_writer.add_config(config)
    
    def close(self):
        self.tb_writer.close()
        self.swanlab_writer.close()


def create_swanlab_writer(project_name: str = "ScaleGuidance", 
                         experiment_name: str = None,
                         api_key: str = None,
                         log_dir: str = None) -> SwanLabWriter:
    """Create SwanLab writer"""
    return SwanLabWriter(project_name, experiment_name, api_key, log_dir)


def create_dual_writer(tb_log_dir: str,
                      project_name: str = "ScaleGuidance",
                      experiment_name: str = None,
                      api_key: str = None) -> DualWriter:
    """Create dual writer (TensorBoard + SwanLab)"""
    from .logging_tb import create_writer
    
    tb_writer = create_writer(tb_log_dir)
    swanlab_writer = create_swanlab_writer(project_name, experiment_name, api_key, tb_log_dir)
    
    return DualWriter(tb_writer, swanlab_writer)


def log_scalars_swanlab(writer: Union[SwanLabWriter, DualWriter], scalars: Dict[str, float], 
                       step: int, prefix: str = ""):
    """Log scalars to SwanLab"""
    for k, v in scalars.items():
        writer.add_scalar(f"{prefix}{k}", v, step)


def log_images_swanlab(writer: Union[SwanLabWriter, DualWriter], tag: str, 
                      pil_images: List[Image.Image], step: int, max_images: int = 8):
    """Log images to SwanLab"""
    for i, im in enumerate(pil_images[:max_images]):
        writer.add_image(f"{tag}/{i}", im, step)


def log_model_scales_swanlab(writer: Union[SwanLabWriter, DualWriter], alpha_dict: Dict, 
                            step: int, prefix: str = ""):
    """Log model scales to SwanLab"""
    if "models_scales" in alpha_dict:
        ms = np.asarray(alpha_dict["models_scales"])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(np.arange(len(ms)), ms, color='blue')
        ax.set_xlabel('Model Index')
        ax.set_ylabel('Scale Value')
        ax.set_title('Model Scales')
        ax.grid(True)
        
        writer.add_figure(f"{prefix}models_scales", fig, step)


def log_scatter_swanlab(writer: Union[SwanLabWriter, DualWriter], alpha_dict: Dict, 
                       step: int, prefix: str = ""):
    """Log scatter plots to SwanLab"""
    # Similar to log_scatter in logging_tb.py but for SwanLab
    is_multi = isinstance(alpha_dict.get("double_attn"), (list, tuple)) \
               and isinstance(alpha_dict.get("double_mlp"), (list, tuple)) \
               and isinstance(alpha_dict.get("single"), (list, tuple))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    if is_multi:
        attn_by_model = [np.asarray(x) for x in alpha_dict["double_attn"]]
        mlp_by_model = [np.asarray(x) for x in alpha_dict["double_mlp"]]
        single_by_model = [np.asarray(x) for x in alpha_dict["single"]]
        n_models = max(len(attn_by_model), len(mlp_by_model), len(single_by_model))
        
        # Attention plot
        fig_attn, ax1 = plt.subplots(figsize=(8, 5))
        for m_idx in range(n_models):
            if m_idx < len(attn_by_model):
                a_arr = attn_by_model[m_idx]
                ax1.scatter(range(len(a_arr)), a_arr, alpha=0.6,
                           color=colors[m_idx % len(colors)], marker='o', label=f"Model {m_idx}")
        ax1.set_xlabel('Block Index')
        ax1.set_ylabel('W_attn Value')
        ax1.set_title('Transformer Blocks: Attention Gate Coefficients')
        ax1.grid(True)
        ax1.legend()
        writer.add_figure(f"{prefix}double_attn", fig_attn, step)
        
        # MLP plot
        fig_mlp, ax2 = plt.subplots(figsize=(8, 5))
        for m_idx in range(n_models):
            if m_idx < len(mlp_by_model):
                m_arr = mlp_by_model[m_idx]
                ax2.scatter(range(len(m_arr)), m_arr, alpha=0.6,
                           color=colors[m_idx % len(colors)], marker='o', label=f"Model {m_idx}")
        ax2.set_xlabel('Block Index')
        ax2.set_ylabel('W_mlp Value')
        ax2.set_title('Transformer Blocks: MLP Gate Coefficients')
        ax2.grid(True)
        ax2.legend()
        writer.add_figure(f"{prefix}double_mlp", fig_mlp, step)
        
        # Single plot
        fig_single, ax3 = plt.subplots(figsize=(12, 5))
        for m_idx in range(n_models):
            if m_idx < len(single_by_model):
                s_arr = single_by_model[m_idx]
                ax3.scatter(range(len(s_arr)), s_arr, alpha=0.6,
                           color=colors[m_idx % len(colors)], marker='o', label=f"Model {m_idx}")
        ax3.set_xlabel('Block Index')
        ax3.set_ylabel('W Value')
        ax3.set_title('Single Blocks: Gate Coefficients')
        ax3.grid(True)
        ax3.legend()
        writer.add_figure(f"{prefix}single", fig_single, step)
    
    # Model scales
    if "models_scales" in alpha_dict:
        ms = np.asarray(alpha_dict["models_scales"])
        fig_ms, ax = plt.subplots(figsize=(8, 5))
        ax.bar(np.arange(len(ms)), ms, color='blue')
        ax.set_xlabel('Model Index')
        ax.set_ylabel('Scale Value')
        ax.set_title('Model Scales')
        ax.grid(True)
        writer.add_figure(f"{prefix}models_scales", fig_ms, step)
