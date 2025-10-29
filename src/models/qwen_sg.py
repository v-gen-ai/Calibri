import types
from typing import Sequence, Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np
from diffusers import DiffusionPipeline

from .base_sg import BaseSGPipeline


def extend_qwen_transformer_with_sg(
    model_transformer,
    num_models: int = 1,
    # per-block scales: 4 parameters per block [img_attn, txt_attn, img_mlp, txt_mlp]
    scales_blocks: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    # blending weights across model heads
    models_scales: Optional[Sequence[float]] = None,
):
    """
    Modifies Qwen-Image transformer by adding trainable scaling factors applied to gate outputs.
    
    Each block has 4 gates:
      - img_attn: Image attention gate
      - txt_attn: Text attention gate  
      - img_mlp: Image MLP gate
      - txt_mlp: Text MLP gate
    
    scales_blocks: list[num_models][num_blocks][4] -> [img_attn, txt_attn, img_mlp, txt_mlp]
    """
    n_blocks = len(model_transformer.transformer_blocks)
    
    # Defaults
    if scales_blocks is None:
        scales_blocks = [[[1.0, 1.0, 1.0, 1.0] for _ in range(n_blocks)] for _ in range(num_models)]
    if models_scales is None:
        models_scales = [0.0 for _ in range(num_models)]
        if num_models > 0:
            models_scales[0] = 1.0
    
    # Validations
    assert len(scales_blocks) == num_models, "Length of scales_blocks != num_models"
    assert len(models_scales) == num_models, "Length of models_scales != num_models"
    assert all(len(v) == n_blocks for v in scales_blocks), f"scales_blocks inner length != {n_blocks}"
    assert all(len(v[0]) == 4 for v in scales_blocks), "Each block must have 4 scale parameters"
    
    # Device and dtype
    device = next(model_transformer.parameters()).device
    dtype = next(model_transformer.parameters()).dtype
    
    # Freeze base model
    for p in model_transformer.parameters():
        p.requires_grad = False
    
    # Preserve original forward
    if not hasattr(model_transformer, "_original_forward"):
        model_transformer._original_forward = model_transformer.forward
    
    # Create learnable gate scales
    model_transformer.qwen_gate_scales = nn.ModuleList()
    for m in range(num_models):
        plist = nn.ParameterList(
            [
                nn.ParameterList(
                    [
                        nn.Parameter(torch.tensor(scales_blocks[m][i][j], device=device, dtype=dtype))
                        for j in range(4)  # [img_attn, txt_attn, img_mlp, txt_mlp]
                    ]
                )
                for i in range(n_blocks)
            ]
        )
        model_transformer.qwen_gate_scales.append(plist)
    
    # Blending weights
    initial_scales = torch.tensor(models_scales, device=device, dtype=dtype)
    model_transformer.models_scales = nn.Parameter(initial_scales)
    
    # Hook storage
    model_transformer._gate_hooks = []
    
    def _create_qwen_hook(block_idx: int, model_idx: int, gate_type: str):
        """Hook that applies scales based on gate type"""
        def hook(module, args, kwargs, output):
            if not isinstance(output, (tuple, list)):
                return output
            
            out_list = list(output)
            scales = model_transformer.qwen_gate_scales[model_idx][block_idx]
            
            if gate_type == "attn":
                # For attention: [img_attn, txt_attn]
                if len(out_list) >= 2:
                    out_list[0] = out_list[0] * scales[0]  # img_attn
                    out_list[1] = out_list[1] * scales[1]  # txt_attn
            elif gate_type == "img_mlp":
                # For image MLP
                out_list[0] = out_list[0] * scales[2]  # img_mlp
            elif gate_type == "txt_mlp":
                # For text MLP
                out_list[0] = out_list[0] * scales[3]  # txt_mlp
            
            return tuple(out_list)
        return hook
    
    def _register_hooks(self, model_idx: int):
        for idx, block in enumerate(self.transformer_blocks):
            # Attention hook
            if hasattr(block, 'attn'):
                h1 = block.attn.register_forward_hook(
                    _create_qwen_hook(idx, model_idx, "attn"), with_kwargs=True
                )
                self._gate_hooks.append(h1)
            
            # Image MLP hook
            if hasattr(block, 'img_mlp'):
                h2 = block.img_mlp.register_forward_hook(
                    _create_qwen_hook(idx, model_idx, "img_mlp"), with_kwargs=True
                )
                self._gate_hooks.append(h2)
            
            # Text MLP hook
            if hasattr(block, 'txt_mlp'):
                h3 = block.txt_mlp.register_forward_hook(
                    _create_qwen_hook(idx, model_idx, "txt_mlp"), with_kwargs=True
                )
                self._gate_hooks.append(h3)
    
    def _remove_hooks(self):
        for h in self._gate_hooks:
            h.remove()
        self._gate_hooks = []
    
    def new_forward(self, *args, **kwargs):
        result = None
        # Blend across model heads
        for idx_model in range(len(self.models_scales)):
            scale = self.models_scales[idx_model]
            if scale.abs().item() == 0.0:
                continue
            
            self._register_hooks(idx_model)
            try:
                tmp_res = self._original_forward(*args, **kwargs)
            finally:
                self._remove_hooks()
            
            # Scale and accumulate
            if isinstance(tmp_res, tuple):
                scaled = tuple(scale * t if torch.is_tensor(t) else t for t in tmp_res)
            else:
                scaled = scale * tmp_res
            
            if result is None:
                result = scaled
            else:
                if isinstance(result, tuple):
                    result = tuple((r + s) if torch.is_tensor(r) else r for r, s in zip(result, scaled))
                else:
                    result = result + scaled
        
        # Fallback if all scales were zero
        if result is None:
            result = self._original_forward(*args, **kwargs)
        
        return result
    
    # Bind methods
    model_transformer._register_hooks = types.MethodType(_register_hooks, model_transformer)
    model_transformer._remove_hooks = types.MethodType(_remove_hooks, model_transformer)
    model_transformer.forward = types.MethodType(new_forward, model_transformer)
    
    return model_transformer


class SGQwenPipeline(BaseSGPipeline):
    def __init__(self, device, dtype, model_name="Qwen/Qwen-Image", num_models=1, verbose=False):
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        self.pipeline.transformer = extend_qwen_transformer_with_sg(
            self.pipeline.transformer, num_models=num_models
        )
        if not verbose:
            self.pipeline.set_progress_bar_config(disable=True)
    
    def get_coefficient_shapes(self) -> Dict[str, int]:
        """Get dimensions for Qwen coefficient vector"""
        t = self.pipeline.transformer
        n_blocks = len(t.transformer_blocks)
        n_models = len(t.models_scales)
        
        blocks = n_models * n_blocks * 4  # 4 parameters per block
        models = n_models
        
        return {
            # must have
            "total": blocks + models,
            "n_models": n_models,
            # optional
            "blocks": blocks,
            "n_blocks": n_blocks,
        }
    
    def flatten_coefficients(self) -> np.ndarray:
        t = self.pipeline.transformer
        s = self.get_coefficient_shapes()
        n_blocks, n_models = s["n_blocks"], s["n_models"]
        vec: List[float] = []
        
        for m in range(n_models):
            for b in range(n_blocks):
                scales = t.qwen_gate_scales[m][b]
                vec += [float(scales[i].detach().cpu().item()) for i in range(4)]
        
        vec += t.models_scales.detach().cpu().to(torch.float64).numpy().tolist()
        return np.asarray(vec, dtype=np.float64)
    
    def flat_to_struct(self, x: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if x is None:
            x = self.flatten_coefficients()
        s = self.get_coefficient_shapes()
        d_blocks = s["blocks"]
        n_blocks, n_models = s["n_blocks"], s["n_models"]
        blocks = x[:d_blocks]
        models = x[d_blocks:]
        
        scales_blocks: List[List[List[float]]] = []
        per_m_blocks = n_blocks * 4
        for m in range(n_models):
            start = m * per_m_blocks
            mb = []
            for b in range(n_blocks):
                block_start = start + b * 4
                mb.append([
                    float(blocks[block_start + i]) for i in range(4)
                ])
            scales_blocks.append(mb)
        
        models_scales = [float(v) for v in models]
        
        return {
            "num_models": n_models,
            "scales_blocks": scales_blocks,
            "models_scales": models_scales,
        }
    
    def struct_to_flat(self, sdict: Dict[str, Any]) -> np.ndarray:
        s = self.get_coefficient_shapes()
        n_blocks, n_models = s["n_blocks"], s["n_models"]
        sb, ms = sdict["scales_blocks"], sdict["models_scales"]
        assert len(sb) == n_models and len(ms) == n_models
        
        vec: List[float] = []
        for m in range(n_models):
            assert len(sb[m]) == n_blocks and all(len(p) == 4 for p in sb[m])
            for b in range(n_blocks):
                vec += [float(v) for v in sb[m][b]]
        vec += [float(v) for v in ms]
        return np.asarray(vec, dtype=np.float64)
    
    def apply_coefficients(self, x: np.ndarray) -> None:
        kwargs = self.flat_to_struct(x)
        self.pipeline.transformer = extend_qwen_transformer_with_sg(self.pipeline.transformer, **kwargs)
    
    def __call__(self, prompts, num_inference_steps, guidance_scale, height, width, generator, **kwargs):
        return self.pipeline(
            prompt=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            **kwargs
        )
