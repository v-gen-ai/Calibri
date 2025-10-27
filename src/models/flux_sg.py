import types
from typing import Sequence, Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np
from diffusers import FluxPipeline

from .base_sg import BaseSGPipeline


def extend_transformer_with_sg(
    model_transformer,
    num_models: int = 1,
    scales_double: Sequence[Sequence[tuple[float, float]]] = None,
    scales_single: Sequence[Sequence[float]] = None,
    models_scales: Sequence[float] = None
):
    """
    Modifies the transformer by adding trainable scaling factors for gates
    in double blocks (attn, mlp) and single blocks, with separate initialization for each block.

    Scaling order:
    1) All double blocks in order, each containing a tuple (w_attn, w_mlp).
    2) All single blocks in order, each containing a scalar w.

    scales_double: list[num_models][num_double_blocks] -> (attn, mlp)
    scales_single: list[num_models][num_single_blocks] -> w
    """
    n_double = len(model_transformer.transformer_blocks)
    n_single = len(model_transformer.single_transformer_blocks)

    # Normalize/validate scales_double
    if scales_double is None:
        scales_double = [[(1.0, 1.0) for _ in range(n_double)] for _ in range(num_models)]
    
    if scales_single is None:
        scales_single = [[1.0 for _ in range(n_single)] for _ in range(num_models)]
    
    if models_scales is None:
        models_scales = [0.0 for _ in range(num_models)]
        models_scales[0] = 1.0

    assert len(scales_double) == num_models, "Length of scales_double doesn't correspond to num_models"
    assert len(scales_single) == num_models, "Length of scales_single doesn't correspond to num_models"
    assert len(models_scales) == num_models, "Length of models_scales doesn't correspond to num_models"
    assert len(scales_double[0]) == n_double, "Shape of scales_double doesn't correspond to number of double blocks in FLUX"
    assert len(scales_single[0]) == n_single, "Shape of scales_double doesn't correspond to number of double blocks in FLUX"
    assert all([len(elem) == 2 for vec_model in scales_double for elem in vec_model]), "Each double block need 2 scale params"
    
    device = next(model_transformer.parameters()).device
    dtype = next(model_transformer.parameters()).dtype

    # Freeze base model
    for p in model_transformer.parameters():
        p.requires_grad = False

    # Preserve original forward
    if not hasattr(model_transformer, "_original_forward"):
        model_transformer._original_forward = model_transformer.forward

    # Create per-block learnable gate scales
    model_transformer.transformer_gate_scales = nn.ModuleList()
    for model_idx in range(num_models):
        model_coeffs = nn.ModuleList()
        for block_idx in range(n_double):
            attn_init, mlp_init = scales_double[model_idx][block_idx]
            block_coeffs = nn.ParameterList([
                nn.Parameter(torch.tensor(attn_init, device=device, dtype=dtype)),
                nn.Parameter(torch.tensor(mlp_init, device=device, dtype=dtype)),
            ])
            model_coeffs.append(block_coeffs)
        model_transformer.transformer_gate_scales.append(model_coeffs)

    model_transformer.single_gate_scales = nn.ModuleList()
    for m in range(num_models):
        plist = nn.ParameterList([
            nn.Parameter(torch.tensor(scales_single[m][b], device=device, dtype=dtype))
            for b in range(n_single)
        ])
        model_transformer.single_gate_scales.append(plist)

    # Blending weights for models
    initial_scales = torch.tensor(models_scales, device=device, dtype=dtype)
    model_transformer.models_scales = nn.Parameter(initial_scales)

    # Hook storage
    model_transformer._gate_hooks = []

    def _create_transformer_hook(block_idx, model_idx):
        def hook(module, args, kwargs, output):
            w_attn, w_mlp = model_transformer.transformer_gate_scales[model_idx][block_idx]
            if isinstance(output, (tuple, list)):
                out_list = list(output)
                # assumes positions [1] -> gate_msa, [4] -> gate_mlp
                out_list[1] = out_list[1] * w_attn
                out_list[4] = out_list[4] * w_mlp
                return tuple(out_list)
            return output
        return hook

    def _create_single_hook(block_idx, model_idx):
        def hook(module, args, kwargs, output):
            w = model_transformer.single_gate_scales[model_idx][block_idx]
            if isinstance(output, (tuple, list)):
                out_list = list(output)
                # assumes position [1] -> gate
                out_list[1] = out_list[1] * w
                return tuple(out_list)
            return output
        return hook

    def _register_hooks(self, model_idx):
        for idx, block in enumerate(self.transformer_blocks):
            h1 = block.norm1.register_forward_hook(_create_transformer_hook(idx, model_idx), with_kwargs=True)
            self._gate_hooks.append(h1)
            if hasattr(block, "norm1_context") and block.norm1_context is not None:
                h2 = block.norm1_context.register_forward_hook(_create_transformer_hook(idx, model_idx), with_kwargs=True)
                self._gate_hooks.append(h2)
        for idx, block in enumerate(self.single_transformer_blocks):
            h = block.norm.register_forward_hook(_create_single_hook(idx, model_idx), with_kwargs=True)
            self._gate_hooks.append(h)

    def _remove_hooks(self):
        for h in self._gate_hooks:
            h.remove()
        self._gate_hooks = []

    def new_forward(self, *args, **kwargs):
        result = None
        for idx_model in range(len(self.models_scales)):
            scale = self.models_scales[idx_model]
            self._register_hooks(idx_model)
            try:
                tmp_res = self._original_forward(*args, **kwargs)
            finally:
                self._remove_hooks()
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
        return result

    model_transformer._register_hooks = types.MethodType(_register_hooks, model_transformer)
    model_transformer._remove_hooks = types.MethodType(_remove_hooks, model_transformer)
    model_transformer.forward = types.MethodType(new_forward, model_transformer)
    return model_transformer


class SGFluxPipeline(BaseSGPipeline):
    def __init__(self, device, dtype, model_name="black-forest-labs/FLUX.1-dev", num_models=1, verbose=False):
        self.pipeline = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        self.pipeline.transformer = extend_transformer_with_sg(self.pipeline.transformer, num_models=num_models)
        if not verbose:
            self.pipeline.set_progress_bar_config(disable=True)
    
    def get_coefficient_shapes(self) -> Dict[str, int]:
        """Get dimensions for FLUX coefficient vector"""
        t = self.pipeline.transformer
        n_double = len(t.transformer_blocks)
        n_single = len(t.single_transformer_blocks)
        n_models = len(t.models_scales)
        
        doubles = n_models * n_double * 2
        singles = n_models * n_single
        models = n_models
        
        return {
            # must have
            "total": doubles + singles + models,
            "n_models": n_models,
            # optional
            "doubles": doubles,
            "singles": singles,
            "n_double": n_double,
            "n_single": n_single,
        }

    def flatten_coefficients(self) -> np.ndarray:
        # как раньше: doubles(attn,mlp) -> singles -> models_scales  
        t = self.pipeline.transformer
        s = self.get_coefficient_shapes()
        n_double, n_single, n_models = s["n_double"], s["n_single"], s["n_models"]
        vec: List[float] = []
        for m in range(n_models):
            for b in range(n_double):
                w_attn, w_mlp = t.transformer_gate_scales[m][b]
                vec += [float(w_attn.detach().cpu().item()), float(w_mlp.detach().cpu().item())]
        for m in range(n_models):
            for b in range(n_single):
                vec.append(float(t.single_gate_scales[m][b].detach().cpu().item()))
        vec += t.models_scales.detach().cpu().to(torch.float64).numpy().tolist()
        return np.asarray(vec, dtype=np.float64)

    def flat_to_struct(self, x: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if x is None:
            x = self.flatten_coefficients()
        s = self.get_coefficient_shapes()
        d_d, d_s = s["doubles"], s["singles"]
        n_double, n_single, n_models = s["n_double"], s["n_single"], s["n_models"]
        doubles = x[:d_d]; singles = x[d_d:d_d + d_s]; models = x[d_d + d_s:]  

        scales_double: List[List[tuple]] = []
        per_model_doubles = n_double * 2
        for m in range(n_models):
            start = m * per_model_doubles
            md = [(float(doubles[start + 2*b + 0]), float(doubles[start + 2*b + 1])) for b in range(n_double)]
            scales_double.append(md)
        scales_single: List[List[float]] = []
        for m in range(n_models):
            start = m * n_single
            ms = [float(singles[start + b]) for b in range(n_single)]
            scales_single.append(ms)
        models_scales = [float(v) for v in models]  

        return {
            "num_models": n_models,
            "scales_double": scales_double,
            "scales_single": scales_single,
            "models_scales": models_scales,
        }

    def struct_to_flat(self, sdict: Dict[str, Any]) -> np.ndarray:
        s = self.get_coefficient_shapes()
        n_double, n_single, n_models = s["n_double"], s["n_single"], s["n_models"]
        sd = sdict["scales_double"]; ss = sdict["scales_single"]; ms = sdict["models_scales"]
        assert len(sd) == n_models and len(ss) == n_models and len(ms) == n_models  
        vec: List[float] = []
        for m in range(n_models):
            assert len(sd[m]) == n_double and all(len(p) == 2 for p in sd[m])
            for b in range(n_double):
                vec += [float(sd[m][b][0]), float(sd[m][b][1])]
        for m in range(n_models):
            assert len(ss[m]) == n_single
            vec += [float(v) for v in ss[m]]
        vec += [float(v) for v in ms]
        return np.asarray(vec, dtype=np.float64)

    def apply_coefficients(self, x: np.ndarray) -> None:
        kwargs = self.flat_to_struct(x)
        self.pipeline.transformer = extend_transformer_with_sg(self.pipeline.transformer, **kwargs)

    def __call__(self, prompts, num_inference_steps, guidance_scale, height, width, generator, **kwargs):
        return self.pipeline(prompts,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=guidance_scale,
                             height=height,
                             width=width,
                             generator=generator,
                             **kwargs)
