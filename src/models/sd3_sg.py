import types
from typing import Sequence, Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np

from diffusers import DiffusionPipeline

from .base_sg import BaseSGPipeline


def extend_sd3_transformer_with_sg(
    model_transformer,
    num_models: int = 1,
    # per-block scales: automatically sized based on block flags
    scales_main: Optional[Sequence[Sequence[tuple[float, float]]]] = None,
    scales_context: Optional[Sequence[Sequence[tuple[float, float]]]] = None,
    scales_attn2: Optional[Sequence[Sequence[float]]] = None,
    # blending weights across model heads
    models_scales: Optional[Sequence[float]] = None,
):
    """
    Modifies SD3 transformer by adding trainable scaling factors applied to gate outputs.
    
    Scales are created only where gates actually exist based on block configuration:
      - main: always present (gate_msa, gate_mlp)
      - context: only when not context_pre_only (c_gate_msa, c_gate_mlp)
      - attn2: only when use_dual_attention (gate_msa2)
    
    Hook automatically detects which gates to scale using block flags.
    """
    n_blocks = len(model_transformer.transformer_blocks)
    
    # Detect which blocks have context gates and dual attention
    has_context_gates = []
    has_dual_attn = []
    for block in model_transformer.transformer_blocks:
        has_context_gates.append(not block.context_pre_only)
        has_dual_attn.append(block.use_dual_attention)
    
    # Count total parameters needed
    n_main = n_blocks  # always present
    n_context = sum(has_context_gates)
    n_attn2 = sum(has_dual_attn)
    
    # Defaults
    if scales_main is None:
        scales_main = [[(1.0, 1.0) for _ in range(n_main)] for _ in range(num_models)]
    if scales_context is None:
        scales_context = [[(1.0, 1.0) for _ in range(n_context)] for _ in range(num_models)]
    if scales_attn2 is None:
        scales_attn2 = [[1.0 for _ in range(n_attn2)] for _ in range(num_models)]
    if models_scales is None:
        models_scales = [0.0 for _ in range(num_models)]
        if num_models > 0:
            models_scales[0] = 1.0
    
    # Validations
    assert len(scales_main) == num_models, "Length of scales_main != num_models"
    assert len(scales_context) == num_models, "Length of scales_context != num_models"
    assert len(scales_attn2) == num_models, "Length of scales_attn2 != num_models"
    assert len(models_scales) == num_models, "Length of models_scales != num_models"
    assert all(len(v) == n_main for v in scales_main), f"scales_main inner length != {n_main}"
    assert all(len(v) == n_context for v in scales_context), f"scales_context inner length != {n_context}"
    assert all(len(v) == n_attn2 for v in scales_attn2), f"scales_attn2 inner length != {n_attn2}"
    
    # Device and dtype
    device = next(model_transformer.parameters()).device
    dtype = next(model_transformer.parameters()).dtype
    
    # Freeze base model
    for p in model_transformer.parameters():
        p.requires_grad = False
    
    # Preserve original forward
    if not hasattr(model_transformer, "_original_forward"):
        model_transformer._original_forward = model_transformer.forward
    
    # Create learnable gate scales - main (always present)
    model_transformer.sd3_gate_scales_main = nn.ModuleList()
    for m in range(num_models):
        plist = nn.ParameterList(
            [
                nn.ParameterList(
                    [
                        nn.Parameter(torch.tensor(scales_main[m][i][0], device=device, dtype=dtype)),
                        nn.Parameter(torch.tensor(scales_main[m][i][1], device=device, dtype=dtype)),
                    ]
                )
                for i in range(n_main)
            ]
        )
        model_transformer.sd3_gate_scales_main.append(plist)
    
    # Create learnable gate scales - context (only where not context_pre_only)
    model_transformer.sd3_gate_scales_context = nn.ModuleList()
    for m in range(num_models):
        plist = nn.ParameterList(
            [
                nn.ParameterList(
                    [
                        nn.Parameter(torch.tensor(scales_context[m][i][0], device=device, dtype=dtype)),
                        nn.Parameter(torch.tensor(scales_context[m][i][1], device=device, dtype=dtype)),
                    ]
                )
                for i in range(n_context)
            ]
        )
        model_transformer.sd3_gate_scales_context.append(plist)
    
    # Create learnable gate scales - attn2 (only where use_dual_attention)
    model_transformer.sd3_gate_scales_attn2 = nn.ModuleList()
    for m in range(num_models):
        plist = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(scales_attn2[m][i], device=device, dtype=dtype))
                for i in range(n_attn2)
            ]
        )
        model_transformer.sd3_gate_scales_attn2.append(plist)
    
    # Blending weights
    initial_scales = torch.tensor(models_scales, device=device, dtype=dtype)
    model_transformer.models_scales = nn.Parameter(initial_scales)
    
    # Store block flags and indexing
    model_transformer._has_context_gates = has_context_gates
    model_transformer._has_dual_attn = has_dual_attn
    model_transformer._gate_hooks = []
    
    # Universal hook that checks block flags
    def _create_universal_hook(block_idx: int, model_idx: int, context: bool):
        """Hook that applies scales based on what gates exist in the block"""
        def hook(module, args, kwargs, output):
            if not isinstance(output, (tuple, list)):
                return output
            
            out_list = list(output)
            
            if context:
                # Context branch: only scale if has gates (not context_pre_only)
                if model_transformer._has_context_gates[block_idx]:
                    # c_gate_msa [1] and c_gate_mlp [4]
                    # Find index in context scales (count how many context-gate blocks before this)
                    context_idx = sum(model_transformer._has_context_gates[:block_idx])
                    w_c_attn, w_c_mlp = model_transformer.sd3_gate_scales_context[model_idx][context_idx]
                    out_list[1] = out_list[1] * w_c_attn
                    out_list[4] = out_list[4] * w_c_mlp

            else:
                # Main branch: always has gate_msa [1] and gate_mlp [4]
                w_attn, w_mlp = model_transformer.sd3_gate_scales_main[model_idx][block_idx]
                out_list[1] = out_list[1] * w_attn
                out_list[4] = out_list[4] * w_mlp
                
                # Check for dual attention gate_msa2 at position [6]
                if model_transformer._has_dual_attn[block_idx] and len(out_list) > 6:
                    # Find index in attn2 scales (count how many dual-attn blocks before this)
                    attn2_idx = sum(model_transformer._has_dual_attn[:block_idx])
                    w_attn2 = model_transformer.sd3_gate_scales_attn2[model_idx][attn2_idx]
                    out_list[6] = out_list[6] * w_attn2
            
            return tuple(out_list)
        return hook
    
    def _register_hooks(self, model_idx: int):
        for idx, block in enumerate(self.transformer_blocks):
            # Hook norm1 (main branch)
            h1 = block.norm1.register_forward_hook(
                _create_universal_hook(idx, model_idx, context=False), with_kwargs=True
            )
            self._gate_hooks.append(h1)
            
            # Hook norm1_context if exists
            if hasattr(block, "norm1_context") and block.norm1_context is not None:
                h2 = block.norm1_context.register_forward_hook(
                    _create_universal_hook(idx, model_idx, context=True), with_kwargs=True
                )
                self._gate_hooks.append(h2)
    
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


class SGSD3Pipeline(BaseSGPipeline):
    def __init__(self, device, dtype, model_name="stabilityai/stable-diffusion-3-medium", num_models=1, verbose = False):
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        self.pipeline.transformer = extend_sd3_transformer_with_sg(
            self.pipeline.transformer, num_models=num_models
        )
        if not verbose:
            self.pipeline.set_progress_bar_config(disable=True)
    
    def get_coefficient_shapes(self) -> Dict[str, int]:
        """Get dimensions for SD3 coefficient vector"""
        t = self.pipeline.transformer
        n_blocks = len(t.transformer_blocks)
        n_models = len(t.models_scales)
        
        # Count blocks with context gates and dual attention
        n_context = sum(t._has_context_gates)
        n_attn2 = sum(t._has_dual_attn)
        
        main = n_models * n_blocks * 2
        context = n_models * n_context * 2
        attn2 = n_models * n_attn2
        models = n_models
        
        return {
            # must have
            "total": main + context + attn2 + models,
            "n_models": n_models,
            # optional
            "main": main,
            "context": context,
            "attn2": attn2,
            "n_blocks": n_blocks,
            "n_context": n_context,
            "n_attn2": n_attn2,
        }
    
    def flatten_coefficients(self) -> np.ndarray:
        t = self.pipeline.transformer
        s = self.get_coefficient_shapes()
        n_blocks, n_context, n_attn2, n_models = s["n_blocks"], s["n_context"], s["n_attn2"], s["n_models"]
        vec: List[float] = []
        for m in range(n_models):
            for b in range(n_blocks):
                w_attn, w_mlp = t.sd3_gate_scales_main[m][b]
                vec += [float(w_attn.detach().cpu().item()), float(w_mlp.detach().cpu().item())]
        for m in range(n_models):
            for b in range(n_context):
                w_attn, w_mlp = t.sd3_gate_scales_context[m][b]
                vec += [float(w_attn.detach().cpu().item()), float(w_mlp.detach().cpu().item())]
        for m in range(n_models):
            for b in range(n_attn2):
                vec.append(float(t.sd3_gate_scales_attn2[m][b].detach().cpu().item()))
        vec += t.models_scales.detach().cpu().to(torch.float64).numpy().tolist()
        return np.asarray(vec, dtype=np.float64)  

    def flat_to_struct(self, x: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if x is None:
            x = self.flatten_coefficients()
        s = self.get_coefficient_shapes()
        d_main, d_ctx, d_a2 = s["main"], s["context"], s["attn2"]
        n_blocks, n_context, n_attn2, n_models = s["n_blocks"], s["n_context"], s["n_attn2"], s["n_models"]
        main = x[:d_main]; ctx = x[d_main:d_main + d_ctx]; a2 = x[d_main + d_ctx:d_main + d_ctx + d_a2]; models = x[d_main + d_ctx + d_a2:]  

        scales_main: List[List[tuple]] = []
        per_m_main = n_blocks * 2
        for m in range(n_models):
            start = m * per_m_main
            mm = [(float(main[start + 2*b + 0]), float(main[start + 2*b + 1])) for b in range(n_blocks)]
            scales_main.append(mm)

        scales_context: List[List[tuple]] = []
        per_m_ctx = n_context * 2
        for m in range(n_models):
            start = m * per_m_ctx
            cc = [(float(ctx[start + 2*b + 0]), float(ctx[start + 2*b + 1])) for b in range(n_context)]
            scales_context.append(cc)

        scales_attn2: List[List[float]] = []
        for m in range(n_models):
            start = m * n_attn2
            aa = [float(a2[start + b]) for b in range(n_attn2)]
            scales_attn2.append(aa)

        models_scales = [float(v) for v in models]  

        return {
            "num_models": n_models,
            "scales_main": scales_main,
            "scales_context": scales_context,
            "scales_attn2": scales_attn2,
            "models_scales": models_scales,
        }  

    def struct_to_flat(self, sdict: Dict[str, Any]) -> np.ndarray:
        s = self.get_coefficient_shapes()
        n_blocks, n_context, n_attn2, n_models = s["n_blocks"], s["n_context"], s["n_attn2"], s["n_models"]
        sm, sc, sa2, ms = sdict["scales_main"], sdict["scales_context"], sdict["scales_attn2"], sdict["models_scales"]
        assert len(sm) == n_models and len(sc) == n_models and len(sa2) == n_models and len(ms) == n_models  
        vec: List[float] = []
        for m in range(n_models):
            assert len(sm[m]) == n_blocks and all(len(p) == 2 for p in sm[m])
            for b in range(n_blocks):
                vec += [float(sm[m][b][0]), float(sm[m][b][1])]
        for m in range(n_models):
            assert len(sc[m]) == n_context and all(len(p) == 2 for p in sc[m])
            for b in range(n_context):
                vec += [float(sc[m][b][0]), float(sc[m][b][1])]
        for m in range(n_models):
            assert len(sa2[m]) == n_attn2
            vec += [float(v) for v in sa2[m]]
        vec += [float(v) for v in ms]
        return np.asarray(vec, dtype=np.float64)  

    def apply_coefficients(self, x: np.ndarray) -> None:
        kwargs = self.flat_to_struct(x)
        self.pipeline.transformer = extend_sd3_transformer_with_sg(self.pipeline.transformer, **kwargs)
    
    def __call__(self, prompts, num_inference_steps, guidance_scale, height, width, generator, **kwargs):
        return self.pipeline(prompts, 
                             num_inference_steps=num_inference_steps,
                             guidance_scale=guidance_scale,
                             height=height,
                             width=width,
                             generator=generator,
                             **kwargs)
