import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from diffusers import FluxPipeline

class AlphaParameters:
    def __init__(self, num_double_blocks, num_single_blocks, device, model_name="main"):
        self.device = device
        self.model_name = model_name
        self.double_alpha_attn = nn.Parameter(torch.ones(num_double_blocks, device=device))
        self.double_alpha_ff = nn.Parameter(torch.ones(num_double_blocks, device=device))
        self.double_alpha_context_attn = nn.Parameter(torch.ones(num_double_blocks, device=device))
        self.double_alpha_context_ff = nn.Parameter(torch.ones(num_double_blocks, device=device))
        self.single_alpha = nn.Parameter(torch.ones(num_single_blocks, device=device))

    def get_double_alphas(self, block_idx: int):
        return {
            'attn': self.double_alpha_attn[block_idx],
            'ff': self.double_alpha_ff[block_idx],
            'context_attn': self.double_alpha_context_attn[block_idx],
            'context_ff': self.double_alpha_context_ff[block_idx],
        }

    def get_single_alpha(self, block_idx: int):
        return self.single_alpha[block_idx]

class AlphaModifiedFluxTransformerBlock(nn.Module):
    def __init__(self, original_block, block_idx, alpha_params: AlphaParameters):
        super().__init__()
        self.original_block = original_block
        self.block_idx = block_idx
        self.alpha_params = alpha_params
        for name, module in original_block.named_children():
            setattr(self, name, module)

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                joint_attention_kwargs=None, ip_hidden_states=None):
        alphas = self.alpha_params.get_double_alphas(self.block_idx)
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {})
        )
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + alphas['attn'] * attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + alphas['ff'] * ff_output

        if ip_hidden_states is not None:
            ip_attn_output = self.attn_ip(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=ip_hidden_states,
                **(joint_attention_kwargs or {})
            )
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + alphas['context_attn'] * context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + alphas['context_ff'] * c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

class AlphaModifiedFluxSingleTransformerBlock(nn.Module):
    def __init__(self, original_block, block_idx, alpha_params: AlphaParameters):
        super().__init__()
        self.original_block = original_block
        self.block_idx = block_idx
        self.alpha_params = alpha_params
        for name, module in original_block.named_children():
            setattr(self, name, module)

    def forward(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        alpha = self.alpha_params.get_single_alpha(self.block_idx)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {})
        )
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + alpha * hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states

class AutoGuidanceFluxPipeline:
    def __init__(self, main_model_name, guidance_model_name, torch_dtype="float16",
                 device_map="balanced", low_cpu_mem_usage=True):
        dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        self.main_pipeline = FluxPipeline.from_pretrained(
            main_model_name, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.guidance_pipeline = FluxPipeline.from_pretrained(
            guidance_model_name, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )
        main_num_double = len(self.main_pipeline.transformer.transformer_blocks)
        main_num_single = len(self.main_pipeline.transformer.single_transformer_blocks)
        guidance_num_double = len(self.guidance_pipeline.transformer.transformer_blocks)
        guidance_num_single = len(self.guidance_pipeline.transformer.single_transformer_blocks)

        self.main_alpha_params = AlphaParameters(main_num_double, main_num_single, self.main_pipeline.transformer.device, "main")
        self.guidance_alpha_params = AlphaParameters(guidance_num_double, guidance_num_single, self.guidance_pipeline.transformer.device, "guidance")
        self._modify_pipeline(self.main_pipeline, self.main_alpha_params)
        self._modify_pipeline(self.guidance_pipeline, self.guidance_alpha_params)

    def _modify_pipeline(self, pipeline, alpha_params):
        new_double = nn.ModuleList([
            AlphaModifiedFluxTransformerBlock(b, i, alpha_params)
            for i, b in enumerate(pipeline.transformer.transformer_blocks)
        ])
        pipeline.transformer.transformer_blocks = new_double
        new_single = nn.ModuleList([
            AlphaModifiedFluxSingleTransformerBlock(b, i, alpha_params)
            for i, b in enumerate(pipeline.transformer.single_transformer_blocks)
        ])
        pipeline.transformer.single_transformer_blocks = new_single

    @torch.no_grad()
    def generate_with_autoguidance(self, prompts, guidance_weight=0.3, **kwargs):
        # prompts: list[str] — батч
        main_res = self.main_pipeline(prompt=prompts, **kwargs)
        guidance_res = self.guidance_pipeline(prompt=prompts, **kwargs)

        if 0 < guidance_weight < 1:
            blended = []
            for img_main, img_guid in zip(main_res.images, guidance_res.images):
                main_arr = np.array(img_main).astype(np.float32)
                guid_arr = np.array(img_guid).astype(np.float32)
                blend = (1 - guidance_weight) * main_arr + guidance_weight * guid_arr
                blend = np.clip(blend, 0, 255).astype(np.uint8)
                blended.append(Image.fromarray(blend))
            main_res.images = blended
        return main_res
