import types
from typing import Sequence, Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np

from diffusers import DiffusionPipeline
from peft import PeftModel

from .base_sg import BaseSGPipeline

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import *

@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def call___calibri(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    skip_layer_guidance_stop: float = 0.2,
    skip_layer_guidance_start: float = 0.01,
    mu: Optional[float] = None,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            will be used instead
        prompt_3 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
            will be used instead
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion
            Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
            of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
            `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
            the text `prompt`, usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used instead
        negative_prompt_3 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
            `text_encoder_3`. If not defined, `negative_prompt` is used instead
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        ip_adapter_image (`PipelineImageInput`, *optional*):
            Optional image input to work with IP Adapters.
        ip_adapter_image_embeds (`torch.Tensor`, *optional*):
            Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
            emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
            `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
            a plain tuple.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
        skip_guidance_layers (`List[int]`, *optional*):
            A list of integers that specify layers to skip during guidance. If not provided, all layers will be
            used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
            Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
        skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
            `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
            with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
            with a scale of `1`.
        skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
            `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
            `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
            StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
        skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
            `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
            `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
            StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
        mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # if self.do_classifier_free_guidance:
    #     if skip_guidance_layers is not None:
    #         original_prompt_embeds = prompt_embeds
    #         original_pooled_prompt_embeds = pooled_prompt_embeds
        # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        # pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, height, width = latents.shape
        image_seq_len = (height // self.transformer.config.patch_size) * (
            width // self.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue


            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
                uncond=False
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                latent_model_input = latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    pooled_projections=negative_pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    uncond=True
                )[0]
                noise_pred = noise_pred + noise_pred_uncond

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    if output_type == "latent":
        image = latents

    else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image)


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
        if model_transformer.cfg_scale is not None:
            models_scales[0] = model_transformer.cfg_scale
            models_scales[1] = -model_transformer.cfg_scale + 1

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

        def get_result(idx_model, result):
            scale = self.models_scales[idx_model]
            if scale.abs().item() == 0.0:
                return result
            
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
            return result

        result = None

        if self.cfg_case and "uncond" in kwargs:
            if kwargs["uncond"]:
                idx_model = 1
            else:
                idx_model = 0
            kwargs.pop("uncond")
            result = get_result(idx_model, result)
        else:
            if "uncond" in kwargs:
                kwargs.pop("uncond")
            for idx_model in range(len(self.models_scales)):
                result = get_result(idx_model, result)
        
        # Fallback if all scales were zero
        if result is None:
            result = self._original_forward(*args, **kwargs)
        
        return result
    
    # Bind methods
    model_transformer._register_hooks = types.MethodType(_register_hooks, model_transformer)
    model_transformer._remove_hooks = types.MethodType(_remove_hooks, model_transformer)
    model_transformer.forward = types.MethodType(new_forward, model_transformer)
    
    return model_transformer


class SGSD3PipelineLORA(BaseSGPipeline):
    def __init__(self, device, dtype, model_name="jieliu/SD3.5M-FlowGRPO-GenEval", num_models=1, cfg_case=True, cfg_scale=7.0, verbose = False):
        base_model_name = "stabilityai/stable-diffusion-3.5-medium"
        ### specify cfg_scale with float only if cfg_case is true
        pipeline = DiffusionPipeline.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
        )

        pipeline.transformer.cfg_case = cfg_case
        pipeline.transformer.cfg_scale = cfg_scale
        pipeline.transformer = extend_sd3_transformer_with_sg(
            pipeline.transformer, num_models=num_models
        )
        pipeline.transformer.cfg_scale = None
        if not verbose:
            pipeline.set_progress_bar_config(disable=True)
        
        pipeline.__class__.__call__ = call___calibri

        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, model_name)
        pipeline.transformer = pipeline.transformer.merge_and_unload()

        pipeline = pipeline.to(device)
        self.pipeline = pipeline
    
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
