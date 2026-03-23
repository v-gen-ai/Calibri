import types
from typing import Sequence, Dict, List, Optional, Any, Union, Callable
import torch
import torch.nn as nn
import numpy as np
from diffusers import DiffusionPipeline


import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.utils import replace_example_docstring

from .base_sg import BaseSGPipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import QwenImagePipeline

        >>> pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=50).images[0]
        >>> image.save("qwenimage.png")
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def call_qwen_sg(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: Optional[float] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
            not greater than `1`).
        true_cfg_scale (`float`, *optional*, defaults to 1.0):
            Guidance scale as defined in [Classifier-Free Diffusion
            Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2.
            of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by
            setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to
            generate images that are closely linked to the text `prompt`, usually at the expense of lower image
            quality.
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
        guidance_scale (`float`, *optional*, defaults to None):
            A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
            where the guidance scale is applied during inference through noise prediction rescaling, guidance
            distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
            scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
            that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
            parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
            ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
            please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
            enable classifier-free guidance computations).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will be generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
        attention_kwargs (`dict`, *optional*):
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
        max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

    Examples:

    Returns:
        [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
        [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is a list with the generated images.
    """

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    if true_cfg_scale > 1.0 and negative_prompt is None and negative_prompt_embeds is None:
        if isinstance(prompt, str):
            negative_prompt = ""
        elif isinstance(prompt, list):
            negative_prompt = [""] * len(prompt)

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )

    if true_cfg_scale > 1 and not has_neg_prompt:
        logger.warning(
            f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
        )
    elif true_cfg_scale <= 1 and has_neg_prompt:
        logger.warning(
            " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
        )

    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
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
    img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
        logger.warning(
            f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
        )
        guidance = None
    elif not self.transformer.config.guidance_embeds and guidance_scale is None:
        guidance = None

    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
    )

    # 6. Denoising loop
    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                    uncond=False 
                )[0]

            if do_true_cfg:
                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=negative_txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                        uncond=True
                    )[0]
                # comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                comb_pred = noise_pred + neg_noise_pred

                scale_cond = self.transformer.models_scales[0]

                if abs(scale_cond) > 1e-5:
                    raw_noise_pred = noise_pred / scale_cond
                else:
                    raw_noise_pred = noise_pred

                cond_norm = torch.norm(raw_noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

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

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    self._current_timestep = None
    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return QwenImagePipelineOutput(images=image)


def extend_qwen_transformer_with_sg(
    model_transformer,
    num_models: int = 1,
    scales_blocks: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    models_scales: Optional[Sequence[float]] = None,
):
    n_blocks = len(model_transformer.transformer_blocks)
    
    # Defaults
    if scales_blocks is None:
        scales_blocks = [[[1.0, 1.0, 1.0, 1.0] for _ in range(n_blocks)] for _ in range(num_models)]
    if models_scales is None:
        models_scales = [0.0 for _ in range(num_models)]
        if num_models > 0:
            models_scales[0] = 1.0
        
        # Initialize with CFG logic if configured
        if getattr(model_transformer, "cfg_case", False) and getattr(model_transformer, "cfg_scale", None) is not None:
             if num_models >= 2:

                 cfg = model_transformer.cfg_scale
                 models_scales[0] = cfg          # Weight for Cond
                 models_scales[1] = 1.0 - cfg    # Weight for Uncond
    
    # Validations
    assert len(scales_blocks) == num_models
    assert len(models_scales) == num_models
    assert all(len(v) == n_blocks for v in scales_blocks)
    
    device = next(model_transformer.parameters()).device
    dtype = next(model_transformer.parameters()).dtype
    
    for p in model_transformer.parameters():
        p.requires_grad = False
    
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
                        for j in range(4) 
                    ]
                )
                for i in range(n_blocks)
            ]
        )
        model_transformer.qwen_gate_scales.append(plist)
    
    initial_scales = torch.tensor(models_scales, device=device, dtype=dtype)
    model_transformer.models_scales = nn.Parameter(initial_scales)
    
    model_transformer._gate_hooks = []
    
    def _create_qwen_hook(block_idx: int, model_idx: int, gate_type: str):
        def hook(module, args, kwargs, output):
            if not isinstance(output, (tuple, list)):
                return output
            out_list = list(output)
            scales = model_transformer.qwen_gate_scales[model_idx][block_idx]
            
            if gate_type == "attn":
                if len(out_list) >= 2:
                    out_list[0] = out_list[0] * scales[0]
                    out_list[1] = out_list[1] * scales[1]
            elif gate_type == "img_mlp":
                out_list[0] = out_list[0] * scales[2]
            elif gate_type == "txt_mlp":
                out_list[0] = out_list[0] * scales[3]
            
            return tuple(out_list)
        return hook

    def _register_hooks(self, model_idx: int):
        for idx, block in enumerate(self.transformer_blocks):
            if hasattr(block, 'attn'):
                self._gate_hooks.append(block.attn.register_forward_hook(
                    _create_qwen_hook(idx, model_idx, "attn"), with_kwargs=True
                ))
            if hasattr(block, 'img_mlp'):
                self._gate_hooks.append(block.img_mlp.register_forward_hook(
                    _create_qwen_hook(idx, model_idx, "img_mlp"), with_kwargs=True
                ))
            if hasattr(block, 'txt_mlp'):
                self._gate_hooks.append(block.txt_mlp.register_forward_hook(
                    _create_qwen_hook(idx, model_idx, "txt_mlp"), with_kwargs=True
                ))

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
        
        # Check for CFG case (cond/uncond switch)
        is_cfg_case = getattr(self, "cfg_case", False)
        
        if is_cfg_case and "uncond" in kwargs:
            if kwargs["uncond"]:
                idx_model = 1
            else:
                idx_model = 0
            kwargs.pop("uncond")
            result = get_result(idx_model, result)
        else:
            # Standard ensemble blending
            if "uncond" in kwargs:
                kwargs.pop("uncond")
            for idx_model in range(len(self.models_scales)):
                result = get_result(idx_model, result)
        
        if result is None:
            result = self._original_forward(*args, **kwargs)
        
        return result

    model_transformer._register_hooks = types.MethodType(_register_hooks, model_transformer)
    model_transformer._remove_hooks = types.MethodType(_remove_hooks, model_transformer)
    model_transformer.forward = types.MethodType(new_forward, model_transformer)
    
    return model_transformer


class SGQwenPipelineCFG(BaseSGPipeline):
    def __init__(self, device, dtype, model_name="Qwen/Qwen-Image", num_models=2, cfg_case=True, cfg_scale=7.0, verbose=False):
        pipeline = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        
        pipeline.transformer.cfg_case = cfg_case
        pipeline.transformer.cfg_scale = cfg_scale
        
        pipeline.transformer = extend_qwen_transformer_with_sg(
            pipeline.transformer, num_models=num_models
        )
        pipeline.transformer.cfg_scale = None
        
        if not verbose:
            pipeline.set_progress_bar_config(disable=True)

        pipeline.__class__.__call__ = call_qwen_sg
        self.pipeline = pipeline

    def get_coefficient_shapes(self) -> Dict[str, int]:
        t = self.pipeline.transformer
        n_blocks = len(t.transformer_blocks)
        n_models = len(t.models_scales)
        
        blocks_total = n_models * n_blocks * 4
        
        return {
            "total": blocks_total + n_models,
            "n_models": n_models,
            "n_blocks": n_blocks
        }

    def flatten_coefficients(self) -> np.ndarray:
        t = self.pipeline.transformer
        s = self.get_coefficient_shapes()
        vec = []
        for m in range(s["n_models"]):
            for b in range(s["n_blocks"]):
                scales = t.qwen_gate_scales[m][b]
                vec += [float(p.detach().cpu().item()) for p in scales]
        vec += t.models_scales.detach().cpu().to(torch.float64).numpy().tolist()
        return np.asarray(vec, dtype=np.float64)

    def flat_to_struct(self, x: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if x is None:
            x = self.flatten_coefficients()
        s = self.get_coefficient_shapes()
        n_blocks, n_models = s["n_blocks"], s["n_models"]
        
        n_params = n_models * n_blocks * 4
        params_vec = x[:n_params]
        models_vec = x[n_params:]
        
        scales_blocks = []
        curr = 0
        for m in range(n_models):
            m_blocks = []
            for b in range(n_blocks):
                blk = [float(params_vec[curr + i]) for i in range(4)]
                m_blocks.append(blk)
                curr += 4
            scales_blocks.append(m_blocks)
            
        return {
            "num_models": n_models,
            "scales_blocks": scales_blocks,
            "models_scales": [float(v) for v in models_vec],
        }

    def struct_to_flat(self, sdict: Dict[str, Any]) -> np.ndarray:
        s = self.get_coefficient_shapes()
        n_blocks, n_models = s["n_blocks"], s["n_models"]
        
        sb = sdict["scales_blocks"]
        ms = sdict["models_scales"]
        
        assert len(sb) == n_models
        assert len(ms) == n_models
        
        vec = []
        for m in range(n_models):
            assert len(sb[m]) == n_blocks
            for b in range(n_blocks):
                blk = sb[m][b]
                vec.extend([float(v) for v in blk])
        
        vec.extend([float(v) for v in ms])
        return np.asarray(vec, dtype=np.float64)

    def apply_coefficients(self, x: np.ndarray) -> None:
        kwargs = self.flat_to_struct(x)
        self.pipeline.transformer = extend_qwen_transformer_with_sg(self.pipeline.transformer, **kwargs)

    def __call__(self, prompts, num_inference_steps, guidance_scale, height, width, generator, **kwargs):
        return self.pipeline(prompts, 
                             num_inference_steps=num_inference_steps,
                             true_cfg_scale=guidance_scale,
                             height=height,
                             width=width,
                             generator=generator,
                             **kwargs)
