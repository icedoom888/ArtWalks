# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, List, Optional, Union

from pathlib import Path
import time
import json
import torch
import numpy as np
from typing import Callable, List, Optional, Union, Tuple
import math

import PIL
from PIL import Image
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from .upsampling import RealESRGANModel
from .utils import get_timesteps_arr, make_video_pyav, slerp

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionImageVariationVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate variations from an input image using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder. Stable Diffusion Image Variation uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.image_encoder, self.vae, self.safety_checker]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        ### THIS was commented out because is done later in L400
        # if do_classifier_free_guidance:
        #     uncond_embeddings = torch.zeros_like(image_embeddings)

        #     # For classifier free guidance, we need to do two forward passes.
        #     # Here we concatenate the unconditional and text embeddings into a single batch
        #     # to avoid doing two forward passes
        #     image_embeddings = torch.cat([uncond_embeddings, image_embeddings])

        return image_embeddings

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, image, height, width, callback_steps):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor] = None,
        image_embeddings: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                The image or images to guide the image generation. If you provide a tensor, it needs to comply with the
                configuration of
                [this](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json)
                `CLIPFeatureExtractor`
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        if image is not None:
            self.check_inputs(image, height, width, callback_steps)

            # 2. Define call parameters
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(image, list):
                batch_size = len(image)
            else:
                batch_size = image.shape[0]
        else:
            batch_size, _, _ = image_embeddings.shape

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Encode input image
        if image is not None:
            image_embeddings = self._encode_image(image, device, num_images_per_prompt, do_classifier_free_guidance)

        
        if do_classifier_free_guidance:
            uncond_embeddings = torch.zeros_like(image_embeddings)
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([uncond_embeddings, image_embeddings])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=image_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, image_embeddings.dtype)
        
        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
    
    def generate_inputs(self, prompt_a, prompt_b, seed_a, seed_b, noise_shape, T, batch_size, device, num_images_per_prompt, do_classifier_free_guidance):
        prompt_a = Image.open(prompt_a).convert("RGB")
        prompt_b = Image.open(prompt_b).convert("RGB")
        embeds_a = self._encode_image(prompt_a, device, num_images_per_prompt, do_classifier_free_guidance)
        embeds_b = self._encode_image(prompt_b, device, num_images_per_prompt, do_classifier_free_guidance)
        latents_dtype = embeds_a.dtype
        latents_a = self.init_noise(seed_a, noise_shape, latents_dtype)
        latents_b = self.init_noise(seed_b, noise_shape, latents_dtype)

        batch_idx = 0
        embeds_batch, noise_batch = None, None
        for i, t in enumerate(T):
            embeds = torch.lerp(embeds_a, embeds_b, t)
            noise = slerp(float(t), latents_a, latents_b)

            embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
            noise_batch = noise if noise_batch is None else torch.cat([noise_batch, noise])
            batch_is_ready = embeds_batch.shape[0] == batch_size or i + 1 == T.shape[0]
            if not batch_is_ready:
                continue
            yield batch_idx, embeds_batch, noise_batch
            batch_idx += 1
            del embeds_batch, noise_batch
            torch.cuda.empty_cache()
            embeds_batch, noise_batch = None, None

    def make_clip_frames(
        self,
        prompt_a: str,
        prompt_b: str,
        seed_a: int,
        seed_b: int,
        num_interpolation_steps: int = 5,
        save_path: Union[str, Path] = "outputs/",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsample: bool = False,
        batch_size: int = 1,
        image_file_ext: str = ".png",
        T: np.ndarray = None,
        skip: int = 0,
        negative_prompt: str = None,
        step: Optional[Tuple[int, int]] = None,
        num_images_per_prompt: Optional[int] = 1
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        T = T if T is not None else np.linspace(0.0, 1.0, num_interpolation_steps)
        if T.shape[0] != num_interpolation_steps:
            raise ValueError(f"Unexpected T shape, got {T.shape}, expected dim 0 to be {num_interpolation_steps}")

        if upsample:
            if getattr(self, "upsampler", None) is None:
                self.upsampler = RealESRGANModel.from_pretrained("nateraw/real-esrgan")
            self.upsampler.to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0
        device = self._execution_device

        batch_generator = self.generate_inputs(
            prompt_a,
            prompt_b,
            seed_a,
            seed_b,
            (1, self.unet.in_channels, height // 8, width // 8),
            T[skip:],
            batch_size,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance
        )
        num_batches = math.ceil(num_interpolation_steps/batch_size)

        log_prefix = '' if step is None else f'[{step[0]}/{step[1]}] '

        frame_index = skip
        for batch_idx, embeds_batch, noise_batch in batch_generator:
            if batch_size == 1:
                msg = f"Generating frame {frame_index}"
            else:
                msg = f"Generating frames {frame_index}-{frame_index+embeds_batch.shape[0]-1}"
            logger.info(f'{log_prefix}[{batch_idx}/{num_batches}] {msg}')
            outputs = self(
                latents=noise_batch,
                image_embeddings=embeds_batch,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                eta=eta,
                num_inference_steps=num_inference_steps,
                output_type="pil" if not upsample else "numpy",
            )["images"]

            for image in outputs:
                frame_filepath = save_path / (f"frame%06d{image_file_ext}" % frame_index)
                image = image if not upsample else self.upsampler(image)
                image.save(frame_filepath)
                frame_index += 1

    def walk(
        self,
        prompts: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        num_interpolation_steps: Optional[Union[int, List[int]]] = 5,  # int or list of int
        output_dir: Optional[str] = "./dreams",
        name: Optional[str] = None,
        image_file_ext: Optional[str] = ".png",
        fps: Optional[int] = 30,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsample: Optional[bool] = False,
        batch_size: Optional[int] = 1,
        resume: Optional[bool] = False,
        audio_filepath: str = None,
        audio_start_sec: Optional[Union[int, float]] = None,
        margin: Optional[float] = 1.0,
        smooth: Optional[float] = 0.0,
        negative_prompt: Optional[str] = None,
        make_video: Optional[bool] = True,
    ):
        """Generate a video from a sequence of prompts and seeds. Optionally, add audio to the
        video to interpolate to the intensity of the audio.

        Args:
            prompts (Optional[List[str]], optional):
                list of text prompts. Defaults to None.
            seeds (Optional[List[int]], optional):
                list of random seeds corresponding to prompts. Defaults to None.
            num_interpolation_steps (Union[int, List[int]], *optional*):
                How many interpolation steps between each prompt. Defaults to None.
            output_dir (Optional[str], optional):
                Where to save the video. Defaults to './dreams'.
            name (Optional[str], optional):
                Name of the subdirectory of output_dir. Defaults to None.
            image_file_ext (Optional[str], *optional*, defaults to '.png'):
                The extension to use when writing video frames.
            fps (Optional[int], *optional*, defaults to 30):
                The frames per second in the resulting output videos.
            num_inference_steps (Optional[int], *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (Optional[float], *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (Optional[float], *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            height (Optional[int], *optional*, defaults to None):
                height of the images to generate.
            width (Optional[int], *optional*, defaults to None):
                width of the images to generate.
            upsample (Optional[bool], *optional*, defaults to False):
                When True, upsamples images with realesrgan.
            batch_size (Optional[int], *optional*, defaults to 1):
                Number of images to generate at once.
            resume (Optional[bool], *optional*, defaults to False):
                When True, resumes from the last frame in the output directory based
                on available prompt config. Requires you to provide the `name` argument.
            audio_filepath (str, *optional*, defaults to None):
                Optional path to an audio file to influence the interpolation rate.
            audio_start_sec (Optional[Union[int, float]], *optional*, defaults to 0):
                Global start time of the provided audio_filepath.
            margin (Optional[float], *optional*, defaults to 1.0):
                Margin from librosa hpss to use for audio interpolation.
            smooth (Optional[float], *optional*, defaults to 0.0):
                Smoothness of the audio interpolation. 1.0 means linear interpolation.
            negative_prompt (Optional[str], *optional*, defaults to None):
                Optional negative prompt to use. Same across all prompts.
            make_video (Optional[bool], *optional*, defaults to True):
                When True, makes a video from the generated frames. If False, only
                generates the frames.

        This function will create sub directories for each prompt and seed pair.

        For example, if you provide the following prompts and seeds:

        ```
        prompts = ['a dog', 'a cat', 'a bird']
        seeds = [1, 2, 3]
        num_interpolation_steps = 5
        output_dir = 'output_dir'
        name = 'name'
        fps = 5
        ```

        Then the following directories will be created:

        ```
        output_dir
        ├── name
        │   ├── name_000000
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000000.mp4
        │   ├── name_000001
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000001.mp4
        │   ├── ...
        │   ├── name.mp4
        |   |── prompt_config.json
        ```

        Returns:
            str: The resulting video filepath. This video includes all sub directories' video clips.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        output_path = Path(output_dir)

        name = name or time.strftime("%Y%m%d-%H%M%S")
        save_path_root = output_path / name
        save_path_root.mkdir(parents=True, exist_ok=True)

        # Where the final video of all the clips combined will be saved
        output_filepath = save_path_root / f"{name}.mp4"

        # If using same number of interpolation steps between, we turn into list
        if not resume and isinstance(num_interpolation_steps, int):
            num_interpolation_steps = [num_interpolation_steps] * (len(prompts) - 1)

        if not resume:
            audio_start_sec = audio_start_sec or 0

        # Save/reload prompt config
        prompt_config_path = save_path_root / "prompt_config.json"
        if not resume:
            prompt_config_path.write_text(
                json.dumps(
                    dict(
                        prompts=prompts,
                        seeds=seeds,
                        num_interpolation_steps=num_interpolation_steps,
                        fps=fps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        upsample=upsample,
                        height=height,
                        width=width,
                        audio_filepath=audio_filepath,
                        audio_start_sec=audio_start_sec,
                        negative_prompt=negative_prompt,
                    ),
                    indent=2,
                    sort_keys=False,
                )
            )
        else:
            data = json.load(open(prompt_config_path))
            prompts = data["prompts"]
            seeds = data["seeds"]
            num_interpolation_steps = data["num_interpolation_steps"]
            fps = data["fps"]
            num_inference_steps = data["num_inference_steps"]
            guidance_scale = data["guidance_scale"]
            eta = data["eta"]
            upsample = data["upsample"]
            height = data["height"]
            width = data["width"]
            audio_filepath = data["audio_filepath"]
            audio_start_sec = data["audio_start_sec"]
            negative_prompt = data.get("negative_prompt", None)

        for i, (prompt_a, prompt_b, seed_a, seed_b, num_step) in enumerate(
            zip(prompts, prompts[1:], seeds, seeds[1:], num_interpolation_steps)
        ):
            # {name}_000000 / {name}_000001 / ...
            save_path = save_path_root / f"{name}_{i:06d}"

            # Where the individual clips will be saved
            step_output_filepath = save_path / f"{name}_{i:06d}.mp4"

            # Determine if we need to resume from a previous run
            skip = 0
            if resume:
                if step_output_filepath.exists():
                    print(f"Skipping {save_path} because frames already exist")
                    continue

                existing_frames = sorted(save_path.glob(f"*{image_file_ext}"))
                if existing_frames:
                    skip = int(existing_frames[-1].stem[-6:]) + 1
                    if skip + 1 >= num_step:
                        print(f"Skipping {save_path} because frames already exist")
                        continue
                    print(f"Resuming {save_path.name} from frame {skip}")

            audio_offset = audio_start_sec + sum(num_interpolation_steps[:i]) / fps
            audio_duration = num_step / fps

            self.make_clip_frames(
                prompt_a,
                prompt_b,
                seed_a,
                seed_b,
                num_interpolation_steps=num_step,
                save_path=save_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                height=height,
                width=width,
                upsample=upsample,
                batch_size=batch_size,
                T=get_timesteps_arr(
                    audio_filepath,
                    offset=audio_offset,
                    duration=audio_duration,
                    fps=fps,
                    margin=margin,
                    smooth=smooth,
                )
                if audio_filepath
                else None,
                skip=skip,
                negative_prompt=negative_prompt,
                step=(i, len(prompts) - 1),
            )
            if make_video:
                make_video_pyav(
                    save_path,
                    audio_filepath=audio_filepath,
                    fps=fps,
                    output_filepath=step_output_filepath,
                    glob_pattern=f"*{image_file_ext}",
                    audio_offset=audio_offset,
                    audio_duration=audio_duration,
                    sr=44100,
                )
        if make_video:
            return make_video_pyav(
                save_path_root,
                audio_filepath=audio_filepath,
                fps=fps,
                audio_offset=audio_start_sec,
                audio_duration=sum(num_interpolation_steps) / fps,
                output_filepath=output_filepath,
                glob_pattern=f"**/*{image_file_ext}",
                sr=44100,
            )

    def init_noise(self, seed, noise_shape, dtype):
        """Helper to initialize noise"""
        # randn does not exist on mps, so we create noise on CPU here and move it to the device after initialization
        if self.device.type == "mps":
            noise = torch.randn(
                noise_shape,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).to(self._execution_device)
        else:
            noise = torch.randn(
                noise_shape,
                device=self.device,
                generator=torch.Generator(device=self._execution_device).manual_seed(seed),
                dtype=dtype,
            )
        return noise
