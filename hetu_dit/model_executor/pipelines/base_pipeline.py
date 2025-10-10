from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import List
import torch
import torch.distributed
import torch.nn as nn
import time

from diffusers import DiffusionPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import hetu_dit.envs as envs
from hetu_dit.model_executor.vae_executor.modules.adapters.vae.decoder_adapters import (
    DecoderAdapter,
)
from hetu_dit.config.config import (
    EngineConfig,
    InputConfig,
)
from hetu_dit.envs import PACKAGES_CHECKER
from hetu_dit.logger import init_logger
from hetu_dit.core.distributed import (
    get_data_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_pp_group,
    get_world_group,
    get_runtime_state,
    initialize_runtime_state,
    is_dp_last_group,
    get_world_group_rank,
    get_world_group_world_size,
    get_parallel_groups,
    get_dp_last_group,
)
from hetu_dit.model_executor.base_wrapper import hetuDiTBaseWrapper


PACKAGES_CHECKER.check_diffusers_version()

from hetu_dit.model_executor.diffusion_executor.schedulers import *
from hetu_dit.model_executor.diffusion_executor.models.transformers import *
from hetu_dit.model_executor.diffusion_executor.layers.attention_processor import *
from hetu_dit.model_executor.utils.register_warpper import (
    hetuDiTTransformerWrappersRegister,
    hetuDiTSchedulerWrappersRegister,
)

try:
    import os
    from onediff.infer_compiler import compile as od_compile

    HAS_OF = True
    os.environ["NEXFORT_FUSE_TIMESTEP_EMBEDDING"] = "0"
    os.environ["NEXFORT_FX_FORCE_TRITON_SDPA"] = "1"
except:
    HAS_OF = False

logger = init_logger(__name__)


class hetuDiTPipelineBaseWrapper(hetuDiTBaseWrapper, metaclass=ABCMeta):
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        engine_config: EngineConfig,
        is_serving: bool = False,
    ):
        self.module: DiffusionPipeline
        self._init_runtime_state(
            pipeline=pipeline, engine_config=engine_config, is_serving=is_serving
        )
        # backbone
        transformer = getattr(pipeline, "transformer", None)
        unet = getattr(pipeline, "unet", None)
        # vae
        vae = getattr(pipeline, "vae", None)
        # scheduler
        scheduler = getattr(pipeline, "scheduler", None)

        if transformer is not None:
            pipeline.transformer = self._convert_transformer_backbone(
                transformer,
                enable_torch_compile=engine_config.runtime_config.use_torch_compile,
                enable_onediff=engine_config.runtime_config.use_onediff,
            )
        elif unet is not None:
            pipeline.unet = self._convert_unet_backbone(unet)

        if scheduler is not None:
            pipeline.scheduler = self._convert_scheduler(scheduler)

        if (
            vae is not None
            and engine_config.runtime_config.use_parallel_vae
            and not self.use_naive_forward()
        ):
            pipeline.vae = self._convert_vae(vae)

        super().__init__(module=pipeline)

    def reset_activation_cache(self):
        if hasattr(self.module, "transformer") and hasattr(
            self.module.transformer, "reset_activation_cache"
        ):
            self.module.transformer.reset_activation_cache()
        if hasattr(self.module, "unet") and hasattr(
            self.module.unet, "reset_activation_cache"
        ):
            self.module.unet.reset_activation_cache()
        if hasattr(self.module, "vae") and hasattr(
            self.module.vae, "reset_activation_cache"
        ):
            self.module.vae.reset_activation_cache()
        if hasattr(self.module, "scheduler") and hasattr(
            self.module.scheduler, "reset_activation_cache"
        ):
            self.module.scheduler.reset_activation_cache()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    @staticmethod
    def profile_decorator(stage_name: str):
        # stages : encode diffusion vae
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                states = get_runtime_state().worker_state
                if stage_name == "diffusion_async":
                    result = func(self, *args, **kwargs)
                    torch.cuda.synchronize()
                    states["diffusion_end_time"] = time.perf_counter()
                    states["diffusion_mem"] = torch.cuda.max_memory_allocated()
                    elapsed_time = time.perf_counter() - states["diffusion_start_time"]
                    states["diffusion"] = elapsed_time
                else:
                    states["stage_name"] = stage_name
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    states[f"{stage_name}_start_time"] = start_time
                    result = func(self, *args, **kwargs)
                    torch.cuda.synchronize()
                    states[f"{stage_name}_end_time"] = time.perf_counter()
                    states[f"{stage_name}_mem"] = torch.cuda.max_memory_allocated()
                    elapsed_time = time.perf_counter() - start_time
                    states[f"{stage_name}"] = elapsed_time
                return result

            return wrapper

        return decorator

    @staticmethod
    def enable_data_parallel(func):
        @wraps(func)
        def data_parallel_fn(self, *args, **kwargs):
            prompt = kwargs.get("prompt", None)
            negative_prompt = kwargs.get("negative_prompt", "")
            # dp_degree <= batch_size
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            if batch_size > 1:
                dp_degree = get_runtime_state().parallel_config.dp_degree
                dp_group_rank = get_world_group_rank() // (
                    get_world_group_world_size() // get_data_parallel_world_size()
                )
                dp_group_batch_size = (batch_size + dp_degree - 1) // dp_degree
                start_batch_idx = dp_group_rank * dp_group_batch_size
                end_batch_idx = min(
                    (dp_group_rank + 1) * dp_group_batch_size, batch_size
                )
                prompt = prompt[start_batch_idx:end_batch_idx]
                if isinstance(negative_prompt, List):
                    negative_prompt = negative_prompt[start_batch_idx:end_batch_idx]
                kwargs["prompt"] = prompt
                kwargs["negative_prompt"] = negative_prompt
            return func(self, *args, **kwargs)

        return data_parallel_fn

    def use_naive_forward(self):
        return False

    @staticmethod
    def check_to_use_naive_forward(func):
        @wraps(func)
        def check_naive_forward_fn(self, *args, **kwargs):
            if self.use_naive_forward():
                return self.module(*args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return check_naive_forward_fn

    @staticmethod
    def check_model_parallel_state(
        cfg_parallel_available: bool = True,
        sequence_parallel_available: bool = True,
        pipefusion_parallel_available: bool = True,
    ):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if (
                    not cfg_parallel_available
                    and get_runtime_state().parallel_config.cfg_degree > 1
                ):
                    raise RuntimeError("CFG parallelism is not supported by the model")
                if (
                    not sequence_parallel_available
                    and get_runtime_state().parallel_config.sp_degree > 1
                ):
                    raise RuntimeError(
                        "Sequence parallelism is not supported by the model"
                    )
                if (
                    not pipefusion_parallel_available
                    and get_runtime_state().parallel_config.pp_degree > 1
                ):
                    raise RuntimeError(
                        "Pipefusion parallelism is not supported by the model"
                    )
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def forward(self):
        pass

    def prepare_run(
        self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1
    ):
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            use_resolution_binning=input_config.use_resolution_binning,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(42),
            output_type=input_config.output_type,
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    def _init_runtime_state(
        self,
        pipeline: DiffusionPipeline,
        engine_config: EngineConfig,
        is_serving: bool = False,
    ):
        initialize_runtime_state(
            pipeline=pipeline, engine_config=engine_config, is_serving=is_serving
        )

    def _convert_transformer_backbone(
        self, transformer: nn.Module, enable_torch_compile: bool, enable_onediff: bool
    ):
        logger.info("Transformer backbone found, paralleling transformer...")
        wrapper = hetuDiTTransformerWrappersRegister.get_wrapper(transformer)
        transformer = wrapper(transformer)

        if enable_torch_compile and enable_onediff:
            logger.warning(
                "apply --use_torch_compile and --use_onediff togather. we use torch compile only"
            )

        if enable_torch_compile or enable_onediff:
            if getattr(transformer, "forward") is not None:
                if enable_torch_compile:
                    optimized_transformer_forward = torch.compile(
                        getattr(transformer, "forward")
                    )
                elif enable_onediff:
                    # O3: +fp16 reduction
                    if not HAS_OF:
                        raise RuntimeError(
                            "install onediff and nexfort to --use_onediff"
                        )
                    options = {"mode": "O3"}  # mode can be O2 or O3
                    optimized_transformer_forward = od_compile(
                        getattr(transformer, "forward"),
                        backend="nexfort",
                        options=options,
                    )
                setattr(transformer, "forward", optimized_transformer_forward)
            else:
                raise AttributeError(
                    f"Transformer backbone type: {transformer.__class__.__name__} has no attribute 'forward'"
                )
        return transformer

    def _convert_unet_backbone(
        self,
        unet: nn.Module,
    ):
        logger.info("UNet Backbone found")
        raise NotImplementedError("UNet parallelisation is not supported yet")

    def _convert_scheduler(
        self,
        scheduler: nn.Module,
    ):
        logger.info("Scheduler found, paralleling scheduler...")
        wrapper = hetuDiTSchedulerWrappersRegister.get_wrapper(scheduler)
        scheduler = wrapper(scheduler)
        return scheduler

    def _convert_vae(
        self,
        vae: AutoencoderKL,
    ):
        logger.info("VAE found, paralleling vae...")
        vae.decoder = DecoderAdapter(vae.decoder)
        return vae

    @abstractmethod
    def __call__(self):
        pass

    def _init_sync_pipeline(self, latents: torch.Tensor):
        get_runtime_state().set_patched_mode(patch_mode=False)

        latents_list = [
            latents[:, :, start_idx:end_idx, :]
            for start_idx, end_idx in get_runtime_state().pp_patches_start_end_idx_global
        ]
        latents = torch.cat(latents_list, dim=-2)
        return latents

    def _init_video_sync_pipeline(self, latents: torch.Tensor):
        get_runtime_state().set_patched_mode(patch_mode=False)
        latents_list = [
            latents[:, :, :, start_idx:end_idx, :]
            for start_idx, end_idx in get_runtime_state().pp_patches_start_end_idx_global
        ]
        latents = torch.cat(latents_list, dim=-2)
        return latents

    def _init_async_pipeline(
        self,
        num_timesteps: int,
        latents: torch.Tensor,
        num_pipeline_warmup_steps: int,
    ):
        get_runtime_state().set_patched_mode(patch_mode=True)

        if is_pipeline_first_stage():
            # get latents computed in warmup stage
            # ignore latents after the last timestep
            latents = (
                get_pp_group().pipeline_recv()
                if num_pipeline_warmup_steps > 0
                else latents
            )
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_height, dim=2)
            )
        elif is_pipeline_last_stage():
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_height, dim=2)
            )
        else:
            patch_latents = [
                None for _ in range(get_runtime_state().num_pipeline_patch)
            ]

        recv_timesteps = (
            num_timesteps - 1 if is_pipeline_first_stage() else num_timesteps
        )
        for _ in range(recv_timesteps):
            for patch_idx in range(get_runtime_state().num_pipeline_patch):
                get_pp_group().add_pipeline_recv_task(patch_idx)

        return patch_latents

    def _process_cfg_split_batch(
        self,
        negative_embeds: torch.Tensor,
        embeds: torch.Tensor,
        negative_embdes_mask: torch.Tensor = None,
        embeds_mask: torch.Tensor = None,
    ):
        if get_classifier_free_guidance_world_size() == 1:
            logger.info("cfg parallelism is not enabled, skip cfg split batch")
            embeds = torch.cat([negative_embeds, embeds], dim=0)
        elif get_classifier_free_guidance_rank() == 0:
            embeds = negative_embeds
        elif get_classifier_free_guidance_rank() == 1:
            embeds = embeds
        else:
            raise ValueError("Invalid classifier free guidance rank")

        if negative_embdes_mask is None:
            return embeds

        if get_classifier_free_guidance_world_size() == 1:
            embeds_mask = torch.cat([negative_embdes_mask, embeds_mask], dim=0)
        elif get_classifier_free_guidance_rank() == 0:
            embeds_mask = negative_embdes_mask
        elif get_classifier_free_guidance_rank() == 1:
            embeds_mask = embeds_mask
        else:
            raise ValueError("Invalid classifier free guidance rank")
        return embeds, embeds_mask

    def is_dp_last_group(self):
        """Return True if in the last data parallel group, False otherwise.
        Also include parallel vae situation.
        """
        if (
            get_runtime_state().runtime_config.use_parallel_vae
            and not self.use_naive_forward()
        ):
            return get_world_group().rank_in_group == 0
        else:
            return is_dp_last_group()

    def gather_broadcast_latents(self, latents: torch.Tensor):
        """gather latents from dp last group and broacast final latents"""

        # ---------gather latents from dp last group-----------
        rank = get_world_group().rank
        device = f"cuda:{0}"
        logger.debug(
            f"ranks = {get_world_group().ranks}rank:{rank}, get_world_group().world_size:{get_world_group().world_size}"
        )
        # all gather dp last group rank list
        dp_rank_list = [
            torch.zeros(1, dtype=int, device=device)
            for _ in range(get_world_group().world_size)
        ]
        if is_dp_last_group():
            gather_rank = int(rank)
        else:
            gather_rank = -1
        torch.distributed.all_gather(
            dp_rank_list,
            torch.tensor([gather_rank], dtype=int, device=device),
            group=get_world_group().device_group,
        )
        logger.debug(f"before filter dp_rank_list:{dp_rank_list}")
        dp_rank_list = [
            int(dp_rank[0]) for dp_rank in dp_rank_list if int(dp_rank[0]) != -1
        ]
        logger.debug(f"after filter dp_rank_list:{dp_rank_list}")
        # dp_last_group = torch.distributed.new_group(dp_rank_list)
        get_parallel_groups().set_activate_dp_last_group(
            len(dp_rank_list), dp_rank_list, [dp_rank_list]
        )
        dp_last_group = get_dp_last_group()
        # gather latents from dp last group
        if rank == dp_rank_list[-1]:
            latents_list = [torch.zeros_like(latents) for _ in dp_rank_list]
        else:
            latents_list = None
        if rank in dp_rank_list:
            logger.debug(
                f" dist.rank = {torch.distributed.get_rank()}, dp_rank_list = {dp_rank_list} ranks = {get_world_group().ranks} rank:{rank}, ranks in group:{get_world_group_rank()}"
            )
            torch.distributed.gather(
                latents,
                latents_list,
                dst=dp_rank_list[-1],
                group=dp_last_group.device_group,
            )

        if rank == dp_rank_list[-1]:
            latents = torch.cat(latents_list, dim=0)

        # ------broadcast latents to all nodes---------
        src = dp_rank_list[-1]
        latents_shape_len = torch.zeros(1, dtype=torch.int, device=device)
        logger.debug(
            f"ranks = {get_world_group().ranks} rank:{rank}, src:{src}, ranks in group:{get_world_group_rank()}, get_world_group().ranks.index(src) = {get_world_group().ranks.index(src)}"
        )
        # broadcast latents shape len
        if rank == src:
            latents_shape_len[0] = len(latents.shape)
        get_world_group().broadcast(
            latents_shape_len, src=get_world_group().ranks.index(src)
        )

        # broadcast latents shape
        if rank == src:
            input_shape = torch.tensor(latents.shape, dtype=torch.int, device=device)
        else:
            input_shape = torch.zeros(
                latents_shape_len[0], dtype=torch.int, device=device
            )
        get_world_group().broadcast(input_shape, src=get_world_group().ranks.index(src))

        # broadcast latents
        if rank != src:
            dtype = get_runtime_state().runtime_config.dtype
            latents = torch.zeros(torch.Size(input_shape), dtype=dtype, device=device)
        get_world_group().broadcast(latents, src=get_world_group().ranks.index(src))

        return latents
