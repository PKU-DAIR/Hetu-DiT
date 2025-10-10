from .model_runner import ModelRunner
from datetime import timedelta
from hetu_dit.config.config import EngineConfig, InputConfig
from hetu_dit.logger import init_logger

logger = init_logger(__name__)
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, List, Tuple
from ray.actor import ActorHandle
import pickle
import os
import time
import asyncio
import hashlib
from hetu_dit.core.resource_manager.singleton_model_manager import (
    get_singleton_model_manager,
    set_singleton_model_manager,
)
from hetu_dit.core.resource_manager.cache_manager import reset_cache_manager
from hetu_dit.core.distributed.parallel_state import *
from hetu_dit import (
    hetuDiTStableDiffusion3Pipeline,
    hetuDiTCogVideoXPipeline,
    hetuDiTFluxPipeline,
    hetuDiTHunyuanDiTPipeline,
    hetuDiTHunyuanVideoPipeline,
)
from hetu_dit.core.distributed.parallel_state import get_parallel_groups
from hetu_dit.core.distributed.runtime_state import (
    reset_runtime_state,
    get_runtime_state,
)
from hetu_dit.utils import get_gpu_metadata, adjust_pipeline, adjust_text_encoder

from hetu_dit.model_executor.utils.register_warpper import (
    hetuDiTLayerWrappersRegister,
    hetuDiTAttentionProcessorRegister,
)

from hetu_dit.profiler import global_profiler

from hetu_dit.core.distributed.nixl_manager import NixlP2PManager

from hetu_dit.model_executor.diffusion_executor.layers.attention_processor import (
    hetuDiTAttentionWrapper,
    hetuDiTAttnProcessor2_0,
    hetuDiTJointAttnProcessor2_0,
    hetuDiTFluxAttnProcessor2_0,
    hetuDiTCogVideoXAttnProcessor2_0,
    hetuDiTHunyuanAttnProcessor2_0,
    hetuDiTHunyuanVideoAttnProcessor2_0,
)
from hetu_dit.model_executor.cache import TransformerBlockCache, TextEncoderBlockCache
from hetu_dit.model_executor.cache.base_cache import NeededPiece
from diffusers import StableDiffusion3Pipeline
from diffusers import CogVideoXPipeline
from diffusers import FluxPipeline
from diffusers import HunyuanDiTPipeline
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
from hetu_dit.core.distributed import (
    is_dp_last_group,
)


def _prepare_cuda_device(rank: int, torch_dtype: torch.dtype) -> torch.device:
    """Configure CUDA environment variables and select the current device."""
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _check_if_gpu_supports_dtype(torch_dtype)
    return device


def _generate_mapped_groups(
    rank_generator: "RankGenerator", group_type: str, rank_map: Dict[int, int]
) -> Tuple[List[List[int]], List[List[int]]]:
    """Return both base groups and mapped rank groups for the given type."""
    base_groups = rank_generator.get_ranks(group_type)
    mapped_groups = [[rank_map[r] for r in group] for group in base_groups]
    return base_groups, mapped_groups


def _activate_group(setter, ranks: List[int], mapped_groups: List[List[int]]) -> None:
    """Invoke the provided setter to activate a parallel group."""
    setter(
        degree=len(ranks),
        ranks=ranks,
        group_ranks=mapped_groups,
    )


def _load_serving_pipeline(engine_config: EngineConfig, model_class):
    """Instantiate the serving pipeline for the given model class."""
    if model_class == StableDiffusion3Pipeline:
        return hetuDiTStableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
            is_serving=True,
        )
    if model_class == CogVideoXPipeline:
        engine_config.runtime_config.dtype = torch.bfloat16
        return hetuDiTCogVideoXPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.bfloat16,
            is_serving=True,
        )
    if model_class == FluxPipeline:
        engine_config.runtime_config.dtype = torch.bfloat16
        return hetuDiTFluxPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.bfloat16,
            is_serving=True,
        )
    if model_class == HunyuanDiTPipeline:
        text_encoder_2 = T5EncoderModel.from_pretrained(
            engine_config.model_config.model,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
        )
        return hetuDiTHunyuanDiTPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
            text_encoder_2=text_encoder_2,
            is_serving=True,
        )
    if model_class == HunyuanVideoPipeline:
        engine_config.runtime_config.dtype = torch.bfloat16
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            revision="refs/pr/18",
        )
        return hetuDiTHunyuanVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            transformer=transformer,
            torch_dtype=torch.float16,
            revision="refs/pr/18",
            is_serving=True,
        )
    raise ValueError(f"Unsupported model class: {model_class}")


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the meta data and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        work_dir=None,
        all_worker_handles: Optional[Dict[int, "ActorHandle"]] = None,
    ) -> None:
        self.local_rank = local_rank
        self.engine_config = engine_config
        self.rank = rank
        self.dist_init_method = distributed_init_method
        self.model_runner = ModelRunner()
        self.model_parameter_metaData = None  # TODO: add model parameter metadata
        self.states = {}
        self.work_dir = (work_dir,)
        self.all_worker_handles = all_worker_handles or {}

    def init_static_env(
        self, world_size: int, ranks: List[int] = [], engine_config: EngineConfig = None
    ) -> None:
        """
        Only called in the first static placement, usually used to init the global distributed env, but not the small comm group
        """
        engine_config = engine_config or self.engine_config
        logger.debug(
            "pid=%s, worker rank=%s, local_rank=%s",
            os.getpid(),
            self.rank,
            self.local_rank,
        )
        self.device = _prepare_cuda_device(
            self.rank, engine_config.runtime_config.dtype
        )
        logger.debug("CUDA device prepared for pid=%s", os.getpid())
        logger.debug("ranks=%s", ranks)

        ################################################################################
        #         set activate world_group
        ################################################################################
        get_parallel_groups().set_activate_world_group(len(ranks), ranks, [ranks])
        logger.debug(get_parallel_groups().activate_world_groups.world_size)
        # dist.barrier()

        ################################################################################
        #         set activate other group
        ################################################################################
        parallel_config = engine_config.parallel_config

        assert len(ranks) == (
            parallel_config.tp_degree
            * parallel_config.sp_degree
            * parallel_config.pp_degree
            * parallel_config.cfg_degree
            * parallel_config.dp_degree
        ), "Product of parallel degrees must equal length of ranks"

        # TODO: The rank generator method can be changed later, parallel arrangement does not have to be sequential
        # create rank map: 0->ranks[0], 1->ranks[1], ...
        rank_map = {i: rank for i, rank in enumerate(sorted(ranks))}

        # Use RankGenerator to generate basic rank combinations
        rank_generator = RankGenerator(
            tp=parallel_config.tp_degree,
            sp=parallel_config.sp_degree,
            pp=parallel_config.pp_degree,
            cfg=parallel_config.cfg_degree,
            dp=parallel_config.dp_degree,
            order="tp-sp-pp-cfg-dp",
        )

        base_tp_groups, mapped_tp_groups = _generate_mapped_groups(
            rank_generator, "tp", rank_map
        )
        _activate_group(
            get_parallel_groups().set_activate_tp_group, ranks, mapped_tp_groups
        )
        logger.debug(
            "tensor_parallel: rank=%s, tp_group=%s, ranks=%s, group_ranks=%s",
            torch.distributed.get_rank(),
            get_tp_group().rank_in_group,
            ranks,
            mapped_tp_groups,
        )
        logger.debug(
            "get_parallel_groups().activate_tp_groups.world_size=%s",
            get_parallel_groups().activate_tp_groups.world_size,
        )

        # get pipeline parallel group
        base_pp_groups, mapped_pp_groups = _generate_mapped_groups(
            rank_generator, "pp", rank_map
        )
        _activate_group(
            get_parallel_groups().set_activate_pp_group, ranks, mapped_pp_groups
        )
        logger.debug(
            "pipeline_parallel: rank=%s, pp_rank=%s, ranks=%s, group_ranks=%s",
            torch.distributed.get_rank(),
            get_pp_group().rank_in_group,
            ranks,
            mapped_pp_groups,
        )

        # get cfg group
        _, mapped_cfg_groups = _generate_mapped_groups(rank_generator, "cfg", rank_map)
        _activate_group(
            get_parallel_groups().set_activate_cfg_group, ranks, mapped_cfg_groups
        )

        # get data parallel group
        _, mapped_dp_groups = _generate_mapped_groups(rank_generator, "dp", rank_map)
        _activate_group(
            get_parallel_groups().set_activate_dp_group, ranks, mapped_dp_groups
        )

        # get sequence parallel group

        base_sp_groups, mapped_sp_groups = _generate_mapped_groups(
            rank_generator, "sp", rank_map
        )
        get_parallel_groups().set_activate_sp_group(
            degree=len(ranks),
            ranks=ranks,
            group_ranks=mapped_sp_groups,
            ulysses_degree=parallel_config.ulysses_degree,
            ring_degree=parallel_config.ring_degree,
        )
        logger.debug(
            "sp_parallel: rank=%s, sp_group=%s, ulysses_rank=%s, ulysses_world=%s, ulysses_group=%s, ring_world=%s, ring_rank=%s, ring_group=%s, base_sp_groups=%s",
            torch.distributed.get_rank(),
            get_sp_group().rank_in_group,
            get_sp_group().ulysses_rank,
            get_sp_group().ulysses_world_size,
            get_sp_group().ulysses_group,
            get_sp_group().ring_world_size,
            get_sp_group().ring_rank,
            get_sp_group().ring_group,
            base_sp_groups,
        )

        ################################################################################
        #         set activate text_encoder group
        ################################################################################
        # Use RankGenerator to generate basic rank combinations
        if engine_config.runtime_config.use_parallel_text_encoder:
            text_encoder_rank_generator = RankGenerator(
                tp=parallel_config.text_encoder_tp_degree,
                sp=1,
                pp=1,
                cfg=1,
                dp=len(ranks) // parallel_config.text_encoder_tp_degree,
                order="tp-sp-pp-cfg-dp",
            )

            _, mapped_text_encoder_tp_groups = _generate_mapped_groups(
                text_encoder_rank_generator,
                "tp",
                rank_map,
            )
            _activate_group(
                get_parallel_groups().set_activate_text_encoder_tp_group,
                ranks,
                mapped_text_encoder_tp_groups,
            )

        logger.debug(f"finished init_static_env, pid = {os.getpid()}")
        torch.cuda.empty_cache()

    async def _initialize_and_warmup_cache(
        self,
        gpu_module: nn.Module,
        cpu_module: nn.Module,
        cache_class: type,
        initial_gpu_blocks: nn.ModuleList,
        component_name: str,
    ):
        """
        Initializes, registers, and warms up a block cache.
        """
        logger.info(f"Initializing {cache_class.__name__} for {component_name}...")

        # 1. Calculate the initial PP and TP ranges
        initial_pp_rank = get_pipeline_parallel_rank()
        initial_pp_world_size = get_pipeline_parallel_world_size()
        pp_start_prop = initial_pp_rank / initial_pp_world_size
        pp_end_prop = (initial_pp_rank + 1) / initial_pp_world_size
        initial_pp_range = (pp_start_prop, pp_end_prop)

        initial_tp_rank = get_tensor_model_parallel_rank()
        initial_tp_world_size = get_tensor_model_parallel_world_size()
        # Initially, the TP range of each block is the same as the range of its TP process
        initial_tp_range = (
            initial_tp_rank / initial_tp_world_size,
            (initial_tp_rank + 1) / initial_tp_world_size,
        )
        initial_tp_width = initial_tp_range[1] - initial_tp_range[0]

        # 2. Calculate the initial cache limit based on weights
        # initial limit = number of initial blocks * initial weight of each block
        initial_limit_weight = len(initial_gpu_blocks) * initial_tp_width

        cache_instance = cache_class(
            global_blocks_limit=initial_limit_weight,
            all_worker_handles=self.all_worker_handles,
            worker_instance=self,
        )

        # 3. Associate the cache
        gpu_module.cache = cache_instance
        logger.info(
            f"{cache_class.__name__} instance created and assigned to {component_name} with weight limit {initial_limit_weight:.2f}."
        )

        # 4. Register initial blocks
        total_cpu_blocks = cache_instance.get_total_blocks(cpu_module)
        absolute_start_index, _ = cache_instance.range_to_indices(
            initial_pp_range, total_cpu_blocks
        )
        cache_instance.register_initial_blocks(initial_gpu_blocks, absolute_start_index)

        # 5. Warm up
        logger.info(f"Warming up cache for {component_name}...")
        try:
            active_indices = cache_instance.active_block_indices
            if active_indices:
                logger.debug(
                    f"Warm-up: processing {len(active_indices)} active blocks for {component_name}."
                )

                import inspect

                if inspect.iscoroutinefunction(cache_instance.get_block):
                    tasks = [
                        cache_instance.get_block(idx, cpu_module, initial_tp_range)
                        for idx in active_indices
                    ]
                    await asyncio.gather(*tasks)
                else:
                    for idx in active_indices:
                        cache_instance.get_block(idx, cpu_module, initial_tp_range)

                logger.debug(f"Cache warm-up for {component_name} finished.")
            else:
                logger.info(f"No active blocks to warm up for {component_name}.")
        except Exception as e:
            logger.error(
                f"Cache warm-up for {component_name} failed: {e}", exc_info=True
            )

    async def init_instance_model(
        self, engine_config: EngineConfig = None, model_class=StableDiffusion3Pipeline
    ) -> None:
        """
        Used to init the small comm group in a assigned comm group
        """
        engine_config = engine_config or self.engine_config
        self.model = _load_serving_pipeline(engine_config, model_class)

        # move the model to GPU
        self.model = self.model.to("cuda")
        logger.info("Model moved to GPU.")

        if self.engine_config.runtime_config.adjust_strategy in ["cache", "p2p"]:
            # 1. Setup cache for the main Transformer
            transformer_wrapper = self.model.transformer
            cpu_transformer = get_singleton_model_manager().transformer

            # Determine initial blocks for the transformer based on its structure
            if hasattr(
                transformer_wrapper, "single_transformer_blocks"
            ):  # Flux, HunyuanVideo
                initial_transformer_blocks = nn.ModuleList(
                    transformer_wrapper.transformer_blocks
                    + transformer_wrapper.single_transformer_blocks
                )
            elif hasattr(transformer_wrapper, "blocks"):  # HunyuanDiT
                initial_transformer_blocks = transformer_wrapper.blocks
            else:
                initial_transformer_blocks = transformer_wrapper.transformer_blocks

            await self._initialize_and_warmup_cache(
                gpu_module=transformer_wrapper,
                cpu_module=cpu_transformer,
                cache_class=TransformerBlockCache,
                initial_gpu_blocks=initial_transformer_blocks,
                component_name="Transformer",
            )

            # 2. Setup cache for the Text Encoder (if applicable)
            if self.engine_config.runtime_config.use_parallel_text_encoder:
                # Iterate through potential text encoders and set up the first one found
                for prefix in ["text_encoder_3", "text_encoder_2", "text_encoder"]:
                    if hasattr(self.model, prefix):
                        gpu_encoder_wrapper = getattr(self.model, prefix).encoder
                        cpu_encoder_model = getattr(
                            get_singleton_model_manager(), prefix
                        ).encoder

                        await self._initialize_and_warmup_cache(
                            gpu_module=gpu_encoder_wrapper,
                            cpu_module=cpu_encoder_model,
                            cache_class=TextEncoderBlockCache,
                            initial_gpu_blocks=gpu_encoder_wrapper.block,
                            component_name=f"Text Encoder ({prefix})",
                        )
                        # Once the relevant encoder is found and cached, stop searching.
                        break
            if self.engine_config.runtime_config.adjust_strategy == "p2p":
                # Register each worker's existing cache to NIXL (initial blocks)
                await self.register_existing_cache_with_nixl()
                logger.info(
                    "All NIXL agents created, peered and caches registered (serialized)."
                )

    def switch_parallel_env(
        self, world_size: int, ranks: List[int] = [], engine_config: EngineConfig = None
    ) -> None:
        """
        Called when switch parallel group env
        """
        ################################################################################
        #         change comm groups, and set activate comm group
        ################################################################################
        logger.debug(
            f"pid = {os.getpid()}, Worker's rank = {self.rank}, local_rank = {self.local_rank}"
        )
        logger.debug(f"ranks = {ranks}")
        get_parallel_groups().set_activate_world_group(len(ranks), ranks, [ranks])
        logger.debug(get_parallel_groups().activate_world_groups)

        parallel_config = engine_config.parallel_config
        assert len(ranks) == (
            parallel_config.tp_degree
            * parallel_config.sp_degree
            * parallel_config.pp_degree
            * parallel_config.cfg_degree
            * parallel_config.dp_degree
        ), "Product of parallel degrees must equal length of ranks"

        # TODO: The rank generator method can be changed later, parallel arrangement does not have to be sequential
        # # create rank map: 0->ranks[0], 1->ranks[1], ...
        rank_map = {i: rank for i, rank in enumerate(sorted(ranks))}

        # Use RankGenerator to generate basic rank combinations
        rank_generator = RankGenerator(
            tp=parallel_config.tp_degree,
            sp=parallel_config.sp_degree,
            pp=parallel_config.pp_degree,
            cfg=parallel_config.cfg_degree,
            dp=parallel_config.dp_degree,
            order="tp-sp-pp-cfg-dp",
        )

        base_tp_groups = rank_generator.get_ranks("tp")
        get_parallel_groups().set_activate_tp_group(
            degree=len(ranks),
            ranks=ranks,  # actual ranks
            group_ranks=[
                [rank_map[r] for r in group] for group in base_tp_groups
            ],  # mapped group_ranks
        )

        # get sequence parallel group
        base_sp_groups = rank_generator.get_ranks("sp")
        get_parallel_groups().set_activate_sp_group(
            degree=len(ranks),
            ranks=ranks,
            group_ranks=[[rank_map[r] for r in group] for group in base_sp_groups],
            ulysses_degree=parallel_config.ulysses_degree,
            ring_degree=parallel_config.ring_degree,
        )

        # get pipeline parallel group
        base_pp_groups = rank_generator.get_ranks("pp")
        get_parallel_groups().set_activate_pp_group(
            degree=len(ranks),
            ranks=ranks,
            group_ranks=[[rank_map[r] for r in group] for group in base_pp_groups],
        )

        # get cfg group
        base_cfg_groups = rank_generator.get_ranks("cfg")
        get_parallel_groups().set_activate_cfg_group(
            degree=len(ranks),
            ranks=ranks,
            group_ranks=[[rank_map[r] for r in group] for group in base_cfg_groups],
        )

        # get data parallel group
        base_dp_groups = rank_generator.get_ranks("dp")
        get_parallel_groups().set_activate_dp_group(
            degree=len(ranks),
            ranks=ranks,
            group_ranks=[[rank_map[r] for r in group] for group in base_dp_groups],
        )

        if engine_config.runtime_config.use_parallel_text_encoder:
            text_encoder_rank_generator = RankGenerator(
                tp=parallel_config.text_encoder_tp_degree,
                sp=1,
                pp=1,
                cfg=1,
                dp=len(ranks) // parallel_config.text_encoder_tp_degree,
                order="tp-sp-pp-cfg-dp",
            )

            base_text_encoder_tp_groups = text_encoder_rank_generator.get_ranks("tp")
            get_parallel_groups().set_activate_text_encoder_tp_group(
                degree=len(ranks),
                ranks=ranks,  # actual ranks
                group_ranks=[
                    [rank_map[r] for r in group]
                    for group in base_text_encoder_tp_groups
                ],  # mapped group_ranks
            )

        logger.debug(f"finished init_static_env, pid = {os.getpid()}")

        ################################################################################
        #         reset_runtime_state
        ################################################################################
        reset_runtime_state(engine_config)
        logger.debug(
            f"sp_parallel: my rank is {torch.distributed.get_rank()}, my sp_group is {get_sp_group().rank_in_group}, ulysses_degree is {get_sp_group().ulysses_rank}, ulysses_world_size is {get_sp_group().ulysses_world_size}, ulysses_group is {get_sp_group().ulysses_group}, ring_world_size is {get_sp_group().ring_world_size}, ring_degree is {get_sp_group().ring_rank}, ring_group is {get_sp_group().ring_group}, base_sp_groups is {base_sp_groups}"
        )
        torch.cuda.empty_cache()

        ################################################################################
        #         reset_cache_manager
        ################################################################################
        logger.info("begin reset_cache_manager")
        reset_cache_manager()
        logger.info("finished reset_cache_manager")

    def execute_model(
        self,
        engine_config: EngineConfig = None,
        input_config: InputConfig = None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ) -> None:
        """
        Run the all pipeline, need to modify to do 3 stage separate
        """
        get_runtime_state().set_p2p_state("computing")
        t1 = global_profiler.timer()

        self.states["running_task"] = task_id
        self.states["task_start_time"] = time.perf_counter()
        get_runtime_state().worker_state = self.states
        logger.debug(f"runtimestate is {get_runtime_state()}")
        if model_class == StableDiffusion3Pipeline:
            output = self.model(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                negative_prompt=input_config.negative_prompt,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(0),
                seed=input_config.seed,
                max_sequence_length=input_config.max_sequence_length,
            )

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )

            dp_group_index = get_data_parallel_rank()
            num_dp_groups = get_data_parallel_world_size()
            dp_batch_size = (1 + num_dp_groups - 1) // num_dp_groups
            logger.debug("finished vae decode, come into save")
            if (
                self.model.is_dp_last_group()
                and "Model_Profiler" not in input_config.prompt
            ):
                os.makedirs("results", exist_ok=True)
                for i, image in enumerate(output.images):
                    resolution = f"{input_config.width}x{input_config.height}_{i}"
                    image_rank = dp_group_index * dp_batch_size + i
                    image.save(f"results/stable_diffusion_3_result_{task_id}.png")
                    logger.info(
                        f"image {i} saved to ./results/stable_diffusion_3_result_{task_id}.png"
                    )

                t3 = global_profiler.timer()
                results["store"] = t3
            logger.debug("finished save")
        elif model_class == CogVideoXPipeline:
            output = self.model(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}

            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )
            if is_dp_last_group() and "Model_Profiler" not in input_config.prompt:
                os.makedirs("results", exist_ok=True)
                resolution = f"{input_config.width}x{input_config.height}x{input_config.num_frames}"
                output_filename = f"results/cogvideox_{task_id}.mp4"
                export_to_video(output, output_filename, fps=8)
                logger.info(f"output saved to {output_filename}")

                t3 = global_profiler.timer()
                results["store"] = t3
        elif model_class == FluxPipeline:
            get_runtime_state().runtime_config.dtype = torch.bfloat16
            output = self.model(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                # prompt_embeds=prompt_embeds,
                # pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )
            torch.cuda.empty_cache()
            if input_config.output_type == "pil":
                dp_group_index = get_data_parallel_rank()
                num_dp_groups = get_data_parallel_world_size()
                dp_batch_size = (
                    input_config.batch_size + num_dp_groups - 1
                ) // num_dp_groups
                resolution = f"{input_config.width}x{input_config.height}"
                if (
                    self.model.is_dp_last_group()
                    and "Model_Profiler" not in input_config.prompt
                ):
                    os.makedirs("results", exist_ok=True)
                    for i, image in enumerate(output.images):
                        image_rank = dp_group_index * dp_batch_size + i
                        image_name = f"flux_result_{task_id}.png"
                        image.save(f"results/{image_name}")
                        logger.info(f"image {i} saved to ./results/{image_name}")
                    t3 = global_profiler.timer()
                    results["store"] = t3
            elif input_config.output_type == "latent":
                t3 = global_profiler.timer()
                results["store"] = t3
        elif model_class == HunyuanDiTPipeline:
            output = self.model(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                use_resolution_binning=input_config.use_resolution_binning,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )
            if input_config.output_type == "pil":
                dp_group_index = get_data_parallel_rank()
                num_dp_groups = get_data_parallel_world_size()
                dp_batch_size = (
                    input_config.batch_size + num_dp_groups - 1
                ) // num_dp_groups
                resolution = f"{input_config.width}x{input_config.height}"
                if (
                    self.model.is_dp_last_group()
                    and "Model_Profiler" not in input_config.prompt
                ):
                    os.makedirs("results", exist_ok=True)
                    for i, image in enumerate(output.images):
                        image_rank = dp_group_index * dp_batch_size + i
                        image.save(f"results/hunyuandit_result_{task_id}.png")
                        logger.info(
                            f"image {i} saved to ./results/hunyuandit_result_{task_id}.png"
                        )
                    t3 = global_profiler.timer()
                    results["store"] = t3
        elif model_class == HunyuanVideoPipeline:
            output = self.model(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt=input_config.prompt,
                # prompt_embeds=prompt_embeds,
                # pooled_prompt_embeds=pooled_prompt_embeds,
                # prompt_attention_mask=prompt_attention_mask,
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            ).frames[0]

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )
            if (
                self.model.is_dp_last_group()
                and "Model_Profiler" not in input_config.prompt
            ):
                os.makedirs("results", exist_ok=True)
                resolution = f"{input_config.width}x{input_config.height}x{input_config.num_frames}"
                output_filename = f"results/hunyuan_video_{task_id}.mp4"
                export_to_video(output, output_filename, fps=15)
                logger.info(f"output saved to {output_filename}")
                t3 = global_profiler.timer()
                results["store"] = t3

        end_time = time.time()
        self.states = {}
        torch.cuda.empty_cache()
        get_runtime_state().set_p2p_state("free")
        return results

    def execute_decode_stage(
        self,
        latents,
        engine_config: EngineConfig = None,
        input_config: InputConfig = None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ) -> None:
        """
        Run the all pipeline, need to modify to do 3 stage separate
        """
        get_runtime_state().set_p2p_state("computing")
        parallel_info = (
            f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
            f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
            f"tp{engine_config.parallel_config.tp_degree}_"
            f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
            f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
        )
        logger.debug(f"begin decode stage, parallel_info is {parallel_info}")
        results = {}
        logger.debug(f"runtimestate is {get_runtime_state()}")
        get_runtime_state().worker_state = self.states
        if model_class == StableDiffusion3Pipeline:
            output = self.model.decode_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                negative_prompt=input_config.negative_prompt,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(0),
                seed=input_config.seed,
                max_sequence_length=input_config.max_sequence_length,
                latents=latents.to("cuda"),
                # src_rank=engine_config.encode_stage_rank,
                # dst_ranks=engine_config.diffusion_stage_ranks
            )

            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )

            dp_group_index = get_data_parallel_rank()
            num_dp_groups = get_data_parallel_world_size()
            dp_batch_size = (1 + num_dp_groups - 1) // num_dp_groups
            logger.debug("finished vae decode, come into save")
            if (
                self.model.is_dp_last_group()
                and "Model_Profiler" not in input_config.prompt
            ):
                os.makedirs("results", exist_ok=True)
                for i, image in enumerate(output.images):
                    resolution = f"{input_config.width}x{input_config.height}_{i}"
                    image_rank = dp_group_index * dp_batch_size + i
                    image.save(f"results/stable_diffusion_3_result_{task_id}.png")
                    logger.info(
                        f"image {i} saved to ./results/stable_diffusion_3_result_{task_id}.png"
                    )
            logger.debug("finished save")

        elif model_class == CogVideoXPipeline:
            output = self.model.decode_stage(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
                latents=latents.to("cuda"),
            ).frames[0]

            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )
            if is_dp_last_group() and "Model_Profiler" not in input_config.prompt:
                os.makedirs("results", exist_ok=True)
                resolution = f"{input_config.width}x{input_config.height}x{input_config.num_frames}"
                output_filename = f"results/cogvideox_{task_id}.mp4"
                export_to_video(output, output_filename, fps=8)
                logger.info(f"output saved to {output_filename}")

        elif model_class == FluxPipeline:
            get_runtime_state().runtime_config.dtype = torch.bfloat16
            output = self.model.decode_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
                latents=latents.to("cuda"),
            )

            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )
            if input_config.output_type == "pil":
                dp_group_index = get_data_parallel_rank()
                num_dp_groups = get_data_parallel_world_size()
                dp_batch_size = (
                    input_config.batch_size + num_dp_groups - 1
                ) // num_dp_groups
                resolution = f"{input_config.width}x{input_config.height}"
                if (
                    self.model.is_dp_last_group()
                    and "Model_Profiler" not in input_config.prompt
                ):
                    os.makedirs("results", exist_ok=True)
                    for i, image in enumerate(output.images):
                        image_rank = dp_group_index * dp_batch_size + i
                        image_name = f"flux_result_{task_id}.png"
                        image.save(f"results/{image_name}")
                        logger.info(f"image {i} saved to ./results/{image_name}")

        elif model_class == HunyuanDiTPipeline:
            output = self.model.decode_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                use_resolution_binning=input_config.use_resolution_binning,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
                latents=latents.to("cuda"),
            )

            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )
            if input_config.output_type == "pil":
                dp_group_index = get_data_parallel_rank()
                num_dp_groups = get_data_parallel_world_size()
                dp_batch_size = (
                    input_config.batch_size + num_dp_groups - 1
                ) // num_dp_groups
                resolution = f"{input_config.width}x{input_config.height}"
                if (
                    self.model.is_dp_last_group()
                    and "Model_Profiler" not in input_config.prompt
                ):
                    os.makedirs("results", exist_ok=True)
                    for i, image in enumerate(output.images):
                        image_rank = dp_group_index * dp_batch_size + i
                        image.save(f"results/hunyuandit_result_{task_id}.png")
                        logger.info(
                            f"image {i} saved to ./results/hunyuandit_result_{task_id}.png"
                        )

        elif model_class == HunyuanVideoPipeline:
            output = self.model.decode_stage(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
                latents=latents.to("cuda"),
            ).frames[0]

            parallel_info = (
                f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
                f"tp{engine_config.parallel_config.tp_degree}_"
                f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
                f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
            )
            if (
                self.model.is_dp_last_group()
                and "Model_Profiler" not in input_config.prompt
            ):
                os.makedirs("results", exist_ok=True)
                resolution = f"{input_config.width}x{input_config.height}x{input_config.num_frames}"
                output_filename = f"results/hunyuan_video_{task_id}.mp4"
                export_to_video(output, output_filename, fps=15)
                logger.info(f"output saved to {output_filename}")

        torch.cuda.empty_cache()
        get_runtime_state().set_p2p_state("free")
        return results

    def execute_diffusion_stage(
        self,
        encode_stage_results,
        engine_config: EngineConfig = None,
        input_config: InputConfig = None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ) -> None:
        """
        Run the all pipeline, need to modify to do 3 stage separate
        """
        get_runtime_state().set_p2p_state("computing")
        t1 = global_profiler.timer()

        self.states["running_task"] = task_id
        self.states["task_start_time"] = time.perf_counter()
        get_runtime_state().worker_state = self.states
        logger.debug(f"runtimestate is {get_runtime_state()}")
        parallel_info = (
            f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
            f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
            f"tp{engine_config.parallel_config.tp_degree}_"
            f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
            f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
        )
        if model_class == StableDiffusion3Pipeline:
            latents = self.model.diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(0),
                seed=input_config.seed,
                max_sequence_length=input_config.max_sequence_length,
                prompt_embeds=encode_stage_results["prompt_embeds"],
                # negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=encode_stage_results["pooled_prompt_embeds"],
                # negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                # src_rank=engine_config.encode_stage_rank,
                # dst_ranks=engine_config.diffusion_stage_ranks
            )

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    return (results, latents)
                else:
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                return (results, latents)
            else:
                return (results, None)
        elif model_class == CogVideoXPipeline:
            latents = self.model.diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt_embeds=encode_stage_results["prompt_embeds"],
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")

            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    return (results, latents)
                else:
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                return (results, latents)
            else:
                return (results, None)
        elif model_class == FluxPipeline:
            get_runtime_state().runtime_config.dtype = torch.bfloat16
            latents = self.model.diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                prompt_embeds=encode_stage_results["prompt_embeds"],
                pooled_prompt_embeds=encode_stage_results["pooled_prompt_embeds"],
                text_ids=encode_stage_results["text_ids"],
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")

            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    return (results, latents)
                else:
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                return (results, latents)
            else:
                return (results, None)
        elif model_class == HunyuanDiTPipeline:
            latents = self.model.diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                prompt_embeds=encode_stage_results["prompt_embeds"],
                prompt_attention_mask=encode_stage_results["prompt_attention_mask"],
                prompt_embeds_2=encode_stage_results["prompt_embeds_2"],
                prompt_attention_mask_2=encode_stage_results["prompt_attention_mask_2"],
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                use_resolution_binning=input_config.use_resolution_binning,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            logger.debug(
                f"self.rank is {self.rank}, parallel_info is {parallel_info}, engine_config.diffusion_stage_ranks is {engine_config.diffusion_stage_ranks}, latents shape is {latents.shape}"
            )
            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    return (results, latents)
                else:
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                logger.debug(
                    f"have latents, self.rank is {self.rank}, parallel_info is {parallel_info}, engine_config.diffusion_stage_ranks is {engine_config.diffusion_stage_ranks}, latents shape is {latents.shape}"
                )
                return (results, latents)
            else:
                logger.debug(
                    f"do not have latents, self.rank is {self.rank}, parallel_info is {parallel_info}, engine_config.diffusion_stage_ranks is {engine_config.diffusion_stage_ranks}, latents shape is {latents.shape}"
                )
                return (results, None)
        elif model_class == HunyuanVideoPipeline:
            latents = self.model.diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt_embeds=encode_stage_results["prompt_embeds"],
                pooled_prompt_embeds=encode_stage_results["pooled_prompt_embeds"],
                prompt_attention_mask=encode_stage_results["prompt_attention_mask"],
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")

            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    return (results, latents)
                else:
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                return (results, latents)
            else:
                return (results, None)
        else:
            results = None
            get_runtime_state().set_p2p_state("free")
            return None

    def execute_encode_diffusion_stage(
        self,
        engine_config: EngineConfig = None,
        input_config: InputConfig = None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ) -> None:
        """
        Run the all pipeline, need to modify to do 3 stage separate
        """
        get_runtime_state().set_p2p_state("computing")
        t1 = global_profiler.timer()

        self.states["running_task"] = task_id
        self.states["task_start_time"] = time.perf_counter()

        get_runtime_state().worker_state = self.states
        logger.debug(f"runtimestate is {get_runtime_state()}")
        parallel_info = (
            f"dp1_cfg{engine_config.parallel_config.cfg_degree}_"
            f"ulysses{engine_config.parallel_config.ulysses_degree}_ring{engine_config.parallel_config.ring_degree}_"
            f"tp{engine_config.parallel_config.tp_degree}_"
            f"texttp{engine_config.parallel_config.text_encoder_tp_degree}_"
            f"pp{engine_config.parallel_config.pp_degree}_patch{engine_config.parallel_config.pp_degree}"
        )
        if model_class == StableDiffusion3Pipeline:
            logger.debug(f"task_id is {task_id}, before enter encode_diffusion_stage")
            latents = self.model.encode_diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                negative_prompt=input_config.negative_prompt,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(0),
                seed=input_config.seed,
                max_sequence_length=input_config.max_sequence_length,
            )

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, latents)
                else:
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, latents)
            else:
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, None)
        elif model_class == CogVideoXPipeline:
            latents = self.model.encode_diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")

            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, latents)
                else:
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, latents)
            else:
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, None)
        elif model_class == FluxPipeline:
            get_runtime_state().runtime_config.dtype = torch.bfloat16
            latents = self.model.encode_diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                # prompt_embeds=prompt_embeds,
                # pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")

            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, latents)
                else:
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, latents)
            else:
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, None)
        elif model_class == HunyuanDiTPipeline:
            latents = self.model.encode_diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                use_resolution_binning=input_config.use_resolution_binning,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            logger.debug(
                f"self.rank is {self.rank}, parallel_info is {parallel_info}, engine_config.diffusion_stage_ranks is {engine_config.diffusion_stage_ranks}, latents shape is {latents.shape}"
            )
            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, latents)
                else:
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                logger.debug(
                    f"have latents, self.rank is {self.rank}, parallel_info is {parallel_info}, engine_config.diffusion_stage_ranks is {engine_config.diffusion_stage_ranks}, latents shape is {latents.shape}"
                )
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, latents)
            else:
                logger.debug(
                    f"do not have latents, self.rank is {self.rank}, parallel_info is {parallel_info}, engine_config.diffusion_stage_ranks is {engine_config.diffusion_stage_ranks}, latents shape is {latents.shape}"
                )
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, None)
        elif model_class == HunyuanVideoPipeline:
            latents = self.model.encode_diffusion_stage(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt=input_config.prompt,
                # prompt_embeds=prompt_embeds,
                # pooled_prompt_embeds=pooled_prompt_embeds,
                # prompt_attention_mask=prompt_attention_mask,
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )

            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")

            if engine_config.parallel_config.pp_degree > 1:
                if is_dp_last_group():
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, latents)
                else:
                    t3 = global_profiler.timer()
                    results["store"] = t3
                    return (results, None)
            elif self.rank == engine_config.diffusion_stage_ranks[0]:
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, latents)
            else:
                t3 = global_profiler.timer()
                results["store"] = t3
                return (results, None)
        else:
            results = None
            get_runtime_state().set_p2p_state("free")
            return None

    def execute_encode_stage(
        self,
        engine_config: EngineConfig = None,
        input_config: InputConfig = None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ) -> None:
        """
        Run the all pipeline, need to modify to do 3 stage separate
        """
        get_runtime_state().set_p2p_state("computing")
        t1 = global_profiler.timer()

        self.states["running_task"] = task_id
        self.states["task_start_time"] = time.perf_counter()
        get_runtime_state().worker_state = self.states
        logger.debug(f"runtimestate is {get_runtime_state()}")
        encode_stage_results = {}
        if model_class == StableDiffusion3Pipeline:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.model.encode_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                negative_prompt=input_config.negative_prompt,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(0),
                seed=input_config.seed,
                max_sequence_length=input_config.max_sequence_length,
                # dst_ranks=engine_config.diffusion_stage_ranks
            )
            encode_stage_results["prompt_embeds"] = prompt_embeds
            encode_stage_results["pooled_prompt_embeds"] = pooled_prompt_embeds
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            return encode_stage_results
        elif model_class == CogVideoXPipeline:
            prompt_embeds = self.model.encode_stage(
                height=input_config.height,
                width=input_config.width,
                num_frames=input_config.num_frames,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )
            encode_stage_results["prompt_embeds"] = prompt_embeds
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            return encode_stage_results
        elif model_class == FluxPipeline:
            get_runtime_state().runtime_config.dtype = torch.bfloat16
            prompt_embeds, pooled_prompt_embeds, text_ids = self.model.encode_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            encode_stage_results["prompt_embeds"] = prompt_embeds
            encode_stage_results["pooled_prompt_embeds"] = pooled_prompt_embeds
            encode_stage_results["text_ids"] = text_ids
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            return encode_stage_results
        elif model_class == HunyuanDiTPipeline:
            (
                prompt_embeds,
                prompt_attention_mask,
                prompt_embeds_2,
                prompt_attention_mask_2,
            ) = self.model.encode_stage(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                use_resolution_binning=input_config.use_resolution_binning,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            encode_stage_results["prompt_embeds"] = prompt_embeds
            encode_stage_results["prompt_attention_mask"] = prompt_attention_mask
            encode_stage_results["prompt_embeds_2"] = prompt_embeds_2
            encode_stage_results["prompt_attention_mask_2"] = prompt_attention_mask_2
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            return encode_stage_results
        elif model_class == HunyuanVideoPipeline:
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = (
                self.model.encode_stage(
                    height=input_config.height,
                    width=input_config.width,
                    num_frames=input_config.num_frames,
                    prompt=input_config.prompt,
                    num_inference_steps=input_config.num_inference_steps,
                    generator=torch.Generator(device="cuda").manual_seed(
                        input_config.seed
                    ),
                )
            )
            encode_stage_results["prompt_embeds"] = prompt_embeds
            encode_stage_results["prompt_attention_mask"] = prompt_attention_mask
            encode_stage_results["pooled_prompt_embeds"] = pooled_prompt_embeds
            t2 = global_profiler.timer()
            results = {"before": t1, "after": t2, "inner_results": self.states.copy()}
            self.states = {}
            torch.cuda.empty_cache()
            get_runtime_state().set_p2p_state("free")
            return encode_stage_results
        else:
            results = None
            get_runtime_state().set_p2p_state("free")
            return None

    def get_worker_state(self):
        result = {}
        result["rank"] = self.rank
        result["running_task"] = self.states.get("running_task", None)
        result["last_detected_time"] = time.perf_counter()
        result["task_start_time"] = self.states.get("task_start_time", None)
        result["memory"] = torch.cuda.memory_allocated()
        result["running_stage"] = self.states.get("stage_name", None)
        result["estimated_running_time"] = self.states.get(
            "estimated_running_time", None
        )
        if result["running_stage"] is not None:
            stage_running_time = result["last_detected_time"] - self.states.get(
                f"{result['running_stage']}_start_time", 0
            )
            result["stage_running_time"] = (
                stage_running_time if stage_running_time else None
            )
        else:
            result["stage_running_time"] = None
        result["task_running_time"] = (
            result["last_detected_time"] - result["task_start_time"]
            if result["task_start_time"]
            else None
        )
        if (
            result["estimated_running_time"] is not None
            and result["task_running_time"] is not None
        ):
            result["estimated_idle_time"] = (
                result["estimated_running_time"] - result["task_running_time"]
                if result["estimated_running_time"] - result["task_running_time"] > 0
                else 2
            )
        else:
            result["estimated_idle_time"] = 0
        return result

    def load_model(self, engine_config: EngineConfig = None) -> None:
        """
        Only used to load the model parameters in the first static placement
        """
        pass

    async def hotspa_model(self, engine_config: EngineConfig = None) -> None:
        """
        Used to comm the parameter when we have to do hotspa
        """
        get_runtime_state().set_p2p_state("adjusting")
        # param slice metadata before hotspa
        gpu0_metadata = get_runtime_state().cur_metadata
        # param slice metadata after hotspa
        gpu1_metadata = get_gpu_metadata(
            get_pipeline_parallel_rank(),
            get_pipeline_parallel_world_size(),
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
        )
        logger.info(f"in hotsopa_model: gpu_metadata={gpu1_metadata}")
        ################################################################################
        #         change  model's weight
        ################################################################################
        old_tp_range = gpu0_metadata["tensor_parallel"]
        new_tp_range = gpu1_metadata["tensor_parallel"]
        old_pp_range = gpu0_metadata["pipeline_parallel"]
        new_pp_range = gpu1_metadata["pipeline_parallel"]
        adjust_strategy = engine_config.runtime_config.adjust_strategy
        if adjust_strategy == "base":
            logger.debug(f"self.__class__.__name__={self.__class__.__name__}")
            if (
                hasattr(self.model.transformer, "single_transformer_blocks")
                and "Flux" in self.model.__class__.__name__
            ):
                self.model.transformer = adjust_pipeline(
                    transformer=self.model.transformer,
                    cpu_full_transformer=get_singleton_model_manager().transformer,
                    old_pp_range=old_pp_range,
                    new_pp_range=new_pp_range,
                    old_tp_range=old_tp_range,
                    new_tp_range=new_tp_range,
                    slice_bias=True,
                    transformer_blocks_name=[
                        "transformer_blocks",
                        "single_transformer_blocks",
                    ],
                    transformer_blocks_num=[19, 38],
                )
            elif (
                hasattr(self.model.transformer, "single_transformer_blocks")
                and "HunyuanVideo" in self.model.__class__.__name__
            ):
                logger.info("adjust hunyuanvideo transformer blocks")
                self.model.transformer = adjust_pipeline(
                    transformer=self.model.transformer,
                    cpu_full_transformer=get_singleton_model_manager().transformer,
                    old_pp_range=old_pp_range,
                    new_pp_range=new_pp_range,
                    old_tp_range=old_tp_range,
                    new_tp_range=new_tp_range,
                    slice_bias=True,
                    transformer_blocks_name=[
                        "transformer_blocks",
                        "single_transformer_blocks",
                    ],
                    transformer_blocks_num=[20, 40],
                )
            elif (
                hasattr(self.model.transformer, "blocks")
                and "HunyuanDiT" in self.model.__class__.__name__
            ):
                self.model.transformer = adjust_pipeline(
                    transformer=self.model.transformer,
                    cpu_full_transformer=get_singleton_model_manager().transformer,
                    old_pp_range=old_pp_range,
                    new_pp_range=new_pp_range,
                    old_tp_range=old_tp_range,
                    new_tp_range=new_tp_range,
                    slice_bias=True,
                    transformer_blocks_name=["blocks"],
                )
            else:
                self.model.transformer = adjust_pipeline(
                    transformer=self.model.transformer,
                    cpu_full_transformer=get_singleton_model_manager().transformer,
                    old_pp_range=old_pp_range,
                    new_pp_range=new_pp_range,
                    old_tp_range=old_tp_range,
                    new_tp_range=new_tp_range,
                    slice_bias=True,
                )
        elif adjust_strategy in ["p2p", "cache"]:
            adjusted_transformer = await self.model.transformer.cache.adjust_pipeline(
                transformer=self.model.transformer,
                cpu_full_transformer=get_singleton_model_manager().transformer,
                old_pp_range=old_pp_range,
                new_pp_range=new_pp_range,
                old_tp_range=old_tp_range,
                new_tp_range=new_tp_range,
                slice_bias=True,
            )
            self.model.transformer = adjusted_transformer
        ################################################################################
        #         set new runtime metadata
        ################################################################################
        get_runtime_state()._set_metadata(gpu1_metadata)
        if engine_config.runtime_config.use_parallel_text_encoder:
            text_encoder_gpu0_metadata = get_runtime_state().text_encoder_cur_metadata
            # param slice metadata after hotspa
            text_encoder_gpu1_metadata = get_gpu_metadata(
                0,
                1,
                get_text_encoder_tensor_model_parallel_rank(),
                get_text_encoder_tensor_model_parallel_world_size(),
            )
            ################################################################################
            #         change  text encdoer's weight
            ################################################################################
            text_encoder_old_tp_range = text_encoder_gpu0_metadata["tensor_parallel"]
            text_encoder_new_tp_range = text_encoder_gpu1_metadata["tensor_parallel"]
            text_encoder_old_pp_range = text_encoder_gpu0_metadata["pipeline_parallel"]
            text_encoder_new_pp_range = text_encoder_gpu1_metadata["pipeline_parallel"]

            if adjust_strategy in ["cache", "p2p"]:
                if hasattr(self.model, "text_encoder_3"):
                    adjusted_text_encoder_3 = await self.model.text_encoder_3.encoder.cache.adjust_pipeline(
                        text_encoder=self.model.text_encoder_3.encoder,
                        cpu_full_text_encoder=get_singleton_model_manager().text_encoder_3.encoder,
                        old_pp_range=text_encoder_old_pp_range,
                        new_pp_range=text_encoder_new_pp_range,
                        old_tp_range=text_encoder_old_tp_range,
                        new_tp_range=text_encoder_new_tp_range,
                        slice_bias=True,
                    )
                    self.model.text_encoder_3.encoder = adjusted_text_encoder_3
                elif hasattr(self.model, "text_encoder_2"):
                    adjusted_text_encoder_2 = await self.model.text_encoder_2.encoder.cache.adjust_pipeline(
                        text_encoder=self.model.text_encoder_2.encoder,
                        cpu_full_text_encoder=get_singleton_model_manager().text_encoder_2.encoder,
                        old_pp_range=text_encoder_old_pp_range,
                        new_pp_range=text_encoder_new_pp_range,
                        old_tp_range=text_encoder_old_tp_range,
                        new_tp_range=text_encoder_new_tp_range,
                        slice_bias=True,
                    )
                    self.model.text_encoder_2.encoder = adjusted_text_encoder_2
                elif hasattr(self.model, "text_encoder"):
                    adjusted_text_encoder = await self.model.text_encoder.encoder.cache.adjust_pipeline(
                        text_encoder=self.model.text_encoder.encoder,
                        cpu_full_text_encoder=get_singleton_model_manager().text_encoder.encoder,
                        old_pp_range=text_encoder_old_pp_range,
                        new_pp_range=text_encoder_new_pp_range,
                        old_tp_range=text_encoder_old_tp_range,
                        new_tp_range=text_encoder_new_tp_range,
                        slice_bias=True,
                    )
                    self.model.text_encoder.encoder = adjusted_text_encoder
            else:
                if hasattr(self.model, "text_encoder_3"):
                    self.model.text_encoder_3.encoder = adjust_text_encoder(
                        transformer=self.model.text_encoder_3.encoder,
                        cpu_full_transformer=get_singleton_model_manager().text_encoder_3.encoder,
                        old_pp_range=text_encoder_old_pp_range,
                        new_pp_range=text_encoder_new_pp_range,
                        old_tp_range=text_encoder_old_tp_range,
                        new_tp_range=text_encoder_new_tp_range,
                        slice_bias=True,
                    )
                elif hasattr(self.model, "text_encoder_2"):
                    self.model.text_encoder_2.encoder = adjust_text_encoder(
                        transformer=self.model.text_encoder_2.encoder,
                        cpu_full_transformer=get_singleton_model_manager().text_encoder_2.encoder,
                        old_pp_range=text_encoder_old_pp_range,
                        new_pp_range=text_encoder_new_pp_range,
                        old_tp_range=text_encoder_old_tp_range,
                        new_tp_range=text_encoder_new_tp_range,
                        slice_bias=True,
                    )
                elif hasattr(self.model, "text_encoder"):
                    self.model.text_encoder.encoder = adjust_text_encoder(
                        transformer=self.model.text_encoder.encoder,
                        cpu_full_transformer=get_singleton_model_manager().text_encoder.encoder,
                        old_pp_range=text_encoder_old_pp_range,
                        new_pp_range=text_encoder_new_pp_range,
                        old_tp_range=text_encoder_old_tp_range,
                        new_tp_range=text_encoder_new_tp_range,
                        slice_bias=True,
                    )
            get_runtime_state()._set_text_encoder_metadata(text_encoder_gpu1_metadata)

        ################################################################################
        #         update model's transformer_blocks_metadata
        ################################################################################
        if hasattr(self.model.transformer, "single_transformer_blocks"):
            get_runtime_state().runtime_config.dtype = torch.bfloat16
            self.model.transformer._set_transformer_blocks_metadata(
                get_singleton_model_manager().transformer,
                ["transformer_blocks", "single_transformer_blocks"],
            )

        ################################################################################
        #         update model's wrapped_layers
        ################################################################################
        self.model.transformer.wrapped_layers = []
        layer_classes = set(
            hetuDiTLayerWrappersRegister._HETUDIT_LAYER_MAPPING.values()
        )
        attention_processor_classes = set(
            hetuDiTAttentionProcessorRegister._HETUDIT_ATTENTION_PROCESSOR_MAPPING.values()
        )
        valid_classes = layer_classes.union(attention_processor_classes)
        self.model.transformer.wrapped_layers = [
            sub_module
            for sub_module in self.model.transformer.modules()
            if isinstance(sub_module, tuple(valid_classes))
        ]

        ################################################################################
        #         update text encoder's wrapped_layers
        ################################################################################
        if engine_config.runtime_config.use_parallel_text_encoder:
            if hasattr(self.model, "text_encoder_3"):
                self.model.text_encoder_3.wrapped_layers = []
                layer_classes = set(
                    hetuDiTLayerWrappersRegister._HETUDIT_LAYER_MAPPING.values()
                )
                attention_processor_classes = set(
                    hetuDiTAttentionProcessorRegister._HETUDIT_ATTENTION_PROCESSOR_MAPPING.values()
                )
                valid_classes = layer_classes.union(attention_processor_classes)
                self.model.text_encoder_3.wrapped_layers = [
                    sub_module
                    for sub_module in self.model.text_encoder_3.modules()
                    if isinstance(sub_module, tuple(valid_classes))
                ]
            elif hasattr(self.model, "text_encoder_2"):  # for flux
                self.model.text_encoder_2.wrapped_layers = []
                layer_classes = set(
                    hetuDiTLayerWrappersRegister._HETUDIT_LAYER_MAPPING.values()
                )
                attention_processor_classes = set(
                    hetuDiTAttentionProcessorRegister._HETUDIT_ATTENTION_PROCESSOR_MAPPING.values()
                )
                valid_classes = layer_classes.union(attention_processor_classes)
                self.model.text_encoder_2.wrapped_layers = [
                    sub_module
                    for sub_module in self.model.text_encoder_2.modules()
                    if isinstance(sub_module, tuple(valid_classes))
                ]
            elif hasattr(self.model, "text_encoder"):  # for cogvideo
                self.model.text_encoder.wrapped_layers = []
                layer_classes = set(
                    hetuDiTLayerWrappersRegister._HETUDIT_LAYER_MAPPING.values()
                )
                attention_processor_classes = set(
                    hetuDiTAttentionProcessorRegister._HETUDIT_ATTENTION_PROCESSOR_MAPPING.values()
                )
                valid_classes = layer_classes.union(attention_processor_classes)
                self.model.text_encoder.wrapped_layers = [
                    sub_module
                    for sub_module in self.model.text_encoder.modules()
                    if isinstance(sub_module, tuple(valid_classes))
                ]
        ################################################################################
        #         change some layer's inited variable, such as hetuDiTJointAttnProcessor2_0's hybrid_seq_parallel_attn
        ################################################################################
        for wrapped_layer in self.model.transformer.wrapped_layers:
            if isinstance(wrapped_layer, hetuDiTAttentionWrapper):
                if isinstance(wrapped_layer.processor, hetuDiTJointAttnProcessor2_0):
                    use_long_ctx_attn_kvcache = True
                    wrapped_layer.processor.use_long_ctx_attn_kvcache = (
                        HAS_LONG_CTX_ATTN
                        and use_long_ctx_attn_kvcache
                        and get_sequence_parallel_world_size() > 1
                    )
                    if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                        from hetu_dit.core.parallel import (
                            hetuDiTJointLongContextAttention,
                            hetuDiTUlyssesAttention,
                        )

                        if HAS_FLASH_ATTN:
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTJointLongContextAttention(
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache
                            )
                        else:
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                                use_fa=False,
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache,
                            )
                elif isinstance(wrapped_layer.processor, hetuDiTAttnProcessor2_0):
                    use_long_ctx_attn_kvcache = True
                    wrapped_layer.processor.use_long_ctx_attn_kvcache = (
                        HAS_LONG_CTX_ATTN
                        and use_long_ctx_attn_kvcache
                        and get_sequence_parallel_world_size() > 1
                    )
                    if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                        from hetu_dit.core.parallel import (
                            hetuDiTLongContextAttention,
                            hetuDiTUlyssesAttention,
                        )

                        if HAS_FLASH_ATTN:
                            # self.hybrid_seq_parallel_attn = LongContextAttention()
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTLongContextAttention(
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache
                            )
                        else:
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                                use_fa=False,
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache,
                            )
                    else:
                        wrapped_layer.processor.hybrid_seq_parallel_attn = None
                elif isinstance(wrapped_layer.processor, hetuDiTFluxAttnProcessor2_0):
                    use_long_ctx_attn_kvcache = True
                    wrapped_layer.processor.use_long_ctx_attn_kvcache = (
                        HAS_LONG_CTX_ATTN
                        and use_long_ctx_attn_kvcache
                        and get_sequence_parallel_world_size() > 1
                    )
                    if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                        from hetu_dit.core.parallel import (
                            hetuDiTFluxLongContextAttention,
                            hetuDiTUlyssesAttention,
                        )

                        if HAS_FLASH_ATTN:
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTFluxLongContextAttention(
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache
                            )
                        else:
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                                use_fa=False,
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache,
                            )
                elif isinstance(
                    wrapped_layer.processor, hetuDiTCogVideoXAttnProcessor2_0
                ):
                    use_long_ctx_attn_kvcache = True
                    wrapped_layer.processor.use_long_ctx_attn_kvcache = (
                        HAS_LONG_CTX_ATTN
                        and use_long_ctx_attn_kvcache
                        and get_sequence_parallel_world_size() > 1
                    )
                    if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                        from hetu_dit.core.parallel import (
                            hetuDiTJointLongContextAttention,
                            hetuDiTUlyssesAttention,
                            hetuDiTLongContextAttention,
                        )

                        if (
                            HAS_FLASH_ATTN
                            and get_runtime_state().split_text_embed_in_sp
                        ):
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTLongContextAttention(
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache
                            )
                        else:
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTJointLongContextAttention(
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache
                            )
                    else:
                        wrapped_layer.processor.hybrid_seq_parallel_attn = None
                elif isinstance(
                    wrapped_layer.processor, hetuDiTHunyuanAttnProcessor2_0
                ):
                    use_long_ctx_attn_kvcache = True
                    wrapped_layer.processor.use_long_ctx_attn_kvcache = (
                        HAS_LONG_CTX_ATTN
                        and use_long_ctx_attn_kvcache
                        and get_sequence_parallel_world_size() > 1
                    )
                    if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                        from hetu_dit.core.parallel import (
                            hetuDiTLongContextAttention,
                            hetuDiTUlyssesAttention,
                        )

                        if HAS_FLASH_ATTN:
                            # self.hybrid_seq_parallel_attn = LongContextAttention()
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTLongContextAttention(
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache
                            )
                        else:
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                                use_fa=False,
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache,
                            )
                    else:
                        wrapped_layer.processor.hybrid_seq_parallel_attn = None
                elif isinstance(
                    wrapped_layer.processor, hetuDiTHunyuanVideoAttnProcessor2_0
                ):
                    use_long_ctx_attn_kvcache = True
                    wrapped_layer.processor.use_long_ctx_attn_kvcache = (
                        HAS_LONG_CTX_ATTN
                        and use_long_ctx_attn_kvcache
                        and get_sequence_parallel_world_size() > 1
                    )
                    if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                        from hetu_dit.core.parallel import (
                            hetuDiTJointLongContextAttention,
                            hetuDiTUlyssesAttention,
                        )

                        if HAS_FLASH_ATTN:
                            # self.hybrid_seq_parallel_attn = LongContextAttention()
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTJointLongContextAttention(
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache
                            )
                        else:
                            wrapped_layer.processor.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                                use_fa=False,
                                use_kv_cache=wrapped_layer.processor.use_long_ctx_attn_kvcache,
                            )
                    else:
                        wrapped_layer.processor.hybrid_seq_parallel_attn = None

        ################################################################################
        #         reset activation cache of every layer and register cache in CacheManager
        ################################################################################
        self.model.reset_activation_cache()
        self.model.transformer._register_cache()
        await self.refresh_nixl_registrations()
        get_runtime_state().set_p2p_state("free")

    def init_worker_distributed_environment(
        self,
        engine_config: EngineConfig,
        num_gpus: int,
    ) -> None:
        """Initialize the distributed environment.
        Only called in the first static placement, usually used to init the global distributed env, but not the small comm group
        """
        if torch.distributed.is_initialized():
            torch_world_size = torch.distributed.get_world_size()
            if torch_world_size != engine_config.parallel_config.world_size:
                raise RuntimeError(
                    "torch.distributed is already initialized but the torch world "
                    "size does not match parallel_config.world_size "
                    f"({torch_world_size} vs. {engine_config.parallel_config.world_size})."
                )
        elif not self.dist_init_method:
            raise ValueError(
                "distributed_init_method must be set if torch.distributed "
                "is not already initialized"
            )
        else:
            logger.info(
                f"init_worker_distributed_environment's pid = {os.getpid()}, self.dist_init_method = {self.dist_init_method}, rank = {self.rank}, num_gpus = {num_gpus}"
            )
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=num_gpus,
                rank=self.rank,
                init_method=self.dist_init_method,
                # init_method="env://",
                timeout=timedelta(seconds=60),
            )
            logger.debug(f"pid = {os.getpid()}, after")

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())
        logger.debug("before parallel_group's post_init")
        get_parallel_groups().lazy_init(is_serving=True)
        logger.debug("finish parallel_group's post init")

    def init_singleton_cpu_model(
        self, engine_config: EngineConfig, model_class, use_text_encoder_parallel
    ) -> None:
        """Initialize the model on the CPU.

        This is used to load the model on the CPU, which is then transferred to
        the GPU in the `init_singleton_gpu_model` method.
        """
        logger.info("set singleton_model_manager")
        set_singleton_model_manager(
            model_class, engine_config.model_config.model, use_text_encoder_parallel
        )
        logger.info("set singleton_model_manager done")

    async def get_nixl_metadata(self) -> Tuple[int, bytes]:
        """
        [RPC Endpoint]
        Called by the main coordinator process to get the NIXL metadata of this worker node.
        The metadata contains the identity and connection information of the NIXL Agent.
        """
        print(f"Rank {self.rank}: Received get_nixl_metadata request.")
        if not hasattr(self, "nixl_manager") or self.nixl_manager is None:
            raise RuntimeError(
                f"Rank {self.rank}: NixlP2PManager has not been initialized!"
            )
        print(f"Rank {self.rank}: Returning NIXL metadata")
        return (self.rank, self.nixl_manager.get_metadata())

    async def init_nixl_peers(self, all_metadata_list: List[Tuple[int, bytes]]):
        """
        [RPC Endpoint]
        Called by the main coordinator process to initialize NIXL P2P connections using metadata from all nodes.
        After this method is executed, this node can communicate with all other nodes via NIXL.
        """
        logger.debug(f"Rank {self.rank}: Received init_nixl_peers request.")
        if not hasattr(self, "nixl_manager") or self.nixl_manager is None:
            raise RuntimeError(
                f"Rank {self.rank}: NixlP2PManager has not been initialized!"
            )

        self.nixl_manager.init_peers(all_metadata_list)
        logger.info(f"Rank {self.rank}: NIXL peer connection initialization complete.")

    async def rpc_find_sources(self, serializable_pieces: List[Dict]) -> List[int]:
        """
        [RPC Endpoint] Based on the pieces broadcast by a peer, return a list of piece uids that this node can provide.
        Supports Transformer and TextEncoder.
        """
        if get_runtime_state().runtime_config.adjust_strategy != "p2p":
            return []
        if get_runtime_state().get_p2p_state() == "adjusting":
            logger.warning(
                f"Rank {self.rank}: Adjustment in progress, rejecting rpc_find_sources request."
            )
            return []
        if get_runtime_state().get_p2p_state() == "computing":
            print(f"Rank {self.rank}: In computation, implementing overlap")

        logger.info(
            f"Rank {self.rank}: Received source finding request for {len(serializable_pieces)} data pieces."
        )

        # Prefetch available cache
        caches: Dict[str, Any] = {}
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "cache"
        ):
            caches["Transformer"] = self.model.transformer.cache

        # Select the first available text encoder cache
        for te_name in ["text_encoder_3", "text_encoder_2", "text_encoder"]:
            if hasattr(self.model, te_name):
                enc = getattr(self.model, te_name)
                if hasattr(enc, "encoder") and hasattr(enc.encoder, "cache"):
                    caches["TextEncoder"] = enc.encoder.cache
                    break

        provided_piece_uids = []
        pieces_to_check = {p["uid"]: p for p in serializable_pieces}

        for uid, piece_info in pieces_to_check.items():
            block_index = piece_info["block_index"]
            full_size = piece_info["full_size"]
            req_start, req_end = piece_info["abs_start_idx"], piece_info["abs_end_idx"]

            component = piece_info.get("component", "Transformer")
            cache = caches.get(component, None)
            if cache is None:
                continue

            if block_index not in cache.cache:
                continue

            _, tp_range = cache.cache[block_index]
            my_start, my_end = cache.range_to_indices(tp_range, full_size)
            if my_start <= req_start and my_end >= req_end:
                provided_piece_uids.append(uid)

        logger.info(
            f"Rank {self.rank}: Can provide {len(provided_piece_uids)} data pieces."
        )
        return provided_piece_uids

    async def rpc_nixl_send_data(
        self,
        pieces_to_send: List["NeededPiece"],
        remote_xfer_desc_bytes: bytes,
        remote_partial_metadata: bytes,
        requester_rank: int,
    ):
        try:
            logger.debug(
                f"[NIXL][SEND] rank={self.rank} begin rpc_nixl_send_data; requester_rank={requester_rank}; pieces={len(pieces_to_send)}"
            )

            agent = self.nixl_manager.agent
            remote_xfer_desc = pickle.loads(remote_xfer_desc_bytes)
            logger.debug(f"[NIXL][SEND] rank={self.rank} remote_xfer_desc deserialized")

            # Select source cache: based on piece.component (defaults to Transformer if not present)
            def _select_cache_for_pieces(pieces_to_send):
                comp = None
                if pieces_to_send and hasattr(pieces_to_send[0], "component"):
                    comp = getattr(pieces_to_send[0], "component", None)
                if comp is None:
                    return getattr(self.model.transformer, "cache", None)

                if str(comp).lower().startswith("text"):
                    # Select the first available text encoder cache
                    for te_name in ["text_encoder_3", "text_encoder_2", "text_encoder"]:
                        if hasattr(self.model, te_name):
                            enc = getattr(self.model, te_name)
                            if hasattr(enc, "encoder") and hasattr(
                                enc.encoder, "cache"
                            ):
                                return enc.encoder.cache
                    return None
                else:
                    return getattr(self.model.transformer, "cache", None)

            cache = _select_cache_for_pieces(pieces_to_send)
            if cache is None:
                raise RuntimeError(
                    f"Rank {self.rank}: No available source cache for sending."
                )

            # Receiver's partial metadata fingerprint
            pm_len = (
                len(remote_partial_metadata)
                if hasattr(remote_partial_metadata, "__len__")
                else -1
            )
            pm_head = (
                bytes(remote_partial_metadata)[:64]
                if hasattr(remote_partial_metadata, "__iter__")
                else b""
            )
            logger.debug(
                f"[NIXL][SEND][PM] rank={self.rank} requester={requester_rank} partial_md_len={pm_len} partial_md_head_hash={hash(pm_head)}"
            )

            # List of blocks involved in the send operation
            blocks_to_touch = sorted({p.block_index for p in pieces_to_send})

            if not hasattr(self.nixl_manager, "_xfer_lock"):
                self.nixl_manager._xfer_lock = asyncio.Lock()

            async with self.nixl_manager._xfer_lock:
                # 1) Pre-flight refresh registration: ensure all touched blocks are correctly registered with consistent pointers
                for bi in blocks_to_touch:
                    if bi not in cache.cache:
                        print(
                            f"[NIXL][SEND][CHK] rank={self.rank} block {bi} NOT in cache -> cannot send"
                        )
                        raise RuntimeError(
                            f"Data for block {bi} evicted before sending."
                        )
                    mod, _ = cache.cache[bi]
                    cur = [p for p in mod.parameters() if p.numel() > 0]
                    need_rereg = False
                    if bi not in self.nixl_manager.registered_blocks:
                        need_rereg = True
                        reason = "not registered yet"
                    else:
                        old = self.nixl_manager.registered_blocks[bi]
                        if len(old) != len(cur) or any(
                            ot.data_ptr() != ct.data_ptr() for ot, ct in zip(old, cur)
                        ):
                            need_rereg = True
                            reason = "count or data_ptr changed"
                        else:
                            reason = "ok"
                    logger.debug(
                        f"[NIXL][SEND][CHK] rank={self.rank} block {bi} preflight reason={reason}"
                    )
                    if need_rereg:
                        if bi in self.nixl_manager.registered_blocks:
                            self.nixl_manager.deregister_block(bi)
                        self.nixl_manager.register_block(bi, mod)
                        print(
                            f"[NIXL][SEND][PRE] rank={self.rank} refreshed registration for block {bi}"
                        )

                # 2) Build local source segment descriptors
                src_descriptors: List[tuple] = []
                seg_info: List[tuple] = []

                # 1) Resolve
                def _resolve_module_by_name(
                    root_mod: nn.Module, layer_path: str, split_dim: int, my_len: int
                ) -> nn.Linear:
                    named_lin = {
                        n: m
                        for n, m in root_mod.named_modules()
                        if isinstance(m, nn.Linear)
                    }

                    # Direct hit on full path
                    if layer_path in named_lin:
                        return named_lin[layer_path]

                    # Suffix hit
                    suffix = [
                        (n, m) for n, m in named_lin.items() if n.endswith(layer_path)
                    ]
                    if len(suffix) == 1:
                        return suffix[0][1]

                    # Synonyms and hints
                    parts = layer_path.split(".")
                    last = parts[-1].lower()

                    def syn_last(tok):
                        if tok in ("q", "query", "q_proj"):
                            return ["q", "to_q", "q_proj", "query", "q_linear"]
                        if tok in ("k", "key", "k_proj"):
                            return ["k", "to_k", "k_proj", "key", "k_linear"]
                        if tok in ("v", "value", "v_proj"):
                            return ["v", "to_v", "v_proj", "value", "v_linear"]
                        if tok in ("o", "out", "proj", "out_proj"):
                            return ["to_out", "out_proj", "proj", "o", "to_o", "out"]
                        return [tok]

                    last_syns = syn_last(last)
                    hint_tokens = []
                    lp = layer_path.lower()
                    if "selfattention" in lp or "self_attention" in lp:
                        hint_tokens += ["attn", "attn1", "self", "attention"]
                    if "cross" in lp or "joint" in lp:
                        hint_tokens += ["attn2", "cross", "joint", "attention"]

                    # Suffix synonyms + hint tokens + dimension matching
                    scored = []
                    for name, mod in named_lin.items():
                        score = 0
                        lname = name.lower()
                        if any(
                            lname.endswith(s) or lname.split(".")[-1] == s
                            for s in last_syns
                        ):
                            score += 3
                        if any(s in lname for s in last_syns):
                            score += 1
                        if any(t in lname for t in hint_tokens):
                            score += 1
                        dim_ok = (
                            (mod.weight.size(0) == my_len)
                            if split_dim == 0
                            else (mod.weight.size(1) == my_len)
                        )
                        if dim_ok:
                            score += 3
                        if score > 0:
                            scored.append((score, len(name), name, mod))
                    if scored:
                        scored.sort(
                            key=lambda x: (-x[0], x[1])
                        )  # Prioritize high score, then shorter path
                        return scored[0][3]

                    # Fallback to dimension matching only
                    dim_only = [
                        (name, m)
                        for name, m in named_lin.items()
                        if (
                            m.weight.size(0) == my_len
                            if split_dim == 0
                            else m.weight.size(1) == my_len
                        )
                    ]
                    if len(dim_only) == 1:
                        return dim_only[0][1]

                    # List visible candidates
                    cand_names = ", ".join(list(named_lin.keys())[:10])
                    raise AttributeError(
                        f"{root_mod.__class__.__name__} has no submodule path '{layer_path}'. "
                        f"Linear candidates(sample): {cand_names}"
                    )

                for piece in pieces_to_send:
                    block_index = piece.block_index
                    if block_index not in cache.cache:
                        print(
                            f"[NIXL][SEND] rank={self.rank} ERROR: request block {block_index} not in cache"
                        )
                        raise RuntimeError(
                            f"Data for block {block_index} evicted before sending."
                        )
                    gpu_block, tp_range = cache.cache[block_index]

                    gpu_named_modules = None  # May not need to be persistent
                    source_module = _resolve_module_by_name(
                        gpu_block, piece.layer_path, piece.split_dim, piece.full_size
                    )

                    my_slice_start, my_slice_end = cache.range_to_indices(
                        tp_range, piece.full_size
                    )
                    my_len = max(0, my_slice_end - my_slice_start)
                    relative_start = piece.abs_start_idx - my_slice_start
                    span = piece.abs_end_idx - piece.abs_start_idx
                    if span <= 0:
                        print(
                            f"[NIXL][SEND] rank={self.rank} skip zero-length piece uid={piece.uid}"
                        )
                        continue

                    base_addr = (
                        source_module.weight.data_ptr()
                        if piece.tensor_type == "weight"
                        else source_module.bias.data_ptr()
                    )
                    elem_size = (
                        source_module.weight.element_size()
                        if piece.tensor_type == "weight"
                        else source_module.bias.element_size()
                    )
                    dev_id = (
                        source_module.weight.device.index
                        if piece.tensor_type == "weight"
                        else source_module.bias.device.index
                    )

                    if piece.tensor_type == "bias":
                        start_addr = base_addr + relative_start * elem_size
                        length_bytes = span * elem_size
                        src_descriptors.append((start_addr, length_bytes, dev_id))
                        seg_info.append((piece, start_addr, length_bytes, dev_id))
                    else:
                        row_stride_bytes = source_module.weight.stride(0) * elem_size
                        if piece.split_dim == 0:
                            start_addr = base_addr + relative_start * row_stride_bytes
                            length_bytes = span * row_stride_bytes
                            src_descriptors.append((start_addr, length_bytes, dev_id))
                            seg_info.append((piece, start_addr, length_bytes, dev_id))
                        else:
                            num_rows = source_module.weight.shape[0]
                            for r in range(num_rows):
                                start_addr = (
                                    base_addr
                                    + r * row_stride_bytes
                                    + relative_start * elem_size
                                )
                                length_bytes = span * elem_size
                                src_descriptors.append(
                                    (start_addr, length_bytes, dev_id)
                                )
                                seg_info.append(
                                    (piece, start_addr, length_bytes, dev_id)
                                )

                if not src_descriptors:
                    logger.debug(
                        f"[NIXL][SEND] rank={self.rank} src_descriptors empty; skip"
                    )
                    return True

                bytes_sum_src = sum(int(x[1]) for x in src_descriptors)
                dev_ids = sorted(set(int(x[2]) for x in src_descriptors))
                logger.debug(
                    f"[NIXL][SEND][DESC] rank={self.rank} to={requester_rank} src_desc_cnt={len(src_descriptors)} bytes_sum={bytes_sum_src} dev_ids={dev_ids} first={src_descriptors[:1]} last={src_descriptors[-1:]}"
                )

                # 3) Coverage check
                reg_ranges_by_dev: Dict[int, List[tuple]] = {}
                for bi in blocks_to_touch:
                    for t in self.nixl_manager.registered_blocks.get(bi, []):
                        dev = t.device.index
                        base = t.data_ptr()
                        size = t.numel() * t.element_size()
                        reg_ranges_by_dev.setdefault(dev, []).append(
                            (base, base + size)
                        )

                def covered(dev: int, s: int, e: int):
                    for lb, ub in reg_ranges_by_dev.get(dev, []):
                        if s >= lb and e <= ub:
                            return True
                    return False

                bad_segments = []
                for piece, saddr, lbytes, dev in seg_info:
                    if lbytes <= 0:
                        continue
                    if not covered(dev, saddr, saddr + lbytes):
                        bad_segments.append((piece, saddr, lbytes, dev))
                if bad_segments:
                    logger.debug(
                        f"[NIXL][SEND][COVER] rank={self.rank} BAD segments not covered by registration: cnt={len(bad_segments)}"
                    )
                    for b in bad_segments[:5]:
                        p, s, l, d = b
                        print(
                            f"[NIXL][SEND][COVER] rank={self.rank} sample uid={p.uid} blk={p.block_index} path={p.layer_path} type={p.tensor_type} split_dim={p.split_dim} s=0x{int(s):x} len={l} dev={d}"
                        )
                    raise RuntimeError(
                        "Local src segment not covered by NIXL registration"
                    )

                # Consistent with receiver: local dlist uses is_sorted=True
                local_dlist = agent.get_xfer_descs(
                    src_descriptors, mem_type="cuda", is_sorted=True
                )
                logger.debug(
                    f"[NIXL][SEND] rank={self.rank} local_dlist prepared; desc_cnt={len(src_descriptors)}"
                )

                # 4) Handshake consistent with single-source: always use partial_md to establish/get remote name first; don't fail early on check failure
                max_try = 5

                pm_head_sha = (
                    hashlib.sha256(bytes(remote_partial_metadata)[:256]).hexdigest()
                    if hasattr(remote_partial_metadata, "__iter__")
                    else "NA"
                )
                logger.debug(
                    f"[NIXL][SEND][PEER] rank={self.rank} pm_head_sha256={pm_head_sha} desc_cnt={len(src_descriptors)} bytes_sum={bytes_sum_src}"
                )

                # Explicitly remove old remote metadata to avoid conflicts with new partial metadata
                remote_agent_name_to_remove = f"agent_{requester_rank}"
                try:
                    # Regardless of the return value of add_remote_agent, we attempt to remove by the conventional name
                    agent.remove_remote_agent(remote_agent_name_to_remove)
                    logger.debug(
                        f"[NIXL][SEND] rank={self.rank} explicitly removed remote agent {remote_agent_name_to_remove} before adding partial MD"
                    )
                except Exception:
                    # If the peer does not exist, an exception will be thrown here, which can be safely ignored
                    pass

                remote_agent_name = agent.add_remote_agent(remote_partial_metadata)
                if isinstance(remote_agent_name, (bytes, bytearray)):
                    remote_agent_name = remote_agent_name.decode(
                        "utf-8", errors="ignore"
                    )
                ok = False
                try:
                    ok = agent.check_remote_metadata(
                        remote_agent_name, remote_xfer_desc
                    )
                    logger.debug(
                        f"[NIXL][SEND] rank={self.rank} add_remote->{remote_agent_name} check_remote_metadata(desc)={ok}"
                    )
                except Exception as e:
                    print(
                        f"[NIXL][SEND] rank={self.rank} check_remote_metadata exception: {e}"
                    )
                try:
                    agent.make_connection(remote_agent_name)
                    logger.debug(
                        f"[NIXL][SEND] rank={self.rank} make_connection({remote_agent_name}) OK"
                    )
                except Exception as ce:
                    print(f"[NIXL][SEND] rank={self.rank} make_connection error: {ce}")
                # Don't fail directly because of ok=False, continue to retry initialize_xfer

                # 5) initialize_xfer: on failure, force re-registration, "rebuild local_dlist", and "redo add_remote", then retry
                def _force_reregister_blocks():
                    for bi in blocks_to_touch:
                        mod, _ = cache.cache[bi]
                        try:
                            if bi in self.nixl_manager.registered_blocks:
                                self.nixl_manager.deregister_block(bi)
                            self.nixl_manager.register_block(bi, mod)
                            logger.debug(
                                f"[NIXL][SEND][RR] rank={self.rank} force re-register block {bi}"
                            )
                        except Exception as e:
                            logger.debug(
                                f"[NIXL][SEND][RR] rank={self.rank} block {bi} re-register failed: {e}"
                            )

                tried_rr = False
                for attempt in range(1, max_try + 1):
                    try:
                        xfer_handle = agent.initialize_xfer(
                            "WRITE",
                            local_dlist,
                            remote_xfer_desc,
                            remote_agent_name,
                            backends=["UCX"],
                        )
                        logger.debug(
                            f"[NIXL][SEND] rank={self.rank} initialize_xfer OK on try={attempt}; handle={xfer_handle}"
                        )
                        break
                    except Exception as e:
                        print(
                            f"[NIXL][SEND][ERR] rank={self.rank} initialize_xfer error: {e} try={attempt}"
                        )
                        if not tried_rr:
                            _force_reregister_blocks()
                            # Rebuild dlist
                            local_dlist = agent.get_xfer_descs(
                                src_descriptors, mem_type="cuda", is_sorted=True
                            )
                            logger.debug(
                                f"[NIXL][SEND] rank={self.rank} rebuilt local_dlist after re-register; desc_cnt={len(src_descriptors)}"
                            )
                            # Re-declare remote
                            remote_agent_name = agent.add_remote_agent(
                                remote_partial_metadata
                            )
                            if isinstance(remote_agent_name, (bytes, bytearray)):
                                remote_agent_name = remote_agent_name.decode(
                                    "utf-8", errors="ignore"
                                )
                            try:
                                agent.make_connection(remote_agent_name)
                                logger.debug(
                                    f"[NIXL][SEND] rank={self.rank} re-make_connection({remote_agent_name}) OK"
                                )
                            except Exception as ce:
                                print(
                                    f"[NIXL][SEND] rank={self.rank} make_connection error after reregister: {ce}"
                                )
                            tried_rr = True
                        await asyncio.sleep(0.05 * attempt)
                else:
                    dbg = []
                    for bi in blocks_to_touch:
                        reg = bi in self.nixl_manager.registered_blocks
                        ptrs = (
                            [
                                hex(p.data_ptr())
                                for p in self.nixl_manager.registered_blocks.get(
                                    bi, []
                                )[:2]
                            ]
                            if reg
                            else []
                        )
                        dbg.append((bi, reg, ptrs))
                    logger.debug(
                        f"[NIXL][SEND][DBG] rank={self.rank} blocks_reg_summary={dbg}"
                    )
                    raise RuntimeError(
                        f"Rank {self.rank}: initialize_xfer failed after {max_try} retries"
                    )

                # 6) Execute transfer
                state = agent.transfer(xfer_handle)
                while state != "DONE":
                    if state == "ERR":
                        logger.debug(
                            f"[NIXL][SEND] rank={self.rank} postXferReq -> ERR"
                        )
                        raise RuntimeError(
                            f"NIXL zero-copy transfer from Rank {self.rank} failed."
                        )
                    await asyncio.sleep(0.001)
                    state = agent.check_xfer_state(xfer_handle)

                try:
                    backend = agent.query_xfer_backend(xfer_handle)
                    logger.debug(
                        f"[NIXL][SEND] rank={self.rank} transfer DONE; backend={backend}"
                    )
                except Exception:
                    pass

                agent.release_xfer_handle(xfer_handle)
                logger.debug(
                    f"[NIXL][SEND] rank={self.rank} SUCCESS to remote {remote_agent_name} (requester_rank={requester_rank})"
                )
                return True

        except Exception as e:
            print(f"[NIXL][SEND] rank={self.rank} ERROR in rpc_nixl_send_data: {e}")
            raise e

    async def create_nixl_manager(self) -> bool:
        """Create NIXL agent serially (called by the engine for each rank one by one)."""
        if getattr(self, "nixl_manager", None) is not None:
            return True
        if self.engine_config.runtime_config.adjust_strategy != "p2p":
            return True
        torch.cuda.synchronize()
        self.nixl_manager = NixlP2PManager(self.rank, self.all_worker_handles)
        logger.info(
            f"Rank {self.rank}: NIXL agent created successfully (UCX_TLS={getattr(self.nixl_manager, 'selected_tls', None)})"
        )
        return True

    async def register_existing_cache_with_nixl(self) -> bool:
        """Register existing blocks in the current cache with NIXL (called after agent creation)."""
        if self.engine_config.runtime_config.adjust_strategy != "p2p":
            return True
        if not hasattr(self, "nixl_manager") or self.nixl_manager is None:
            raise RuntimeError(f"Rank {self.rank}: NIXL manager has not been created")
        # Register Transformer cache
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "cache"
        ):
            for idx, (mod, _) in self.model.transformer.cache.cache.items():
                if idx not in self.nixl_manager.registered_blocks:
                    self.nixl_manager.register_block(idx, mod)
        # Register TextEncoder cache (if any)
        for te_name in ["text_encoder_3", "text_encoder_2", "text_encoder"]:
            if hasattr(self.model, te_name):
                enc = getattr(self.model, te_name)
                if hasattr(enc, "encoder") and hasattr(enc.encoder, "cache"):
                    for idx, (mod, _) in enc.encoder.cache.cache.items():
                        if idx not in self.nixl_manager.registered_blocks:
                            self.nixl_manager.register_block(idx, mod)
        logger.info(
            f"Rank {self.rank}: All existing cache blocks have been registered with NIXL"
        )
        return True

    async def nixl_preflight_get_md(self):
        """Return the full NIXL metadata of this node (for peer loading)."""
        try:
            md = self.nixl_manager.agent.get_agent_metadata()
            logger.debug(f"[NIXL][PREFLIGHT] rank={self.rank} get_agent_metadata OK")
            return (self.rank, md)
        except Exception as e:
            print(f"[NIXL][PREFLIGHT] rank={self.rank} get_agent_metadata error: {e}")
            raise

    async def nixl_preflight_connect(self, all_metadata_list):
        """
        Use full metadata to perform add_remote_agent + make_connection for each peer.
        Called once before each adjustment to reduce initialization/connection race conditions.
        """
        try:
            for r, meta in all_metadata_list:
                if r == self.rank:
                    continue
                try:
                    remote_name = self.nixl_manager.agent.add_remote_agent(meta)
                    if isinstance(remote_name, (bytes, bytearray)):
                        remote_name = remote_name.decode("utf-8", errors="ignore")
                    self.nixl_manager.agent.make_connection(remote_name)
                    logger.debug(
                        f"[NIXL][PREFLIGHT] rank={self.rank} connected {remote_name} (peer={r})"
                    )
                except Exception as e:
                    print(
                        f"[NIXL][PREFLIGHT] rank={self.rank} connect peer={r} error: {e}"
                    )
            return True
        except Exception as e:
            print(f"[NIXL][PREFLIGHT] rank={self.rank} preflight_connect fatal: {e}")
            raise

    def _refresh_nixl_for_cache(self, cache_obj):
        """
        Iterate through all blocks in the cache. If a block is not registered or its data_ptr has changed,
        deregister and then re-register it.
        Print the reason for change and a before/after pointer comparison for each block.
        """
        if cache_obj is None or not hasattr(cache_obj, "cache"):
            return
        for idx, (mod, _) in cache_obj.cache.items():
            cur_tensors = [p for p in mod.parameters() if p.numel() > 0]
            need_reregister = False
            reason = ""

            if idx not in self.nixl_manager.registered_blocks:
                need_reregister = True
                reason = "not registered yet"
            else:
                old_tensors = self.nixl_manager.registered_blocks[idx]
                if len(old_tensors) != len(cur_tensors):
                    need_reregister = True
                    reason = (
                        f"tensor count changed {len(old_tensors)}->{len(cur_tensors)}"
                    )
                else:
                    for ot, ct in zip(old_tensors, cur_tensors):
                        if ot.data_ptr() != ct.data_ptr():
                            need_reregister = True
                            reason = "data_ptr changed"
                            break

            logger.debug(
                f"[NIXL][REFRESH] rank={self.rank} checking block {idx} need_reregister={need_reregister} reason={reason}"
            )
            if need_reregister:
                try:
                    if idx in self.nixl_manager.registered_blocks:
                        logger.debug(
                            f"[NIXL][REFRESH] rank={self.rank} BEFORE reg: old_ptrs={[hex(p.data_ptr()) for p in self.nixl_manager.registered_blocks[idx][:2]]}"
                        )
                        self.nixl_manager.deregister_block(idx)
                    self.nixl_manager.register_block(idx, mod)
                    logger.debug(
                        f"[NIXL][REFRESH] rank={self.rank} AFTER reg: new_ptrs={[hex(p.data_ptr()) for p in cur_tensors[:2]]}"
                    )
                except Exception as e:
                    print(
                        f"[NIXL][REFRESH] rank={self.rank} block {idx} refresh failed: {e}"
                    )
                    raise

    async def refresh_nixl_registrations(self) -> bool:
        """
        Called after a hot adjustment is complete: uniformly refreshes NIXL registrations for Transformer
        and TextEncoder to ensure sender-side registrations are consistent with current GPU tensor addresses.
        """
        if self.engine_config.runtime_config.adjust_strategy != "p2p":
            return True
        if not hasattr(self, "nixl_manager") or self.nixl_manager is None:
            return True

        # Transformer
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "cache"
        ):
            self._refresh_nixl_for_cache(self.model.transformer.cache)

        # Text Encoders
        for te_name in ["text_encoder_3", "text_encoder_2", "text_encoder"]:
            if hasattr(self.model, te_name):
                enc = getattr(self.model, te_name)
                if hasattr(enc, "encoder") and hasattr(enc.encoder, "cache"):
                    self._refresh_nixl_for_cache(enc.encoder.cache)

        logger.debug(
            f"[NIXL][REFRESH] rank={self.rank} total_registered_blocks={len(self.nixl_manager.registered_blocks)} sample={(list(self.nixl_manager.registered_blocks.keys())[:5])}"
        )
        return True


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half."
            )
