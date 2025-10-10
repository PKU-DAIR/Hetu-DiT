import asyncio
import copy
import ipaddress
import os
import socket
import subprocess
import warnings
from functools import partial
from typing import Awaitable, Callable, List, Optional, Tuple, TypeVar, Union

from packaging.version import Version, parse
import torch
from torch import nn

from diffusers import (
    CogVideoXPipeline,
    FluxPipeline,
    HunyuanVideoPipeline,
    StableDiffusion3Pipeline,
)

from hetu_dit.config.config import (
    DataParallelConfig,
    EngineConfig,
    InputConfig,
    ParallelConfig,
    PipeFusionParallelConfig,
    SequenceParallelConfig,
    TensorParallelConfig,
)
from hetu_dit.logger import init_logger

T = TypeVar("T")
logger = init_logger(__name__)

_IPV4_TEST_ENDPOINT = ("8.8.8.8", 80)
_IPV6_TEST_ENDPOINT = ("2001:4860:4860::8888", 80)


def _resolve_ip_through_socket(
    address_family: int, endpoint: Tuple[str, int]
) -> Optional[str]:
    """Attempt to determine the host IP by opening a UDP socket."""
    try:
        with socket.socket(address_family, socket.SOCK_DGRAM) as sock:
            sock.connect(endpoint)
            return sock.getsockname()[0]
    except OSError:
        return None


def get_ip():
    host_ip = os.environ.get("HOST_IP")
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


# def get_distributed_init_method(ip: str, port: int) -> str:
#     return f"tcp://{ip}:{port}"


def get_distributed_init_method(ip: str, port: int) -> str:
    try:
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.version == 6:
            return f"tcp://[{ip}]:{port}"
        else:
            return f"tcp://{ip}:{port}"
    except ValueError:
        raise ValueError(f"Invalid IP address: {ip}")


def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def set_cuda_visible_devices(device_ids: List[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def get_nvcc_cuda_version() -> Optional[Version]:
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        cuda_home = "/usr/local/cuda"
        if os.path.isfile(cuda_home + "/bin/nvcc"):
            logger.info(
                f"CUDA_HOME is not found in the environment. "
                f"Using {cuda_home} as CUDA_HOME."
            )
        else:
            logger.warning(f"Not found nvcc in {cuda_home}. Skip cuda version check!")
            return None
    nvcc_output = subprocess.check_output(
        [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


######################################################################################
# ================ use for scheduler==========================
######################################################################################

HARD_ENCODED_RUNNING_TIME = {
    CogVideoXPipeline: {
        "768-1360-81": {1: 30.5, 2: 17.75, 4: 9.1, 8: 4.75},  # 81600
        "768-1360-33": {1: 9.37, 2: 5.63, 4: 3, 8: 1.57},  # 32640
        "768-1360-65": {1: 22.14, 2: 13.07, 4: 6.77, 8: 3.56},  # 65280
        "768-432-81": {1: 5.72, 2: 3.51, 4: 1.89, 8: 1.01},  # 25920
        "768-432-65": {1: 4.42, 2: 2.76, 4: 1.49, 8: 0.826446281},  # 20736
        "768-432-33": {
            1: 2.16,
            2: 0.751879699,
            4: 0.751879699,
            8: 0.452488688,
        },  # 10368
        "768-432-17": {1: 1.19, 2: 0.763358779, 4: 0.45045045, 8: 0.3},  # 5184
        "768-432-1": {
            1: 0.384615385,
            2: 0.262467192,
            4: 0.225733634,
            8: 0.158730159,
        },  # 1296
        "768-1360-1": {1: 1.26, 2: 0.78125, 4: 0.483091787, 8: 0.297619048},  # 4080
        "768-1360-17": {1: 4.71, 2: 2.91, 4: 1.55, 8: 0.840336134},  # 16320
    },
    HunyuanVideoPipeline: {
        "720-1280-129": {1: 118, 2: 64.48, 4: 31.92, 8: 16.35},  # 115200
        "720-1280-65": {1: 37.53, 2: 21, 4: 10.7, 8: 5.8},  # 57600
        "720-1280-33": {1: 13.7, 2: 8.01, 4: 4.1, 8: 2.35},  # 28800
        "720-1280-17": {1: 5.84, 2: 3.61, 4: 1.86, 8: 1.08},  # 14400
        "720-1280-1": {1: 0.787401575, 2: 0.68, 4: 0.48, 8: 0.38},  # 3600
        "960-544-129": {1: 43.58, 2: 24.6, 4: 12.4, 8: 6.67},  # 65280
        "960-544-65": {1: 15.09, 2: 8.76, 4: 4.51, 8: 2.3},  # 32640
        "960-544-33": {1: 6.01, 2: 3.68, 4: 2.03, 8: 1},  # 16320
        "960-544-17": {1: 2.77, 2: 1.83, 4: 1.02, 8: 0.68},  # 8160
        "960-544-1": {1: 0.54, 2: 0.46, 4: 0.38, 8: 0.3},  # 2040
    },
    FluxPipeline: {
        "128-128-1": {
            1: 0.054288817,
            2: 0.055005501,
            4: 0.050556117,
            8: 0.046296296,
        },  # 64
        "256-256-1": {
            1: 0.108577633,
            2: 0.110011001,
            4: 0.101112235,
            8: 0.092592593,
        },  # 256
        "512-512-1": {
            1: 0.22172949,
            2: 0.194931774,
            4: 0.136612022,
            8: 0.121212121,
        },  # 1024
        "1024-1024-1": {
            1: 0.840336134,
            2: 0.537634409,
            4: 0.323624595,
            8: 0.227272727,
        },  # 4096
        "2048-2048-1": {1: 4.67, 2: 2.85, 4: 1.48, 8: 1.28},  # 16384
        "3072-3072-1": {1: 15.5, 2: 9, 4: 4.55, 8: 2.46},  # 36864
        "4096-4096-1": {1: 39.83, 2: 22.97, 4: 11.76, 8: 5.66},  # 65536
    },
    StableDiffusion3Pipeline: {
        "128-128-1": {
            1: 0.022045855,
            2: 0.022045855,
            4: 0.022045855,
            8: 0.022045855,
        },  # 64
        "256-256-1": {
            1: 0.029664788,
            2: 0.029664788,
            4: 0.029664788,
            8: 0.029664788,
        },  # 256
        "512-512-1": {
            1: 0.059171598,
            2: 0.063011972,
            4: 0.063011972,
            8: 0.063011972,
        },  # 1024
        "1024-1024-1": {
            1: 0.242718447,
            2: 0.178571429,
            4: 0.166666667,
            8: 0.153846154,
        },  # 4096
        "1536-1536-1": {
            1: 0.595238095,
            2: 0.367430923,
            4: 0.297619048,
            8: 0.17303433,
        },  # 9216
    },
}


def make_profile_key(input_cfg) -> str:
    """
    Generate a profile key from InputConfig.
    Convention: use fields height, width, num_frames.

    Example
    -------
    In : input_cfg.height=768, input_cfg.width=432, input_cfg.num_frames=33
    Out: "768-432-33"
    """
    return f"{input_cfg.height}-{input_cfg.width}-{input_cfg.num_frames}"


def estimate_ddl(t_dict: dict[int, float]) -> int:
    """
    Given {k: tk}, find the maximum parallelism k★ with efficiency ≥0.8,
    then return ddl = t_{k★} * 2 (rounded up to the nearest second).

    Parameters
    ----------
    t_dict : dict[int, float]
        Dictionary mapping parallelism degree k to time tk. Example: {1: t1, 2: t2, 4: t4, 8: t8}

    Returns
    -------
    ddl_sec : int
        Deadline in seconds relative to now.
    """
    t1 = t_dict[1]
    best_k = 1
    for k, tk in t_dict.items():
        if (t1 / tk) / k >= 0.8 and k > best_k:
            best_k = k
    ddl = int(round(t_dict[best_k] * 4) * 1.1) + 2
    return ddl


def make_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args, **kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)

    return _async_wrapper


######################################################################################
# ================ use for create config==========================
######################################################################################
def create_new_config(
    old_engine_config: EngineConfig,
    data_parallel_degree: int = 1,
    use_cfg_parallel: bool = False,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_degree: int = 1,
    pipefusion_parallel_degree: int = 1,
    num_pipeline_patch=None,
    text_encoder_tensor_parallel_degree: int = 1,
    use_parallel_text_encoder: bool = False,
    attn_layer_num_for_pp=None,
    height: int = 1024,
    width: int = 1024,
    num_frames: int = 49,
    no_use_resolution_binning: bool = (True,),
    prompt: Union[str, List[str]] = "",
    negative_prompt: Union[str, List[str]] = "",
    num_inference_steps: int = 20,
    max_sequence_length: int = 256,
    seed: int = 42,
    output_type: str = "pil",
    is_serving: bool = True,  # add to test serving code, can remove dan refine code
    task_id: str = "",
    machine_id: int = 0,
    encode_stage_rank: Optional[int] = None,
    decode_stage_ranks: Optional[List[int]] = None,
) -> Tuple[EngineConfig, InputConfig]:
    model_config = copy.deepcopy(old_engine_config.model_config)

    runtime_config = copy.deepcopy(old_engine_config.runtime_config)

    if is_serving and pipefusion_parallel_degree > 1:
        logger.warning(
            "work around, need to fix: to avoid pipefusion oom, set pp warmup tp num_inference_steps"
        )
        runtime_config.warmup_steps = num_inference_steps

    if use_parallel_text_encoder:
        parallel_config = ParallelConfig(
            dp_config=DataParallelConfig(
                dp_degree=data_parallel_degree,
                use_cfg_parallel=use_cfg_parallel,
                is_serving=is_serving,
            ),
            sp_config=SequenceParallelConfig(
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                is_serving=is_serving,
            ),
            tp_config=TensorParallelConfig(
                tp_degree=tensor_parallel_degree,
                split_scheme=old_engine_config.parallel_config.tp_config.split_scheme,
                is_serving=is_serving,
            ),
            pp_config=PipeFusionParallelConfig(
                pp_degree=pipefusion_parallel_degree,
                num_pipeline_patch=num_pipeline_patch,
                attn_layer_num_for_pp=attn_layer_num_for_pp,
                is_serving=is_serving,
            ),
            text_encoder_tp_config=TensorParallelConfig(
                tp_degree=text_encoder_tensor_parallel_degree,
                split_scheme=old_engine_config.parallel_config.tp_config.split_scheme,
                is_serving=is_serving,
            ),
            is_serving=is_serving,
        )
    else:
        parallel_config = ParallelConfig(
            dp_config=DataParallelConfig(
                dp_degree=data_parallel_degree,
                use_cfg_parallel=use_cfg_parallel,
                is_serving=is_serving,
            ),
            sp_config=SequenceParallelConfig(
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                is_serving=is_serving,
            ),
            tp_config=TensorParallelConfig(
                tp_degree=tensor_parallel_degree,
                split_scheme=old_engine_config.parallel_config.tp_config.split_scheme,
                is_serving=is_serving,
            ),
            pp_config=PipeFusionParallelConfig(
                pp_degree=pipefusion_parallel_degree,
                num_pipeline_patch=num_pipeline_patch,
                attn_layer_num_for_pp=attn_layer_num_for_pp,
                is_serving=is_serving,
            ),
            is_serving=is_serving,
        )

    engine_config = EngineConfig(
        model_config=model_config,
        runtime_config=runtime_config,
        parallel_config=parallel_config,
        machine_num=old_engine_config.machine_num,
        machine_id=machine_id,
        encode_stage_rank=encode_stage_rank,
        decode_stage_ranks=decode_stage_ranks,
    )

    input_config = InputConfig(
        height=height,
        width=width,
        num_frames=num_frames,
        use_resolution_binning=not no_use_resolution_binning,
        batch_size=len(prompt) if isinstance(prompt, list) else 1,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        seed=seed,
        output_type=output_type,
        task_id=task_id,
    )

    return engine_config, input_config


######################################################################################
# ================ use for parallel parameter convert==========================
######################################################################################


def range_to_indices(rng, total_size):
    start_prop, end_prop = rng
    start_idx = round(start_prop * total_size)
    end_idx = round(end_prop * total_size)
    return start_idx, end_idx


def copy_cpu_block_to_gpu(cpu_block):
    gpu_block = copy.deepcopy(cpu_block).to("cuda")
    return gpu_block


def determine_tp_split_type_by_name(name: str) -> str:
    # Based on the name, determine the row/column splitting logic:
    # Inside Attention:
    # to_q, to_k, to_v, add_k_proj, add_q_proj, add_v_proj: row split
    # to_out, to_add_out: column split

    # Inside FeedForward:
    # net.0.proj: row split
    # net.2: column split
    lower_name = name.lower()
    if any(
        x in lower_name
        for x in ["to_q", "to_k", "to_v", "add_k_proj", "add_q_proj", "add_v_proj"]
    ):
        return "row"
    if "to_out.0" in lower_name or "to_add_out" in lower_name:
        return "column"
    if ".0.proj" in lower_name:
        return "row"
    if ".2" in lower_name:
        return "column"
    return "row"  # default row


def adjust_linear_tensor_parallel(
    linear_module: nn.Linear,
    cpu_module: nn.Linear,
    old_tp_range: tuple,
    new_tp_range: tuple,
    tp_split_type: str = "row",
    slice_bias: bool = True,
):
    if tp_split_type == "row":
        tensor_param_total_size = cpu_module.weight.size(0)
        tensor_split_dimension = 0
    else:
        tensor_param_total_size = cpu_module.weight.size(1)
        tensor_split_dimension = 1

    old_tp_start, old_tp_end = range_to_indices(old_tp_range, tensor_param_total_size)
    new_tp_start, new_tp_end = range_to_indices(new_tp_range, tensor_param_total_size)

    weight = linear_module.weight.data
    cpu_weight = cpu_module.weight.data

    out_features, in_features = weight.shape
    needed_length = new_tp_end - new_tp_start

    # According to the split dimension, determine the shape of the result
    if tensor_split_dimension == 0:
        result_shape = (needed_length, in_features)
    else:
        result_shape = (out_features, needed_length)

    result_weight = torch.empty(result_shape, dtype=weight.dtype, device="cuda")

    overlap_start = max(old_tp_start, new_tp_start)
    overlap_end = min(old_tp_end, new_tp_end)
    old_len = old_tp_end - old_tp_start

    def slice_along_dim(param, dim, gstart, gend):
        length = gend - gstart
        return torch.narrow(param, dim, gstart, length)

    cpu_slice_weight = slice_along_dim(
        cpu_weight, tensor_split_dimension, new_tp_start, new_tp_end
    )
    result_weight.copy_(cpu_slice_weight.to("cuda"))
    logger.debug(
        f"cpu_weight.shape = {cpu_weight.shape}, old_tp_start={old_tp_start}, old_tp_end={old_tp_end}, new_tp_start={new_tp_start}, new_tp_end={new_tp_end}, overlap_start={overlap_start}, overlap_end={overlap_end}, out_features={out_features}, in_features={in_features}, needed_length={needed_length}, tensor_split_dimension={tensor_split_dimension}"
    )
    logger.debug(f"result_weight.shape = {result_weight.shape}")

    linear_module.weight = nn.Parameter(result_weight.contiguous())
    logger.debug(f"linear_module.weight.shape = {linear_module.weight.shape}")

    if slice_bias and linear_module.bias is not None:
        bias = linear_module.bias.data
        cpu_bias = cpu_module.bias.data
        result_bias = torch.empty((needed_length,), dtype=bias.dtype, device="cuda")
        result_bias.copy_(cpu_bias[new_tp_start:new_tp_end].to("cuda"))
        linear_module.bias = nn.Parameter(result_bias.contiguous())
    else:
        if not slice_bias:
            pass
        else:
            linear_module.bias = None

    return linear_module


def adjust_linears_in_layers(
    layers: nn.Module,
    cpu_full_transformer_layers: nn.Module,
    new_pp_range: tuple,
    old_tp_range: tuple,
    new_tp_range: tuple,
    slice_bias: bool = True,
):
    # get total_blocks from cpu_full_transformer.transformer_blocks
    total_blocks = len(cpu_full_transformer_layers)

    new_pp_start, new_pp_end = range_to_indices(new_pp_range, total_blocks)
    logger.debug(
        f"total_blocks = {total_blocks},len(layers) = {len(layers)}, new_pp_range = {new_pp_range}, new_pp_start = {new_pp_start}, new_pp_end = {new_pp_end}"
    )
    for i, gpu_block in enumerate(layers):
        logger.debug(f"i = {i}, ")
        cpu_block = cpu_full_transformer_layers[new_pp_start + i]
        gpu_named_modules = dict(gpu_block.named_modules())
        cpu_named_modules = dict(cpu_block.named_modules())
        for name, gpu_submodule in gpu_named_modules.items():
            if isinstance(gpu_submodule, nn.Linear) and name in [
                "attn.module.to_q",
                "attn.module.to_k",
                "attn.module.to_v",
                "attn.module.add_k_proj",
                "attn.module.add_q_proj",
                "attn.module.add_v_proj",
                "attn.module.to_add_out",
                "attn.module.to_out.0",
                "attn1.module.to_q",
                "attn1.module.to_k",
                "attn1.module.to_v",
                "attn1.module.add_k_proj",
                "attn1.module.add_q_proj",
                "attn1.module.add_v_proj",
                "attn1.module.to_add_out",
                "attn1.module.to_out.0",
                "ff.module.net.0.proj",
                "ff.module.net.2",
                "ff_context.module.net.0.proj",
                "ff_context.module.net.2",
            ]:
                cpu_submodule = cpu_named_modules[name]
                tp_split_type = determine_tp_split_type_by_name(name)
                logger.debug(
                    f"before: total_len = {len(layers)}, new_pp_start = {new_pp_start}, new_pp_end = {new_pp_end},  name={name}, tp_split_type={tp_split_type}"
                )
                adjust_linear_tensor_parallel(
                    linear_module=gpu_submodule,
                    cpu_module=cpu_submodule,
                    old_tp_range=old_tp_range,
                    new_tp_range=new_tp_range,
                    tp_split_type=tp_split_type,
                    slice_bias=slice_bias,
                )
            else:
                gpu_submodule.to("cuda")


def free_module_gpu_memory(module: nn.Module):
    """Recursively remove all parameters and buffers to free GPU memory."""
    module.to_empty(device="meta")  # Instantly move all parameters/buffers to meta
    torch.cuda.empty_cache()  # Immediately return allocator cache to CUDA
    torch.cuda.ipc_collect()  # Clear IPC handles to prevent long-tail occupation


def adjust_pipeline(
    transformer: nn.Module,
    cpu_full_transformer: nn.Module,
    old_pp_range: Tuple[float, float],
    new_pp_range: Tuple[float, float],
    old_tp_range: Tuple[float, float],
    new_tp_range: Optional[Tuple[float, float]] = None,
    slice_bias: bool = True,
    transformer_blocks_name: List[str] = ["transformer_blocks"],
    transformer_blocks_num: Optional[List[int]] = None,
):
    """
    Supports multiple types of transformer_blocks (specified by name), and uses transformer_blocks_num to indicate the number of layers for each type.
    Performs merging, deletion, or addition of layers between old_pp_range and new_pp_range. For old/new TP ranges (old_tp_range, new_tp_range),
    finally calls adjust_linears_in_layers for each type to perform TP splitting (using each type's local new_pp_range).

    Parameters:
        transformer: The current submodel on GPU (each blocks attribute only contains layers corresponding to old_pp_range)
        cpu_full_transformer: The complete model on CPU (contains all layers)
        old_pp_range: Old PP range (proportion), e.g. (0.0, 0.5)
        new_pp_range: New PP range (proportion), e.g. (0.25, 0.75)
        old_tp_range: Old TP range (proportion)
        new_tp_range: New TP range (proportion), if not None, TP splitting is performed after merging PP layers
        slice_bias: Whether to also split bias (passed to adjust_linears_in_layers)
        transformer_blocks_name: List of attribute names for each block type, e.g. ["transformer_blocks", "single_transformer_blocks"]
        transformer_blocks_num: Total number of layers for each block type in the CPU model, e.g. [20, 40]
    Returns:
        The input transformer, with the corresponding attributes replaced by new nn.ModuleList
    """

    if old_pp_range == new_pp_range and old_tp_range == new_tp_range:
        return transformer
    # 1. Parameter validation: If multiple block names are declared but no corresponding counts are provided, raise an error
    if transformer_blocks_num is None and len(transformer_blocks_name) > 1:
        raise ValueError(
            "When there are multiple elements in transformer_blocks_name, transformer_blocks_num must be provided simultaneously."
        )
    # If transformer_blocks_num is not explicitly provided, automatically get the number of layers for each attribute from cpu_full_transformer
    if transformer_blocks_num is None:
        transformer_blocks_num = [
            len(getattr(cpu_full_transformer, name)) for name in transformer_blocks_name
        ]

    # 2. Calculate the start and end positions (global index) of each block type in the global layer sequence
    prefix_sums: List[int] = []
    running = 0
    for cnt in transformer_blocks_num:
        prefix_sums.append(running)
        running += cnt
    total_blocks = running  # sum(transformer_blocks_num)

    # 3. Map old_pp_range and new_pp_range (proportion) to global index intervals [old_pp_start, old_pp_end), [new_pp_start, new_pp_end)
    old_pp_start, old_pp_end = range_to_indices(old_pp_range, total_blocks)
    new_pp_start, new_pp_end = range_to_indices(new_pp_range, total_blocks)

    # 4. Iterate over each block type, perform merging logic, and calculate the local new_pp_range
    local_pp_ranges: List[Tuple[float, float]] = []
    for idx, block_name in enumerate(transformer_blocks_name):
        # 4.1 The start and end positions of the current type in the global sequence
        global_start = prefix_sums[idx]
        global_end = prefix_sums[idx] + transformer_blocks_num[idx]

        # 4.2 Calculate the local range of this type in old_pp_range (global -> local)
        type_old_start = max(old_pp_start, global_start)
        type_old_end = min(old_pp_end, global_end)
        local_old_start = type_old_start - global_start
        local_old_end = type_old_end - global_start
        expected_old_count = max(0, local_old_end - local_old_start)

        # 4.3 Get the ModuleList of this block type from transformer (GPU); if not present, treat as empty
        old_gpu_blocks = getattr(transformer, block_name, None)
        if old_gpu_blocks is None:
            if expected_old_count != 0:
                raise ValueError(
                    f"Transformer lacks `{block_name}`, but old_pp_range suppose it has {expected_old_count} layers。"
                )
            old_gpu_blocks = nn.ModuleList()
        else:
            if len(old_gpu_blocks) != expected_old_count:
                raise ValueError(
                    f"Current transformer.{block_name} has {len(old_gpu_blocks)} layers, "
                    f"but old_pp_range suppose it has {expected_old_count} layers。"
                )

        # 4.4 Calculate the local range of this type in new_pp_range
        type_new_start = max(new_pp_start, global_start)
        type_new_end = min(new_pp_end, global_end)
        local_new_start = max(0, type_new_start - global_start)
        local_new_end = max(0, type_new_end - global_start)

        # 4.5 Get the complete ModuleList of this type from the CPU model
        cpu_blocks = getattr(cpu_full_transformer, block_name, None)
        if cpu_blocks is None:
            raise ValueError(
                f"CPU model lacks `{block_name}`, cannot copy layers from CPU."
            )

        # 4.6 For this type — construct the new new_layers_i
        new_layers_i = nn.ModuleList()
        for m in old_gpu_blocks:
            free_module_gpu_memory(m)
        torch.cuda.empty_cache()
        if type_new_start < type_new_end:
            # For this type, new_range overlaps with the global range
            intersect_start = max(type_old_start, type_new_start)
            intersect_end = min(type_old_end, type_new_end)

            if intersect_start < intersect_end:
                # 4.6.1 If there is an overlap, first retain this segment from the old GPU
                local_intersect_start = intersect_start - global_start
                local_intersect_end = intersect_end - global_start

                overlap_blocks = copy.deepcopy(
                    cpu_blocks[local_intersect_start:local_intersect_end]
                )

                new_layers_i.extend(overlap_blocks)

                # 4.6.2 If new_range is missing a front segment, copy the missing_front from CPU
                if local_new_start < local_intersect_start:
                    missing_front_cpu = copy.deepcopy(
                        cpu_blocks[local_new_start:local_intersect_start]
                    )
                    # missing_front_gpu = [copy_cpu_block_to_gpu(m) for m in missing_front_cpu]
                    new_layers_i = nn.ModuleList(missing_front_cpu + list(new_layers_i))

                # 4.6.3 If new_range is missing a back segment, copy the missing_back from CPU
                if local_intersect_end < local_new_end:
                    missing_back_cpu = copy.deepcopy(
                        cpu_blocks[local_intersect_end:local_new_end]
                    )
                    # missing_back_gpu = [copy_cpu_block_to_gpu(m) for m in missing_back_cpu]
                    new_layers_i.extend(missing_back_cpu)
            else:
                # 4.6.4 If old and new ranges have no overlap for this type, copy the entire segment local_new_start:local_new_end from CPU

                needed_cpu = copy.deepcopy(cpu_blocks[local_new_start:local_new_end])
                # needed_gpu = [copy_cpu_block_to_gpu(m) for m in needed_cpu]
                new_layers_i = nn.ModuleList(needed_cpu)
        else:
            # If new_range for this type contains no layers
            new_layers_i = nn.ModuleList()

        # 4.7 Write back to transformer
        setattr(transformer, block_name, new_layers_i)

        # 4.8 Calculate the local new_pp_range for this type
        N_i = transformer_blocks_num[idx]
        if local_new_end > local_new_start:
            local_frac_start = local_new_start / N_i
            local_frac_end = local_new_end / N_i
        else:
            # If new_layers_i for this type is empty, set to (0, 0)
            local_frac_start = 0.0
            local_frac_end = 0.0

        local_pp_ranges.append((local_frac_start, local_frac_end))

    # 5. If TP splitting is needed, call adjust_linears_in_layers for each type, passing the corresponding local new_pp_range
    if new_tp_range is not None:
        for idx, block_name in enumerate(transformer_blocks_name):
            new_blocks = getattr(transformer, block_name)
            cpu_blocks = getattr(cpu_full_transformer, block_name)
            if len(new_blocks) == 0:
                continue

            # The local new_pp_range for this type
            local_pp = local_pp_ranges[idx]

            adjust_linears_in_layers(
                layers=new_blocks,
                cpu_full_transformer_layers=cpu_blocks,
                new_pp_range=local_pp,
                old_tp_range=old_tp_range,
                new_tp_range=new_tp_range,
                slice_bias=slice_bias,
            )

    return transformer


def get_gpu_metadata(
    pipeline_parallel_rank: int,
    pipeline_parallel_world_size: int,
    tensor_model_parallel_rank: int,
    tensor_model_parallel_world_size: int,
):
    """
    Given the pipeline_parallel_rank, pipeline_parallel_world_size, tensor_model_parallel_rank, and tensor_model_parallel_world_size, derive the corresponding gpu_metadata.

    The gpu_metadata return format is:
    {
    "tensor_parallel": (tensor_start, tensor_end),
    "pipeline_parallel": (pipeline_start, pipeline_end)
    }

    Partitioning logic:
        •	pipeline_parallel: Divide [0,1) evenly into pipeline_parallel_world_size parts, and the pipeline_parallel_rank corresponds to one of those parts.
        •	tensor_parallel: Divide [0,1) evenly into tensor_model_parallel_world_size parts, and the tensor_model_parallel_rank corresponds to one of those parts.

    Example:
    If pipeline_parallel_rank = 1, pipeline_parallel_world_size = 2, tensor_model_parallel_rank = 0, tensor_model_parallel_world_size = 2, then the pipeline_parallel partitions are [0, 0.5) and [0.5, 1.0), and rank=1 corresponds to [0.5, 1.0).

    The tensor_parallel partitions are [0, 0.5) and [0.5, 1.0), and rank=0 corresponds to [0, 0.5).

    Thus, gpu_metadata = {“tensor_parallel”: (0, 0.5), “pipeline_parallel”: (0.5, 1)}
    """

    # Calculate the tensor_parallel range
    tensor_chunk_size = 1.0 / tensor_model_parallel_world_size
    tensor_start = tensor_model_parallel_rank * tensor_chunk_size
    tensor_end = tensor_start + tensor_chunk_size

    # Calculate the pipeline_parallel range
    pipeline_chunk_size = 1.0 / pipeline_parallel_world_size
    pipeline_start = pipeline_parallel_rank * pipeline_chunk_size
    pipeline_end = pipeline_start + pipeline_chunk_size

    return {
        "tensor_parallel": (tensor_start, tensor_end),
        "pipeline_parallel": (pipeline_start, pipeline_end),
    }


######################################################################################


def adjust_text_encoder(
    transformer: nn.Module,  # encoder(T5Stack)
    cpu_full_transformer: nn.Module,
    old_pp_range: tuple,
    new_pp_range: tuple,
    old_tp_range: tuple,
    new_tp_range: tuple = None,
    slice_bias: bool = True,
):
    total_blocks = len(cpu_full_transformer.block)

    old_pp_start, old_pp_end = range_to_indices(old_pp_range, total_blocks)
    new_pp_start, new_pp_end = range_to_indices(new_pp_range, total_blocks)

    blocks = transformer.block
    cpu_blocks = cpu_full_transformer.block

    current_block_count = len(blocks)
    expected_count = old_pp_end - old_pp_start
    if current_block_count != expected_count:
        raise ValueError(
            "Current transformer.transformer_blocks count does not match old_pp_range assumption."
        )

    intersect_start = max(old_pp_start, new_pp_start)
    intersect_end = min(old_pp_end, new_pp_end)
    logger.debug(
        f"old_pp_start={old_pp_start}, old_pp_end={old_pp_end}, new_pp_start={new_pp_start}, new_pp_end={new_pp_end},current_block_count={current_block_count}, intersect_start={intersect_start}, intersect_end={intersect_end}"
    )
    new_layers = nn.ModuleList()
    if intersect_start < intersect_end:
        logger.debug("enter if")
        rel_start = intersect_start - old_pp_start
        rel_end = rel_start + (intersect_end - intersect_start)
        overlap_blocks = blocks[rel_start:rel_end]
        logger.debug(f"rel_start={rel_start}, rel_end={rel_end}")
        new_layers.extend(overlap_blocks)

        if new_pp_start < intersect_start:
            logger.debug("enter new_pp_start < intersect_start")
            missing_front_blocks = cpu_blocks[new_pp_start:intersect_start]
            missing_front_gpu = [copy_cpu_block_to_gpu(m) for m in missing_front_blocks]
            new_layers = nn.ModuleList(missing_front_gpu + list(new_layers))

        if intersect_end < new_pp_end:
            logger.debug("intersect_end < new_pp_end")
            missing_back_blocks = cpu_blocks[intersect_end:new_pp_end]
            missing_back_gpu = [copy_cpu_block_to_gpu(m) for m in missing_back_blocks]
            new_layers.extend(missing_back_gpu)
    else:
        logger.debug("enter else")
        needed_blocks = cpu_blocks[new_pp_start:new_pp_end]
        needed_blocks_gpu = [copy_cpu_block_to_gpu(m) for m in needed_blocks]
        new_layers = nn.ModuleList(needed_blocks_gpu)

    setattr(transformer, "block", new_layers)
    if new_tp_range is not None:
        adjust_text_encoder_linears_in_layers(
            layers=transformer.block,
            cpu_full_transformer=cpu_full_transformer,
            new_pp_range=new_pp_range,
            old_tp_range=old_tp_range,
            new_tp_range=new_tp_range,
            slice_bias=slice_bias,
        )
    return transformer


def adjust_text_encoder_linears_in_layers(
    layers: nn.ModuleList,
    cpu_full_transformer: nn.Module,
    new_pp_range: tuple,
    old_tp_range: tuple,
    new_tp_range: tuple,
    slice_bias: bool = True,
):
    # get total_blocks from cpu_full_transformer.transformer_blocks
    total_blocks = len(cpu_full_transformer.block)
    new_pp_start, new_pp_end = range_to_indices(new_pp_range, total_blocks)
    for i, gpu_block in enumerate(layers):  # T5Block
        cpu_block = cpu_full_transformer.block[new_pp_start + i]
        gpu_named_modules = dict(gpu_block.named_modules())
        cpu_named_modules = dict(cpu_block.named_modules())
        for name, gpu_submodule in gpu_named_modules.items():
            if isinstance(gpu_submodule, nn.Linear) and name in [
                "layer.0.SelfAttention.module.q",
                "layer.0.SelfAttention.module.k",
                "layer.0.SelfAttention.module.v",
                "layer.0.SelfAttention.module.o",
                "layer.1.module.DenseReluDense.wi_0",
                "layer.1.module.DenseReluDense.wi_1",
                "layer.1.module.DenseReluDense.wo",
            ]:
                cpu_submodule = cpu_named_modules[name]
                tp_split_type = determine_text_encoder_tp_split_type_by_name(name)
                logger.debug(
                    f"before: total_len = {len(layers)}, new_pp_start = {new_pp_start}, new_pp_end = {new_pp_end},  name={name}, tp_split_type={tp_split_type}"
                )
                adjust_linear_tensor_parallel(
                    linear_module=gpu_submodule,
                    cpu_module=cpu_submodule,
                    old_tp_range=old_tp_range,
                    new_tp_range=new_tp_range,
                    tp_split_type=tp_split_type,
                    slice_bias=slice_bias,
                )


def determine_text_encoder_tp_split_type_by_name(name: str) -> str:
    # Based on the name, determine the row/column splitting logic:
    # Inside Attention:
    # to_q, to_k, to_v, add_k_proj, add_q_proj, add_v_proj: row split
    # to_out, to_add_out: column split

    # Inside FeedForward:
    # net.0.proj: row split
    # net.2: column split
    lower_name = name.lower()
    if any(
        x in lower_name
        for x in [
            ".q",
            ".k",
            ".v",
        ]
    ):
        return "row"
    if ".o" in lower_name:
        return "column"
    if ".wi_1" in lower_name or ".wi_0" in lower_name:
        return "row"
    if ".wo" in lower_name:
        return "column"
    return "row"  # default row
