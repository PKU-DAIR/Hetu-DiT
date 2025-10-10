import sys
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import torch
import torch.distributed

from hetu_dit.logger import init_logger
from hetu_dit.core.distributed import init_distributed_environment
from hetu_dit.config.config import (
    EngineConfig,
    ParallelConfig,
    TensorParallelConfig,
    PipeFusionParallelConfig,
    SequenceParallelConfig,
    DataParallelConfig,
    ModelConfig,
    InputConfig,
    RuntimeConfig,
)

logger = init_logger(__name__)


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = "--" + key[len("--") :].replace("-", "_")
                    processed_args.append(f"{key}={value}")
                else:
                    processed_args.append("--" + arg[len("--") :].replace("-", "_"))
            else:
                processed_args.append(arg)

        return super().parse_args(processed_args, namespace)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class hetuDiTArgs:
    """Arguments for hetuDiT engine."""

    # Model arguments
    model: str
    download_dir: Optional[str] = None
    trust_remote_code: bool = False
    # Runtime arguments
    warmup_steps: int = 1
    use_parallel_vae: bool = False
    use_torch_compile: bool = False
    use_onediff: bool = False
    adjust_strategy: str = "cache"
    # Parallel arguments
    # data parallel
    data_parallel_degree: int = 1
    use_cfg_parallel: bool = False
    # sequence parallel
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None
    # tensor parallel
    tensor_parallel_degree: int = 1
    split_scheme: Optional[str] = "row"
    # pipefusion parallel
    pipefusion_parallel_degree: int = 1
    num_pipeline_patch: Optional[int] = None
    attn_layer_num_for_pp: Optional[List[int]] = None
    # Input arguments
    height: int = 1024
    width: int = 1024
    num_frames: int = 49
    num_inference_steps: int = 20
    max_sequence_length: int = 256
    img_file_path: Optional[str] = None
    prompt: Union[str, List[str]] = ""
    negative_prompt: Union[str, List[str]] = ""
    no_use_resolution_binning: bool = False
    seed: int = 42
    output_type: str = "pil"
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_tiling: bool = False
    enable_slicing: bool = False
    # Text encoder parallel
    use_parallel_text_encoder: bool = False
    text_encoder_tensor_parallel_degree: int = 1

    machine_num: int = 1
    use_disaggregated_encode_decode: bool = False
    stage_level: bool = False
    encode_worker_ids: Optional[List[int]] = None
    decode_worker_ids: Optional[List[int]] = None

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser):
        """Add command line arguments for hetuDiT engine configuration.

        Organizes arguments into groups: model options, runtime options,
        parallel processing options, and input options.

        Args:
            parser (FlexibleArgumentParser): The argument parser to configure.

        Returns:
            FlexibleArgumentParser: The configured parser with added CLI arguments.
        """
        # Model arguments
        model_group = parser.add_argument_group("Model Options")
        model_group.add_argument(
            "--model",
            type=str,
            default="/Path/to/your/models/stable-diffusion-3-medium-diffusers",
            help="Name or path of the huggingface model to use.",
        )
        model_group.add_argument(
            "--download-dir",
            type=nullable_str,
            default=hetuDiTArgs.download_dir,
            help="Directory to download and load the weights, default to the default cache dir of huggingface.",
        )
        model_group.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code from huggingface.",
        )

        # Runtime arguments
        runtime_group = parser.add_argument_group("Runtime Options")
        runtime_group.add_argument(
            "--warmup_steps", type=int, default=1, help="Warmup steps in generation."
        )

        runtime_group.add_argument("--use_parallel_vae", action="store_true")

        runtime_group.add_argument(
            "--use_torch_compile",
            action="store_true",
            help="Enable torch.compile to accelerate inference in a single card",
        )
        runtime_group.add_argument(
            "--use_onediff",
            action="store_true",
            help="Enable onediff to accelerate inference in a single card",
        )
        runtime_group.add_argument(
            "--adjust_strategy",
            type=str,
            default="cache",
            choices=["base", "cache", "p2p"],
            help="Execution strategy (base, cache, p2p). Default is cache.",
        )

        # Parallel arguments
        parallel_group = parser.add_argument_group("Parallel Processing Options")
        parallel_group.add_argument(
            "--use_cfg_parallel",
            action="store_true",
            help="Use split batch in classifier_free_guidance. cfg_degree will be 2 if set",
        )
        parallel_group.add_argument(
            "--data_parallel_degree", type=int, default=1, help="Data parallel degree."
        )
        parallel_group.add_argument(
            "--ulysses_degree",
            type=int,
            default=None,
            help="Ulysses sequence parallel degree. Used in attention layer.",
        )
        parallel_group.add_argument(
            "--ring_degree",
            type=int,
            default=None,
            help="Ring sequence parallel degree. Used in attention layer.",
        )
        parallel_group.add_argument(
            "--pipefusion_parallel_degree",
            type=int,
            default=1,
            help="Pipefusion parallel degree. Indicates the number of pipeline stages.",
        )
        parallel_group.add_argument(
            "--num_pipeline_patch",
            type=int,
            default=None,
            help="Number of patches the feature map should be segmented in pipefusion parallel.",
        )
        parallel_group.add_argument(
            "--attn_layer_num_for_pp",
            default=None,
            nargs="*",
            type=int,
            help="List representing the number of layers per stage of the pipeline in pipefusion parallel",
        )
        parallel_group.add_argument(
            "--tensor_parallel_degree",
            type=int,
            default=1,
            help="Tensor parallel degree.",
        )
        parallel_group.add_argument(
            "--split_scheme",
            type=str,
            default="row",
            help="Split scheme for tensor parallel.",
        )

        # Text encoder parallel args
        parallel_group.add_argument(
            "--use_parallel_text_encoder",
            action="store_true",
            help="Use text encoder tensor parallel",
        )
        parallel_group.add_argument(
            "--text_encoder_tensor_parallel_degree",
            type=int,
            default=1,
            help="Text encoder tensor parallel degree.",
        )

        # Input arguments
        input_group = parser.add_argument_group("Input Options")
        input_group.add_argument(
            "--height", type=int, default=1024, help="The height of image"
        )
        input_group.add_argument(
            "--width", type=int, default=1024, help="The width of image"
        )
        input_group.add_argument(
            "--num_frames", type=int, default=49, help="The frames of video"
        )
        input_group.add_argument(
            "--img_file_path", type=str, default=None, help="Path for the input image."
        )
        input_group.add_argument(
            "--prompt", type=str, nargs="*", default="", help="Prompt for the model."
        )
        input_group.add_argument("--no_use_resolution_binning", action="store_true")
        input_group.add_argument(
            "--negative_prompt",
            type=str,
            nargs="*",
            default="",
            help="Negative prompt for the model.",
        )
        input_group.add_argument(
            "--num_inference_steps",
            type=int,
            default=20,
            help="Number of inference steps.",
        )
        input_group.add_argument(
            "--max_sequence_length",
            type=int,
            default=256,
            help="Max sequencen length of prompt",
        )
        runtime_group.add_argument(
            "--seed", type=int, default=42, help="Random seed for operations."
        )
        runtime_group.add_argument(
            "--output_type",
            type=str,
            default="pil",
            help="Output type of the pipeline.",
        )
        runtime_group.add_argument(
            "--enable_sequential_cpu_offload",
            action="store_true",
            help="Offloading the weights to the CPU.",
        )
        runtime_group.add_argument(
            "--enable_model_cpu_offload",
            action="store_true",
            help="Offloading the weights to the CPU.",
        )
        runtime_group.add_argument(
            "--enable_tiling",
            action="store_true",
            help="Making VAE decode a tile at a time to save GPU memory.",
        )
        runtime_group.add_argument(
            "--enable_slicing",
            action="store_true",
            help="Making VAE decode a tile at a time to save GPU memory.",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_config(
        self, is_serving: bool = False
    ) -> Tuple[EngineConfig, InputConfig]:
        """Generate engine and input configuration from arguments.

        Constructs EngineConfig and InputConfig objects based on the parsed
        CLI arguments. Initializes distributed environment if not already done
        when not in serving mode.

        Args:
            is_serving (bool): Flag indicating whether the engine is in serving mode. Defaults to False.

        Returns:
            Tuple[EngineConfig, InputConfig]: Engine and input configurations.
        """
        if not is_serving:
            if not torch.distributed.is_initialized():
                logger.warning(
                    "Distributed environment is not initialized. Initializing..."
                )
                init_distributed_environment()

        model_config = ModelConfig(
            model=self.model,
            download_dir=self.download_dir,
            trust_remote_code=self.trust_remote_code,
        )

        runtime_config = RuntimeConfig(
            warmup_steps=self.warmup_steps,
            use_parallel_vae=self.use_parallel_vae,
            use_torch_compile=self.use_torch_compile,
            use_onediff=self.use_onediff,
            adjust_strategy=self.adjust_strategy,
            use_parallel_text_encoder=self.use_parallel_text_encoder,
        )

        # Text encoder parallel
        if self.use_parallel_text_encoder:
            parallel_config = ParallelConfig(
                dp_config=DataParallelConfig(
                    dp_degree=self.data_parallel_degree,
                    use_cfg_parallel=self.use_cfg_parallel,
                    is_serving=is_serving,
                ),
                sp_config=SequenceParallelConfig(
                    ulysses_degree=self.ulysses_degree,
                    ring_degree=self.ring_degree,
                    is_serving=is_serving,
                ),
                tp_config=TensorParallelConfig(
                    tp_degree=self.tensor_parallel_degree,
                    split_scheme=self.split_scheme,
                    is_serving=is_serving,
                ),
                pp_config=PipeFusionParallelConfig(
                    pp_degree=self.pipefusion_parallel_degree,
                    num_pipeline_patch=self.num_pipeline_patch,
                    attn_layer_num_for_pp=self.attn_layer_num_for_pp,
                    is_serving=is_serving,
                ),
                text_encoder_tp_config=TensorParallelConfig(
                    tp_degree=self.text_encoder_tensor_parallel_degree,
                    split_scheme=self.split_scheme,
                    is_serving=is_serving,
                ),
                is_serving=is_serving,
            )
        else:
            parallel_config = ParallelConfig(
                dp_config=DataParallelConfig(
                    dp_degree=self.data_parallel_degree,
                    use_cfg_parallel=self.use_cfg_parallel,
                    is_serving=is_serving,
                ),
                sp_config=SequenceParallelConfig(
                    ulysses_degree=self.ulysses_degree,
                    ring_degree=self.ring_degree,
                    is_serving=is_serving,
                ),
                tp_config=TensorParallelConfig(
                    tp_degree=self.tensor_parallel_degree,
                    split_scheme=self.split_scheme,
                    is_serving=is_serving,
                ),
                pp_config=PipeFusionParallelConfig(
                    pp_degree=self.pipefusion_parallel_degree,
                    num_pipeline_patch=self.num_pipeline_patch,
                    attn_layer_num_for_pp=self.attn_layer_num_for_pp,
                    is_serving=is_serving,
                ),
                is_serving=is_serving,
            )

        engine_config = EngineConfig(
            model_config=model_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
            machine_num=self.machine_num,
        )

        input_config = InputConfig(
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            use_resolution_binning=not self.no_use_resolution_binning,
            batch_size=len(self.prompt) if isinstance(self.prompt, list) else 1,
            img_file_path=self.img_file_path,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            seed=self.seed,
            output_type=self.output_type,
        )

        return engine_config, input_config
