from typing import Dict, Type
import torch.nn as nn

from hetu_dit.logger import init_logger
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

logger = init_logger(__name__)


class hetuDiTLayerWrappersRegister:
    _HETUDIT_LAYER_MAPPING: Dict = {}

    @classmethod
    def register(cls, origin_layer_class: Type[nn.Module]):
        def decorator(hetu_dit_layer_wrapper):
            cls._HETUDIT_LAYER_MAPPING[origin_layer_class] = hetu_dit_layer_wrapper
            return hetu_dit_layer_wrapper

        return decorator

    @classmethod
    def get_wrapper(cls, layer: nn.Module):
        candidate = None
        candidate_origin = None
        for (
            origin_layer_class,
            hetu_dit_layer_wrapper,
        ) in cls._HETUDIT_LAYER_MAPPING.items():
            if isinstance(layer, origin_layer_class):
                if (
                    (candidate is None and candidate_origin is None)
                    or origin_layer_class == layer.__class__
                    or issubclass(origin_layer_class, candidate_origin)
                ):
                    candidate_origin = origin_layer_class
                    candidate = hetu_dit_layer_wrapper

        if candidate is None:
            raise ValueError(
                f"Layer class {layer.__class__.__name__} is not supported by hetuDiT"
            )
        else:
            return candidate


class hetuDiTAttentionProcessorRegister:
    _HETUDIT_ATTENTION_PROCESSOR_MAPPING = {}

    @classmethod
    def register(cls, origin_processor_class):
        def decorator(hetu_dit_processor):
            if not issubclass(hetu_dit_processor, origin_processor_class):
                raise ValueError(
                    f"{hetu_dit_processor.__class__.__name__} is not a subclass of origin class {origin_processor_class.__class__.__name__}"
                )
            cls._HETUDIT_ATTENTION_PROCESSOR_MAPPING[origin_processor_class] = (
                hetu_dit_processor
            )
            return hetu_dit_processor

        return decorator

    @classmethod
    def get_processor(cls, processor):
        for (
            origin_processor_class,
            hetu_dit_processor,
        ) in cls._HETUDIT_ATTENTION_PROCESSOR_MAPPING.items():
            if isinstance(processor, origin_processor_class):
                return hetu_dit_processor
        raise ValueError(
            f"Attention Processor class {processor.__class__.__name__} is not supported by hetuDiT"
        )


class hetuDiTTransformerWrappersRegister:
    _HETUDIT_TRANSFORMER_MAPPING: Dict = {}

    @classmethod
    def register(cls, origin_transformer_class: Type[nn.Module]):
        def decorator(hetu_dit_transformer_class: Type[nn.Module]):
            cls._HETUDIT_TRANSFORMER_MAPPING[origin_transformer_class] = (
                hetu_dit_transformer_class
            )
            return hetu_dit_transformer_class

        return decorator

    @classmethod
    def get_wrapper(cls, transformer: nn.Module):
        candidate = None
        candidate_origin = None
        for (
            origin_transformer_class,
            wrapper_class,
        ) in cls._HETUDIT_TRANSFORMER_MAPPING.items():
            if isinstance(transformer, origin_transformer_class):
                if (
                    candidate is None
                    or origin_transformer_class == transformer.__class__
                    or issubclass(origin_transformer_class, candidate_origin)
                ):
                    candidate_origin = origin_transformer_class
                    candidate = wrapper_class

        if candidate is None:
            raise ValueError(
                f"Transformer class {transformer.__class__.__name__} "
                f"is not supported by hetuDiT"
            )
        else:
            return candidate


class hetuDiTTextEncoderModelWrappersRegister:
    _HETUDIT_TEXTENCODER_MAPPING: Dict = {}

    @classmethod
    def register(cls, origin_textencoder_class: Type[nn.Module]):
        def decorator(hetu_dit_textencoder_class: Type[nn.Module]):
            cls._HETUDIT_TEXTENCODER_MAPPING[origin_textencoder_class] = (
                hetu_dit_textencoder_class
            )
            return hetu_dit_textencoder_class

        return decorator

    @classmethod
    def get_wrapper(cls, textencoder: nn.Module):
        candidate = None
        candidate_origin = None
        for (
            origin_textencoder_class,
            wrapper_class,
        ) in cls._HETUDIT_TEXTENCODER_MAPPING.items():
            if isinstance(textencoder, origin_textencoder_class):
                if (
                    candidate is None
                    or origin_textencoder_class == textencoder.__class__
                    or issubclass(origin_textencoder_class, candidate_origin)
                ):
                    candidate_origin = origin_textencoder_class
                    candidate = wrapper_class

        if candidate is None:
            raise ValueError(
                f"Textencoder class {textencoder.__class__.__name__} "
                f"is not supported by hetuDiT"
            )
        else:
            return candidate


class hetuDiTSchedulerWrappersRegister:
    _HETUDIT_SCHEDULER_MAPPING: Dict = {}

    @classmethod
    def register(cls, origin_scheduler_class: Type[nn.Module]):
        def decorator(hetu_dit_scheduler_class: Type[nn.Module]):
            cls._HETUDIT_SCHEDULER_MAPPING[origin_scheduler_class] = (
                hetu_dit_scheduler_class
            )
            return hetu_dit_scheduler_class

        return decorator

    @classmethod
    def get_wrapper(cls, scheduler: nn.Module):
        candidate = None
        candidate_origin = None
        for (
            origin_scheduler_class,
            wrapper_class,
        ) in cls._HETUDIT_SCHEDULER_MAPPING.items():
            if isinstance(scheduler, origin_scheduler_class):
                if (
                    (candidate is None and candidate_origin is None)
                    or origin_scheduler_class == scheduler.__class__
                    or issubclass(origin_scheduler_class, candidate_origin)
                ):
                    candidate_origin = origin_scheduler_class
                    candidate = wrapper_class

        if candidate is None:
            raise ValueError(
                f"Scheduler class {scheduler.__class__.__name__} "
                f"is not supported by hetuDiT"
            )
        else:
            return candidate


class hetuDiTPipelineWrapperRegister:
    _HETUDIT_PIPE_MAPPING: Dict = {}

    @classmethod
    def register(cls, origin_pipe_class):
        def decorator(hetu_dit_pipe_class):
            cls._HETUDIT_PIPE_MAPPING[origin_pipe_class] = hetu_dit_pipe_class
            return hetu_dit_pipe_class

        return decorator

    @classmethod
    def get_class(cls, pipe):
        if isinstance(pipe, type):
            candidate = None
            candidate_origin = None
            for (
                origin_model_class,
                hetu_dit_model_class,
            ) in cls._HETUDIT_PIPE_MAPPING.items():
                if issubclass(pipe, origin_model_class):
                    if (candidate is None and candidate_origin is None) or issubclass(
                        origin_model_class, candidate_origin
                    ):
                        candidate_origin = origin_model_class
                        candidate = hetu_dit_model_class
            if candidate is None:
                raise ValueError(
                    f"Diffusion Pipeline class {pipe} is not supported by hetuDiT"
                )
            else:
                return candidate
        elif isinstance(pipe, DiffusionPipeline):
            candidate = None
            candidate_origin = None
            for (
                origin_model_class,
                hetu_dit_model_class,
            ) in cls._HETUDIT_PIPE_MAPPING.items():
                if isinstance(pipe, origin_model_class):
                    if (candidate is None and candidate_origin is None) or issubclass(
                        origin_model_class, candidate_origin
                    ):
                        candidate_origin = origin_model_class
                        candidate = hetu_dit_model_class

            if candidate is None:
                raise ValueError(
                    f"Diffusion Pipeline class {pipe.__class__} "
                    f"is not supported by hetuDiT"
                )
            else:
                return candidate
        else:
            raise ValueError(f"Unsupported type {type(pipe)} for pipe")
