from typing import Optional
import torch
import torch.distributed
import hetu_dit.envs as envs
from hetu_dit.logger import init_logger
from hetu_dit.model_executor.utils.register_warpper import (
    hetuDiTTransformerWrappersRegister,
    hetuDiTTextEncoderModelWrappersRegister,
)
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PatchEmbed, CogVideoXPatchEmbed
from diffusers import StableDiffusion3Pipeline
from diffusers import CogVideoXPipeline
from diffusers import FluxPipeline
from diffusers import HunyuanDiTPipeline
from diffusers import HunyuanVideoPipeline
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerFF

logger = init_logger(__name__)


class ModelSingleton:
    _instances = {}

    def __new__(cls, model_class, model_name_or_path, **kwargs):
        """
        Parameters:
            - model_class: Any class that supports the `from_pretrained` method.
            - model_name_or_path: Model path or name.
            - **kwargs: Other initialization parameters, such as device configuration, loading options, etc.
        """
        cls.model_class = model_class
        key = (
            model_class,
            model_name_or_path,
        )  # Unique identifier for a model instance
        if key not in cls._instances:
            # If the model instance does not exist, create a new instance.
            cls._instances[key] = super().__new__(cls)
            cls._instances[key]._initialize_model(
                model_class, model_name_or_path, **kwargs
            )
        return cls._instances[key]

    def _initialize_model(self, model_class, model_name_or_path, **kwargs):
        """
        Initialize model instance.

        Parameters:
        - model_class: Any class that supports the `from_pretrained` method.
        - model_name_or_path: Model path or name.
        - **kwargs: Optional parameters passed to the `from_pretrained` method.
        """
        self.model_class = model_class
        logger.info(
            f"Initializing model {model_class.__name__} with: {model_name_or_path}"
        )
        self.model = model_class.from_pretrained(model_name_or_path, **kwargs).to("cpu")

    def get_model(self, use_text_encoder_parallel=False):
        """
        Retrieve the loaded model instance.
        """
        transformer = getattr(self.model, "transformer", None)
        unet = getattr(self.model, "unet", None)
        vae = getattr(self.model, "vae", None)
        scheduler = getattr(self.model, "scheduler", None)
        wrapper = hetuDiTTransformerWrappersRegister.get_wrapper(transformer)
        if self.model_class == StableDiffusion3Pipeline:
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward, nn.Conv2d, PatchEmbed],
                submodule_name_to_wrap=["attn"],
            )
        elif self.model_class == CogVideoXPipeline:
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward, nn.Conv2d, CogVideoXPatchEmbed],
                submodule_name_to_wrap=["attn1"],
            )
        elif self.model_class == FluxPipeline:
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward],
                submodule_name_to_wrap=["attn"],
            )
        elif self.model_class == HunyuanVideoPipeline:
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward],
                submodule_name_to_wrap=["attn"],
            )
        elif self.model_class == HunyuanDiTPipeline:
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward, nn.Conv2d, PatchEmbed],
                submodule_name_to_wrap=["attn1"],
            )

        if use_text_encoder_parallel:
            if hasattr(self.model, "text_encoder_3"):
                text_encoder = getattr(self.model, "text_encoder_3", None)
                if text_encoder is not None:
                    text_encoder_wrapper = (
                        hetuDiTTextEncoderModelWrappersRegister.get_wrapper(
                            text_encoder
                        )
                    )
                    self.model.text_encoder_3 = (
                        text_encoder_wrapper._static_wrap_layers(
                            model=text_encoder,
                            submodule_classes_to_wrap=[T5Attention, T5LayerFF],
                        )
                    )
            elif hasattr(self.model, "text_encoder_2"):
                text_encoder = getattr(self.model, "text_encoder_2", None)
                if text_encoder is not None:
                    text_encoder_wrapper = (
                        hetuDiTTextEncoderModelWrappersRegister.get_wrapper(
                            text_encoder
                        )
                    )
                    self.model.text_encoder_2 = (
                        text_encoder_wrapper._static_wrap_layers(
                            model=text_encoder,
                            submodule_classes_to_wrap=[T5Attention, T5LayerFF],
                        )
                    )
            elif hasattr(self.model, "text_encoder"):
                text_encoder = getattr(self.model, "text_encoder", None)
                if text_encoder is not None:
                    text_encoder_wrapper = (
                        hetuDiTTextEncoderModelWrappersRegister.get_wrapper(
                            text_encoder
                        )
                    )
                    self.model.text_encoder = text_encoder_wrapper._static_wrap_layers(
                        model=text_encoder,
                        submodule_classes_to_wrap=[T5Attention, T5LayerFF],
                    )
        return self.model


_SINGLETON_MODEL_MANAGER: Optional[ModelSingleton] = None


def set_singleton_model_manager(
    model_class, model_name_or_path, use_text_encoder_parallel, **kwargs
):
    global _SINGLETON_MODEL_MANAGER
    if _SINGLETON_MODEL_MANAGER is None:
        _SINGLETON_MODEL_MANAGER = ModelSingleton(
            model_class=model_class,
            model_name_or_path=model_name_or_path,
            torch_dtype=torch.float16
            if model_class == StableDiffusion3Pipeline
            or model_class == HunyuanDiTPipeline
            else torch.bfloat16,
        ).get_model(use_text_encoder_parallel)


def get_singleton_model_manager():
    return _SINGLETON_MODEL_MANAGER
