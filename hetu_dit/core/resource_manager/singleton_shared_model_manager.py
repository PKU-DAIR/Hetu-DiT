import ray
import torch
import torch.nn as nn
from hetu_dit.logger import init_logger
from hetu_dit.model_executor.utils.register_warpper import (
    hetuDiTTransformerWrappersRegister,
    hetuDiTTextEncoderModelWrappersRegister,
)
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PatchEmbed, CogVideoXPatchEmbed
from diffusers import (
    StableDiffusion3Pipeline,
    CogVideoXPipeline,
    FluxPipeline,
    HunyuanDiTPipeline,
    HunyuanVideoPipeline,
)
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerFF

logger = init_logger(__name__)


class ModelSingleton:
    _instances = {}

    def __new__(cls, model_class, model_name_or_path, **kwargs):
        key = (model_class, model_name_or_path)
        if key not in cls._instances:
            inst = super().__new__(cls)
            cls._instances[key] = inst
            inst._initialize_model(model_class, model_name_or_path, **kwargs)
        return cls._instances[key]

    def _initialize_model(self, model_class, model_name_or_path, **kwargs):
        logger.info(
            f"[ModelSingleton] loading {model_class.__name__} @ {model_name_or_path}"
        )
        dtype = (
            torch.float16
            if model_class in (StableDiffusion3Pipeline, HunyuanDiTPipeline)
            else torch.bfloat16
        )
        # load model to CPU first
        self.model = model_class.from_pretrained(
            model_name_or_path, torch_dtype=dtype, **kwargs
        ).to("cpu")

    def get_model(self, use_text_encoder_parallel: bool = False):
        transformer = getattr(self.model, "transformer", None)
        wrapper = hetuDiTTransformerWrappersRegister.get_wrapper(transformer)

        if isinstance(self.model, StableDiffusion3Pipeline):
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward, nn.Conv2d, PatchEmbed],
                submodule_name_to_wrap=["attn"],
            )
        elif isinstance(self.model, CogVideoXPipeline):
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward, nn.Conv2d, CogVideoXPatchEmbed],
                submodule_name_to_wrap=["attn1"],
            )
        elif isinstance(self.model, FluxPipeline):
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward],
                submodule_name_to_wrap=["attn"],
            )
        elif isinstance(self.model, HunyuanVideoPipeline):
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward],
                submodule_name_to_wrap=["attn"],
            )
        elif isinstance(self.model, HunyuanDiTPipeline):
            self.model.transformer = wrapper._static_wrap_layers(
                model=transformer,
                submodule_classes_to_wrap=[FeedForward, nn.Conv2d, PatchEmbed],
                submodule_name_to_wrap=["attn1"],
            )

        if use_text_encoder_parallel:
            for attr in ("text_encoder_3", "text_encoder_2", "text_encoder"):
                te = getattr(self.model, attr, None)
                if te is not None:
                    tw = hetuDiTTextEncoderModelWrappersRegister.get_wrapper(te)
                    setattr(
                        self.model,
                        attr,
                        tw._static_wrap_layers(
                            model=te,
                            submodule_classes_to_wrap=[T5Attention, T5LayerFF],
                        ),
                    )
                    break

        return self.model


# Global handle & Actor name
_SINGLETON_LOCAL = None
_MODEL_ACTOR_NAME = "SharedModelActor_v1"


@ray.remote
class SharedModelActor:
    """
    Singleton process, resident model.
    """

    def __init__(
        self, model_class, model_name_or_path, use_text_encoder_parallel=False, **kwargs
    ):
        ms = ModelSingleton(model_class, model_name_or_path, **kwargs)
        self.model = ms.get_model(use_text_encoder_parallel)

    def call(self, method_path: str, *args, **kwargs):
        """
        Support arbitrary depth attribute and method calls.
        Only pass method_path and input arguments, model weights are never serialized.
        Args:
            method_path: Dotted path to method or attribute.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.
        Returns:
            Result of the method or attribute access.
        """
        parts = method_path.split(".")
        attr = self.model
        for p in parts:
            attr = getattr(attr, p)
        if callable(attr):
            return attr(*args, **kwargs)
        if args or kwargs:
            raise AttributeError(f"{method_path} is not callable")
        return attr


class _RemoteModelProxy:
    """
    RPC proxy: intercepts .xxx.attr(...)/.xxx.attr access, concatenates method_path to call Actor.
    """

    def __init__(self):
        self._actor = ray.get_actor(_MODEL_ACTOR_NAME)
        self._path = []

    def __getattr__(self, name: str):
        proxy = object.__new__(_RemoteModelProxy)
        proxy._actor = self._actor
        proxy._path = self._path + [name]
        return proxy

    def __call__(self, *args, **kwargs):
        method_path = ".".join(self._path)
        return ray.get(self._actor.call.remote(method_path, *args, **kwargs))

    def __len__(self):
        method_path = ".".join(self._path + ["__len__"])
        return ray.get(self._actor.call.remote(method_path))

    def __getitem__(self, idx):
        method_path = ".".join(self._path + ["__getitem__"])
        return ray.get(self._actor.call.remote(method_path, idx))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def set_singleton_model_manager(
    model_class, model_name_or_path, use_text_encoder_parallel=False, **kwargs
):
    """
    Only called once at the top level of the Driver:
      1) Load _SINGLETON_LOCAL locally
      2) Start the unique SharedModelActor with max_concurrency=8
    Args:
        model_class: The model class to instantiate.
        model_name_or_path: Path or name of the model.
        use_text_encoder_parallel: Whether to use parallel text encoder.
        **kwargs: Additional arguments for model initialization.
    """
    global _SINGLETON_LOCAL
    if _SINGLETON_LOCAL is None:
        ms = ModelSingleton(model_class, model_name_or_path, **kwargs)
        _SINGLETON_LOCAL = ms.get_model(use_text_encoder_parallel)
        SharedModelActor.options(
            name=_MODEL_ACTOR_NAME,
            lifetime="detached",
            max_concurrency=64,  # Improved concurrency
        ).remote(model_class, model_name_or_path, use_text_encoder_parallel, **kwargs)


def get_singleton_model_manager():
    """
    Returns the local instance in the Driver;
    Returns the RPC proxy in any other Ray Actor/Task,
    External call interface remains unchanged.
    """
    if _SINGLETON_LOCAL is not None:
        return _SINGLETON_LOCAL
    if ray.is_initialized():
        return _RemoteModelProxy()
    raise RuntimeError(
        "Please call set_singleton_model_manager(...) at the top level of the Driver first."
    )
