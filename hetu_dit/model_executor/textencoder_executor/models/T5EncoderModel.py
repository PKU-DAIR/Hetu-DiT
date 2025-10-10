from abc import ABCMeta
from typing import Dict, List, Type
import torch.nn as nn

from hetu_dit.logger import init_logger
from hetu_dit.model_executor.textencoder_executor.models.base_model import (
    hetuDiTTextEncoderModelBaseWrapper,
)
from hetu_dit.model_executor.utils.register_warpper import (
    hetuDiTTextEncoderModelWrappersRegister,
)
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Attention, T5LayerFF

logger = init_logger(__name__)


class StageInfo:
    def __init__(self):
        self.after_flags: Dict[str, bool] = {}


@hetuDiTTextEncoderModelWrappersRegister.register(T5EncoderModel)
class hetuDiTT5EncoderModel(hetuDiTTextEncoderModelBaseWrapper, metaclass=ABCMeta):
    def __init__(
        self,
        encoder_model: T5EncoderModel,
        submodule_classes_to_wrap: List[Type] = [T5Attention, T5LayerFF],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
    ):
        self.stage_info = None
        encoder_model = self._convert_t5encoder_for_parallel(
            encoder_model,
            submodule_classes_to_wrap=submodule_classes_to_wrap,
            submodule_name_to_wrap=submodule_name_to_wrap,
            submodule_addition_args=submodule_addition_args,
        )
        super().__init__(module=encoder_model)

    def _convert_t5encoder_for_parallel(
        self,
        encoder_model: nn.Module,
        submodule_classes_to_wrap: List[Type] = [T5Attention, T5LayerFF],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
    ) -> nn.Module:
        encoder_model = self._wrap_layers(
            model=encoder_model,
            submodule_classes_to_wrap=submodule_classes_to_wrap,
            submodule_name_to_wrap=submodule_name_to_wrap,
            submodule_addition_args=submodule_addition_args,
        )
        return encoder_model

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)
