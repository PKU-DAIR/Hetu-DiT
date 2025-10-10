from abc import abstractmethod, ABCMeta

import torch.nn as nn

from hetu_dit.model_executor.textencoder_executor.base_textencoder_warpper import (
    hetuDiTTextEncoderBaseWrapper,
)


class hetuDiTTextEncoderLayerBaseWrapper(
    nn.Module, hetuDiTTextEncoderBaseWrapper, metaclass=ABCMeta
):
    def __init__(self, module: nn.Module):
        super().__init__()
        super(nn.Module, self).__init__(module=module)
        self.activation_cache = None  # used as cache reuse to accelerate inference

    def __getattr__(self, name: str):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        try:
            return getattr(self.module, name)
        except RecursionError:
            raise AttributeError(
                f"module {type(self.module).__name__} has no attribute {name}"
            )

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
