from abc import abstractmethod, ABCMeta
from functools import wraps

from diffusers.schedulers import SchedulerMixin
from hetu_dit.core.distributed import (
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
)
from hetu_dit.model_executor.diffusion_executor.base_diffusion_warpper import (
    hetuDiTDiffusionBaseWrapper,
)


class hetuDiTSchedulerBaseWrapper(hetuDiTDiffusionBaseWrapper, metaclass=ABCMeta):
    def __init__(
        self,
        module: SchedulerMixin,
    ):
        super().__init__(
            module=module,
        )

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

    def __setattr__(self, name, value):
        if name == "module":
            super().__setattr__(name, value)
        elif (
            hasattr(self, "module")
            and self.module is not None
            and hasattr(self.module, name)
        ):
            setattr(self.module, name, value)
        else:
            super().__setattr__(name, value)

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @staticmethod
    def check_to_use_naive_step(func):
        @wraps(func)
        def check_naive_step_fn(self, *args, **kwargs):
            if (
                get_pipeline_parallel_world_size() == 1
                and get_sequence_parallel_world_size() == 1
            ):
                return self.module.step(*args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return check_naive_step_fn
