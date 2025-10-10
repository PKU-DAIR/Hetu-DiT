from abc import ABC, abstractmethod

from hetu_dit.config.config import EngineConfig


class ExecutorBase(ABC):
    """Base class for all executors.

    An executor is responsible for executing the model on a specific device
    type (e.g., CPU, GPU, Neuron, etc.). Or it can be a distributed executor
    that can execute the model on multiple devices.
    """

    @abstractmethod
    def __init__(
        self,
        engine_config: EngineConfig,
    ):
        pass

    @abstractmethod
    def _init_executor(self):
        pass


class ExecutorAsyncBase(ExecutorBase):
    @abstractmethod
    async def execute_model_async(
        self,
        engine_config: EngineConfig,
    ):
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    @abstractmethod
    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError
