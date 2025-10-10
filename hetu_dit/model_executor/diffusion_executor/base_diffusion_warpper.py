from hetu_dit.model_executor.base_wrapper import hetuDiTBaseWrapper


class hetuDiTDiffusionBaseWrapper(hetuDiTBaseWrapper):
    def __init__(self, module, parallel_config=None, ranks=None):
        """
        Args:
            model: The underlying model
            parallel_config: Parallel configuration
            ranks: The ranks on which this instance will run, e.g. [0, 3, 4, 7]
        """
        super().__init__(module=module)

    def __getattr__(self, name: str):
        try:
            return getattr(self.module, name)
        except RecursionError:
            raise AttributeError(
                f"module {type(self.module).__name__} has no attribute {name}"
            )
