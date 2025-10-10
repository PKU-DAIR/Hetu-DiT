from hetu_dit.model_executor.base_wrapper import hetuDiTBaseWrapper


class hetuDiTVAEBaseWrapper(hetuDiTBaseWrapper):
    def __init__(self, module, parallel_config=None, ranks=None):
        super().__init__(module=module)

    def __getattr__(self, name: str):
        try:
            return getattr(self.module, name)
        except RecursionError:
            raise AttributeError(
                f"module {type(self.module).__name__} has no attribute {name}"
            )
