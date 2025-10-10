from .scheduler import Scheduler


class RequestManager:
    def __init__(self) -> None:
        self.scheduler = Scheduler()
