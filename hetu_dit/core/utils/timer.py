import torch
import time
from hetu_dit.logger import init_logger

logger = init_logger(__name__)


def gpu_timer_decorator(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start_time = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()

        if torch.distributed.get_rank() == 0:
            logger.info(
                f"{func.__name__} took {end_time - start_time} seconds to run on GPU."
            )
        return result

    return wrapper
