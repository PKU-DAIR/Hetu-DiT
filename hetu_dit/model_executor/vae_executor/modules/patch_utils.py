import torch
import torch.nn as nn
import torch.distributed as dist

from hetu_dit.core.distributed import (
    get_world_group,
    get_world_group_rank,
    get_world_group_world_size,
)
from hetu_dit.logger import init_logger

logger = init_logger(__name__)


class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = get_world_group_rank()
        self.world_size = get_world_group_world_size()

    def forward(self, hidden_state):
        height = hidden_state.shape[2]
        start_idx = (
            (height + get_world_group_world_size() - 1)
            // get_world_group_world_size()
            * get_world_group_rank()
        )
        end_idx = min(
            (height + get_world_group_world_size() - 1)
            // get_world_group_world_size()
            * (get_world_group_rank() + 1),
            height,
        )

        return hidden_state[:, :, start_idx:end_idx, :].clone()


class DePatchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = get_world_group_rank()
        self.world_size = get_world_group_world_size()

    def forward(self, patch_hidden_state):
        logger.debug(
            f"before depatchify, get_world_group_rank: {get_world_group_rank()}, get_world_group_world_size: {get_world_group_world_size()}"
        )
        patch_height_list = [
            torch.empty([1], dtype=torch.int64, device=f"cuda:{0}")
            for _ in range(get_world_group_world_size())
        ]
        dist.all_gather(
            patch_height_list,
            torch.tensor(
                [patch_hidden_state.shape[2]], dtype=torch.int64, device=f"cuda:{0}"
            ),
            group=get_world_group().device_group,
        )
        patch_hidden_state_list = [
            torch.empty(
                [
                    patch_hidden_state.shape[0],
                    patch_hidden_state.shape[1],
                    patch_height_list[i].item(),
                    patch_hidden_state.shape[-1],
                ],
                dtype=patch_hidden_state.dtype,
                device=f"cuda:{0}",
            )
            for i in range(get_world_group_world_size())
        ]
        dist.all_gather(
            patch_hidden_state_list,
            patch_hidden_state.contiguous(),
            group=get_world_group().device_group,
        )
        logger.debug(
            f"after depatchify, get_world_group_rank: {get_world_group_rank()}, get_world_group_world_size: {get_world_group_world_size()}"
        )
        return torch.cat(patch_hidden_state_list, dim=2)
