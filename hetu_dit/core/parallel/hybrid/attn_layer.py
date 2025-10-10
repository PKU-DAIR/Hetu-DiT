# Adapted from https://github.com/feifeibear/long-context-attention/tree/main
import torch
from torch import Tensor

import torch.distributed
from yunchang import LongContextAttention
from yunchang.comm.all_to_all import SeqAllToAll4D

from hetu_dit.logger import init_logger
from .utils import RING_IMPL_DICT
from yunchang.globals import PROCESS_GROUP
from hetu_dit.core.distributed.parallel_state import get_sequence_parallel_world_size

logger = init_logger(__name__)


class hetuDiTLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
    ) -> None:
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
        )
        self.use_kv_cache = use_kv_cache
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]

    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
    ) -> Tensor:
        """
        Forward pass for hetuDiTLongContextAttention.
        Args:
            attn: Attention layer.
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            joint_tensor_query: Joint query tensor (optional).
            joint_tensor_key: Joint key tensor (optional).
            joint_tensor_value: Joint value tensor (optional).
            dropout_p: Dropout probability.
            softmax_scale: Softmax scaling factor.
            causal: Whether to use causal attention.
            window_size: Window size for attention.
            alibi_slopes: Alibi slopes for attention.
            deterministic: Whether to use deterministic computation.
            return_attn_probs: Whether to return attention probabilities.
            joint_strategy: Joint strategy for query/key/value.
        Returns:
            output: Context output tensor.
        """
        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1

        self.use_kv_cache = get_sequence_parallel_world_size() > 1
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx
            )

        out = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_layer=attn if self.use_kv_cache else None,
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output


class hetuDiTJointLongContextAttention(hetuDiTLongContextAttention):
    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query,
        joint_tensor_key,
        joint_tensor_value,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="rear",
    ):
        """
        Forward pass for hetuDiTJointLongContextAttention.
        Args:
            attn: Attention layer.
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            joint_tensor_query: Joint query tensor.
            joint_tensor_key: Joint key tensor.
            joint_tensor_value: Joint value tensor.
            dropout_p: Dropout probability.
            softmax_scale: Softmax scaling factor.
            causal: Whether to use causal attention.
            window_size: Window size for attention.
            alibi_slopes: Alibi slopes for attention.
            deterministic: Whether to use deterministic computation.
            return_attn_probs: Whether to return attention probabilities.
            joint_strategy: Joint strategy for query/key/value.
        Returns:
            output: Context output tensor.
        """
        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1

        self.use_kv_cache = get_sequence_parallel_world_size() > 1
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        supported_joint_strategy = ["none", "front", "rear"]
        if joint_strategy not in supported_joint_strategy:
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
            )
        elif joint_strategy != "none" and joint_tensor_query is None:
            raise ValueError(
                "joint_tensor_query must not be None when joint_strategy is not None"
            )
        elif joint_strategy == "rear":
            query = torch.cat([query, joint_tensor_query], dim=1)
        elif joint_strategy == "front":
            query = torch.cat([joint_tensor_query, query], dim=1)
        else:
            pass

        ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
        ulysses_rank = torch.distributed.get_rank(self.ulysses_pg)
        attn_heads_per_ulysses_rank = joint_tensor_key.shape[-2] // ulysses_world_size
        joint_tensor_key = joint_tensor_key[
            ...,
            attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank
            * (ulysses_rank + 1),
            :,
        ]
        joint_tensor_value = joint_tensor_value[
            ...,
            attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank
            * (ulysses_rank + 1),
            :,
        ]

        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx
            )

        out = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_layer=attn if self.use_kv_cache else None,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output


class hetuDiTFluxLongContextAttention(hetuDiTLongContextAttention):
    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query,
        joint_tensor_key,
        joint_tensor_value,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="front",
    ) -> Tensor:
        """
        Forward pass for hetuDiTFluxLongContextAttention.
        Args:
            attn: Attention layer.
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            joint_tensor_query: Joint query tensor.
            joint_tensor_key: Joint key tensor.
            joint_tensor_value: Joint value tensor.
            dropout_p: Dropout probability.
            softmax_scale: Softmax scaling factor.
            causal: Whether to use causal attention.
            window_size: Window size for attention.
            alibi_slopes: Alibi slopes for attention.
            deterministic: Whether to use deterministic computation.
            return_attn_probs: Whether to return attention probabilities.
            joint_strategy: Joint strategy for query/key/value.
        Returns:
            output: Context output tensor.
        """

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1

        self.use_kv_cache = get_sequence_parallel_world_size() > 1
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        query = torch.cat([joint_tensor_query, query], dim=1)
        ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
        ulysses_rank = torch.distributed.get_rank(self.ulysses_pg)
        attn_heads_per_ulysses_rank = joint_tensor_key.shape[-2] // ulysses_world_size
        joint_tensor_key = joint_tensor_key[
            ...,
            attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank
            * (ulysses_rank + 1),
            :,
        ]
        joint_tensor_value = joint_tensor_value[
            ...,
            attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank
            * (ulysses_rank + 1),
            :,
        ]

        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx
            )

        out = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_layer=attn if self.use_kv_cache else None,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output
