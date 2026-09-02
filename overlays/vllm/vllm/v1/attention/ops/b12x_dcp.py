# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional B12X PCIe collectives for MLA decode-context parallelism.

This module keeps the B12X pool and CUDA-graph channel lifecycle separate from
the ordinary DCP implementations in :mod:`vllm.v1.attention.ops.dcp`.  Callers
must retain a complete fallback because B12X only accepts a bounded set of
CUDA layouts, dtypes, and DCP world sizes.

The integration is adapted from the Local Inference Lab v75 B12X DCP path by
Luke Alonso and Martin Vit.  It deliberately preserves r10's direct symmetric
memory and NCCL implementations as fallbacks.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


logger = init_logger(__name__)

_POOLS: dict[tuple[int, int, int, int, int, int], Any] = {}
_DISABLED: set[tuple[int, int, int, int, int, int]] = set()
_ACTIVE_CAPTURE: dict[int, tuple[str, Any, ExitStack]] = {}
_PROFILE_CHANNEL_CHECKPOINTS: dict[tuple[int, int, int, int, int, int], Any] = {}
_EAGER_CHANNEL_ID = "vllm:eager:dcp"
_MAX_CONCURRENT_CHANNELS = 2
_SUPPORTED_WORLD_SIZES = (2, 4, 8, 16)


def _channel_id(group: GroupCoordinator) -> str:
    active = _ACTIVE_CAPTURE.get(id(group.device_group))
    return active[0] if active is not None else _EAGER_CHANNEL_ID


def _supported_bhd_layout(tensor: torch.Tensor) -> bool:
    if tensor.ndim != 3 or int(tensor.stride(2)) != 1:
        return False
    batch, heads, head_dim = (int(value) for value in tensor.shape)
    stride_batch, stride_head, _ = (int(value) for value in tensor.stride())
    return (
        stride_batch == heads * head_dim and stride_head == head_dim
    ) or (
        stride_batch == head_dim and stride_head >= batch * head_dim
    )


@lru_cache(maxsize=1)
def _load_pool() -> Any | None:
    try:
        from b12x.comm.pcie import DcpAllToAllPool
    except Exception:
        return None
    return DcpAllToAllPool


def _init_failed(
    group: GroupCoordinator,
    device: torch.device,
    error: Exception | None,
) -> bool:
    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group.device_group)
    return bool(failed.item())


def _get_pool(
    group: GroupCoordinator,
    *,
    device: torch.device,
    total_heads: int,
    head_dim: int,
    query_head_dim: int,
    max_batch_size: int,
) -> Any | None:
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    key = (
        id(group.device_group),
        int(device_index),
        int(total_heads),
        int(head_dim),
        int(query_head_dim),
        int(max_batch_size),
    )
    if key in _DISABLED:
        return None
    pool = _POOLS.get(key)
    if pool is not None:
        return pool
    if torch.cuda.is_current_stream_capturing():
        return None
    pool_cls = _load_pool()
    if pool_cls is None:
        _DISABLED.add(key)
        return None

    pool = None
    error: Exception | None = None
    try:
        pool = pool_cls.from_exchange_group(
            exchange_group=group.device_group,
            device=device,
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
            single_channel=False,
            max_concurrent_channels=_MAX_CONCURRENT_CHANNELS,
        )
        pool.prepare_channels((_EAGER_CHANNEL_ID,))
        active = _ACTIVE_CAPTURE.get(id(group.device_group))
        if active is None:
            pool.for_stream(channel_id=_EAGER_CHANNEL_ID)
        else:
            active_channel, active_stream, active_stack = active
            if active_channel.endswith(":profile"):
                _PROFILE_CHANNEL_CHECKPOINTS[key] = pool.checkpoint_channels()
            pool.prepare_channels((active_channel,))
            active_stack.enter_context(
                pool.capture(stream=active_stream, channel_id=active_channel)
            )
    except Exception as exc:  # coordinated fallback is intentional
        error = exc

    if _init_failed(group, device, error):
        if pool is not None:
            pool.close()
        _DISABLED.add(key)
        if error is not None:
            logger.warning(
                "B12X PCIe DCP initialization failed; falling back: %s", error
            )
        return None

    assert pool is not None
    _POOLS[key] = pool
    logger.info(
        "Using B12X PCIe DCP collectives "
        "(world_size=%d, max_batch_size=%d, heads=%d, "
        "query_head_dim=%d, output_head_dim=%d).",
        group.world_size,
        max_batch_size,
        total_heads,
        query_head_dim,
        head_dim,
    )
    return pool


@contextmanager
def capture_b12x_dcp(
    group: GroupCoordinator,
    stream: torch.cuda.Stream | None = None,
    *,
    channel_id: str | None = None,
):
    """Bind all pools for ``group`` to a target- or draft-owned graph channel."""
    group_id = id(group.device_group)
    matching = sorted(
        ((key, pool) for key, pool in _POOLS.items() if key[0] == group_id),
        key=lambda item: item[0][1:],
    )
    if channel_id is None:
        if matching:
            raise RuntimeError("B12X DCP graph capture requires a channel_id")
        yield
        return

    active = _ACTIVE_CAPTURE.get(group_id)
    if active is not None:
        active_channel, active_stream, _ = active
        if channel_id != active_channel or stream is not active_stream:
            raise RuntimeError(
                "nested B12X DCP capture must reuse its channel_id and stream"
            )
        yield
        return

    try:
        with ExitStack() as stack:
            _ACTIVE_CAPTURE[group_id] = (channel_id, stream, stack)
            for key, pool in matching:
                # Profiling graphs are discarded and production graphs are
                # independently replayable.  Give each capture owner its own
                # B12X logical channel; prepare_channels is idempotent and
                # rank-validates the catalog before capture.
                if (
                    channel_id.endswith(":profile")
                    and key not in _PROFILE_CHANNEL_CHECKPOINTS
                ):
                    _PROFILE_CHANNEL_CHECKPOINTS[key] = pool.checkpoint_channels()
                pool.prepare_channels((channel_id,))
                stack.enter_context(pool.capture(stream=stream, channel_id=channel_id))
            yield
    finally:
        _ACTIVE_CAPTURE.pop(group_id, None)


def rollback_profile_b12x_dcp_channels() -> None:
    """Release channels owned only by disposable memory-profile graphs."""
    for key, checkpoint in tuple(_PROFILE_CHANNEL_CHECKPOINTS.items()):
        pool = _POOLS.get(key)
        if pool is not None:
            pool.rollback_channels(checkpoint)
        _PROFILE_CHANNEL_CHECKPOINTS.pop(key, None)


def try_lse_reduce(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    group: GroupCoordinator,
    *,
    is_lse_base_on_e: bool,
    max_batch_size: int,
    query_head_dim: int,
) -> torch.Tensor | None:
    """Return the B12X reduction result, or ``None`` for caller fallback."""
    world_size = group.world_size
    if (
        not partial_output.is_cuda
        or partial_output.dtype not in (torch.float16, torch.bfloat16)
        or partial_lse.dtype != torch.float32
        or world_size not in _SUPPORTED_WORLD_SIZES
        or partial_output.ndim != 3
        or partial_lse.shape != partial_output.shape[:2]
    ):
        return None
    batch, total_heads, head_dim = partial_output.shape
    if (
        batch < 1
        or batch > max_batch_size
        or total_heads % world_size != 0
        or head_dim % 8 != 0
        or query_head_dim <= 0
        or query_head_dim % 8 != 0
    ):
        return None

    pool = _get_pool(
        group,
        device=partial_output.device,
        total_heads=total_heads,
        head_dim=head_dim,
        query_head_dim=query_head_dim,
        max_batch_size=max_batch_size,
    )
    if pool is None:
        return None
    if not _supported_bhd_layout(partial_output):
        partial_output = partial_output.contiguous()
    if not partial_lse.is_contiguous():
        partial_lse = partial_lse.contiguous()
    reduced_storage = torch.empty(
        (total_heads // world_size, batch, head_dim),
        device=partial_output.device,
        dtype=partial_output.dtype,
    )
    reduced = reduced_storage.transpose(0, 1)
    return pool.lse_reduce_scatter(
        partial_output,
        partial_lse,
        out=reduced,
        is_lse_base_on_e=is_lse_base_on_e,
        channel_id=_channel_id(group),
    )


def try_all_gather_heads(
    local_input: torch.Tensor,
    group: GroupCoordinator,
    *,
    max_batch_size: int,
    output_head_dim: int,
) -> torch.Tensor | None:
    """Return gathered query heads, or ``None`` for caller fallback."""
    world_size = group.world_size
    if (
        not local_input.is_cuda
        or local_input.dtype
        not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        or world_size not in _SUPPORTED_WORLD_SIZES
        or local_input.ndim != 3
    ):
        return None
    if not local_input.is_contiguous():
        local_input = local_input.contiguous()
    batch, local_heads, head_dim = local_input.shape
    alignment = 16 if local_input.dtype == torch.float8_e4m3fn else 8
    if (
        batch < 1
        or batch > max_batch_size
        or local_heads <= 0
        or head_dim % alignment != 0
        or output_head_dim <= 0
        or output_head_dim % 8 != 0
    ):
        return None
    pool = _get_pool(
        group,
        device=local_input.device,
        total_heads=local_heads * world_size,
        head_dim=output_head_dim,
        query_head_dim=head_dim,
        max_batch_size=max_batch_size,
    )
    if pool is None:
        return None
    return pool.all_gather_heads(local_input, channel_id=_channel_id(group))
