"""Grouped exact decode with online softmax / 分组精确解码与在线 softmax。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .cache import DecodeBatch, FloatArray, _array, _positive_int


@dataclass(frozen=True)
class AttentionPlan:
    """Scratch planning knobs; budget excludes inputs/output and vendor BLAS workspace.

    控制分块及显式数组临时空间预算，不含输入、输出与 BLAS 内部工作区。
    """

    tile_tokens: int = 1024
    workspace_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        _positive_int(self.tile_tokens, "tile_tokens")
        _positive_int(self.workspace_bytes, "workspace_bytes")

    def effective_tile(self, batch_size: int, dim: int) -> int:
        """Conservative live-array estimate: 4B(4T + 8D + 32) bytes.

        用保守显式数组估算裁剪 T；无法容纳一个 token 时拒绝执行。
        """
        b, d = _positive_int(batch_size, "batch_size"), _positive_int(dim, "dim")
        available = self.workspace_bytes // (4 * b) - 8 * d - 32
        if available < 4:
            raise ValueError("workspace too small for one tile / 临时空间不足以容纳一个分块")
        return min(self.tile_tokens, available // 4)


def _update(
    scores: FloatArray,
    values: FloatArray,
    m: FloatArray,
    total: FloatArray,
    numerator: FloatArray,
) -> None:
    # State (m,l,z) obeys l=sum exp(x-m), z=sum exp(x-m)v.
    # 状态不变量：(m,l,z) 为最大值、平移指数和、加权值和；使用新最大值统一尺度。
    new_m = np.maximum(m, np.max(scores, axis=-1))
    safe_m = np.where(np.isfinite(new_m), new_m, np.float32(0))
    alpha = np.exp(m - safe_m)
    scores -= safe_m[:, None]
    np.exp(scores, out=scores)
    total *= alpha
    total += scores.sum(axis=-1)
    numerator *= alpha[:, None]
    if values.ndim == 2:
        numerator += scores @ values
    else:
        numerator += np.matmul(scores[:, None, :], values)[:, 0, :]
    m[:] = new_m


def _validate_query(q: FloatArray, batch: DecodeBatch) -> FloatArray:
    q = _array(q, "q", 3)
    if q.shape != (batch.batch_size, batch.heads, batch.dim):
        raise ValueError("query shape must be (B,H,D) / query 形状必须为 (B,H,D)")
    prefix = batch._ensure_open()
    if prefix.shape[1] == 0 and np.any(batch._lengths == 0):
        raise ValueError("each query needs at least one key / 每个 query 至少需要一个 key")
    return q


def attend(
    q: FloatArray,
    batch: DecodeBatch,
    plan: AttentionPlan | None = None,
    *,
    grouped: bool = True,
) -> FloatArray:
    """FP32 exact one-token decode. Query/output (B,H,D); ragged suffixes are masked.

    FP32 单 token 精确解码，支持不等长后缀；grouped=False 关闭跨请求前缀矩阵合并。
    Validation/overflow raises ValueError/FloatingPointError. Closed batches raise RuntimeError.
    参数或溢出抛出明确异常；批次锁贯穿计算，追加与读取不会交错。
    """
    if not isinstance(batch, DecodeBatch):
        raise ValueError("batch must be DecodeBatch / batch 必须为 DecodeBatch")
    if not isinstance(grouped, bool):
        raise ValueError("grouped must be bool / grouped 必须为布尔值")
    if plan is not None and not isinstance(plan, AttentionPlan):
        raise ValueError("plan must be AttentionPlan / plan 必须为 AttentionPlan")
    with batch._lock, np.errstate(over="raise", invalid="raise", divide="raise"):
        q = _validate_query(q, batch)
        prefix = batch._ensure_open()
        selected = plan or AttentionPlan()
        tile = selected.effective_tile(batch.batch_size, batch.dim)
        output = np.empty_like(q)
        scale = np.float32(1 / np.sqrt(batch.dim))
        width = batch.batch_size if grouped else 1
        for h in range(batch.heads):
            m = np.full(batch.batch_size, -np.inf, dtype=np.float32)
            total = np.zeros(batch.batch_size, dtype=np.float32)
            numerator = np.zeros((batch.batch_size, batch.dim), dtype=np.float32)
            for first in range(0, batch.batch_size, width):
                last = min(first + width, batch.batch_size)
                queries = q[first:last, h]
                for start in range(0, prefix.shape[1], tile):
                    stop = start + tile
                    scores = queries @ prefix._k[h, start:stop].T
                    scores *= scale
                    if not np.isfinite(scores).all():
                        raise FloatingPointError("nonfinite attention logits / 注意力分数溢出")
                    _update(
                        scores,
                        prefix._v[h, start:stop],
                        m[first:last],
                        total[first:last],
                        numerator[first:last],
                    )
            # Suffix path is identical in both modes, isolating the prefix grouping ablation.
            # 两种模式的后缀路径相同，使消融仅改变共享前缀的跨请求分组。
            queries = q[:, h]
            lengths = batch._lengths
            for start in range(0, int(lengths.max()), tile):
                stop = min(start + tile, int(lengths.max()))
                keys = batch._k[:, h, start:stop]
                scores = np.matmul(queries[:, None, :], keys.swapaxes(-1, -2))[:, 0, :]
                scores *= scale
                if not np.isfinite(scores).all():
                    raise FloatingPointError("nonfinite attention logits / 注意力分数溢出")
                scores[np.arange(start, stop)[None, :] >= lengths[:, None]] = -np.inf
                _update(scores, batch._v[:, h, start:stop], m, total, numerator)
            output[:, h] = numerator / total[:, None]
        if not np.isfinite(output).all():
            raise FloatingPointError("nonfinite attention output / 注意力输出溢出")
        return output


def dense_reference(q: FloatArray, batch: DecodeBatch) -> npt.NDArray[np.float64]:
    """Float64 stable dense softmax oracle, one request/head at a time.

    float64 稠密参考实现；逐请求逐头构造，避免同时复制整个批次的前缀。
    """
    with batch._lock:
        q = _validate_query(q, batch)
        prefix = batch._ensure_open()
        result = np.empty(q.shape, dtype=np.float64)
        for b in range(batch.batch_size):
            for h in range(batch.heads):
                s = int(batch._lengths[b])
                keys = np.concatenate((prefix._k[h], batch._k[b, h, :s])).astype(np.float64)
                values = np.concatenate((prefix._v[h], batch._v[b, h, :s])).astype(np.float64)
                logits = keys @ q[b, h].astype(np.float64) / np.sqrt(batch.dim)
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                result[b, h] = probabilities @ values
        return result
