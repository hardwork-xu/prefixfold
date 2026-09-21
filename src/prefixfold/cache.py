"""Owned shared prefixes and bounded suffix storage / 共享前缀与有界后缀存储。"""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer / 必须是整数")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} out of range / 超出允许范围")
    return int(value)


def _array(value: Any, name: str, ndim: int) -> FloatArray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != ndim:
        raise ValueError(
            f"{name} must be a {ndim}D float32 ndarray / 必须是指定维数的 float32 数组"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf / 包含非有限数值")
    return value


class SharedPrefix:
    """Immutable owned K/V, shape (heads, tokens, dim), FP32.

    不可变、自持有的 FP32 前缀；构造时复制输入，同一对象可供多个批次复用。
    Positional encoding and exact prefix equality are the caller's responsibility.
    位置编码与前缀完全相同的语义由调用者保证。
    """

    def __init__(self, k: FloatArray, v: FloatArray) -> None:
        k, v = _array(k, "prefix_k", 3), _array(v, "prefix_v", 3)
        if k.shape != v.shape or k.shape[0] < 1 or k.shape[2] < 1:
            raise ValueError(
                "prefix shapes must agree; heads/dim > 0 / 前缀形状须一致，头数和维度须为正"
            )
        self._k = np.array(k, dtype=np.float32, order="C", copy=True)
        self._v = np.array(v, dtype=np.float32, order="C", copy=True)
        self._k.flags.writeable = False
        self._v.flags.writeable = False

    @property
    def shape(self) -> tuple[int, int, int]:
        """(heads, prefix tokens, dim) /（头数、前缀长度、维度）。"""
        h, p, d = self._k.shape
        return int(h), int(p), int(d)

    @property
    def nbytes(self) -> int:
        """Allocated K/V payload bytes, not RSS / 已分配 K/V 字节，不是进程内存。"""
        return self._k.nbytes + self._v.nbytes


class DecodeBatch:
    """Ragged suffixes sharing one prefix; all mutations are atomic under a lock.

    一个前缀对应多个不等长后缀；变更在锁内先验证再写入。固定容量不自动扩张。
    The batch holds the prefix alive. close() drops this ownership and suffix buffers.
    批次持有前缀引用；close() 释放该引用及后缀数组，不会关闭其他批次。
    """

    def __init__(self, prefix: SharedPrefix, batch_size: int, capacity: int) -> None:
        if not isinstance(prefix, SharedPrefix):
            raise ValueError("prefix must be SharedPrefix / prefix 必须为 SharedPrefix")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.capacity = _positive_int(capacity, "capacity", allow_zero=True)
        self.heads, _, self.dim = prefix.shape
        self._prefix: SharedPrefix | None = prefix
        self._k = np.zeros((self.batch_size, self.heads, capacity, self.dim), dtype=np.float32)
        self._v = np.zeros_like(self._k)
        self._lengths = np.zeros(self.batch_size, dtype=np.int64)
        self._lock = RLock()
        self._closed = False

    def _ensure_open(self) -> SharedPrefix:
        if self._closed or self._prefix is None:
            raise RuntimeError("batch is closed / 批次已关闭")
        return self._prefix

    @property
    def lengths(self) -> npt.NDArray[np.int64]:
        """Copy of suffix lengths / 返回后缀长度的副本。"""
        with self._lock:
            self._ensure_open()
            return self._lengths.copy()

    @property
    def kv_nbytes(self) -> int:
        """Prefix + reserved suffix K/V + lengths, in bytes; not process RSS.

        前缀、预留后缀和长度数组的已分配字节。同一前缀跨批次求和时勿重复计数。
        """
        with self._lock:
            prefix = self._ensure_open()
            return prefix.nbytes + self._k.nbytes + self._v.nbytes + self._lengths.nbytes

    def load_suffix(
        self,
        k: FloatArray,
        v: FloatArray,
        lengths: npt.NDArray[np.integer[Any]] | None = None,
    ) -> None:
        """Replace all suffixes from (B,H,S,D) arrays, optionally with lengths (B,).

        从数组替换后缀；0 <= lengths <= S <= capacity。错误时旧状态不变。
        """
        with self._lock:
            self._ensure_open()
            k, v = _array(k, "suffix_k", 4), _array(v, "suffix_v", 4)
            if (
                k.shape != v.shape
                or k.shape[0] != self.batch_size
                or k.shape[1] != self.heads
                or k.shape[3] != self.dim
                or k.shape[2] > self.capacity
            ):
                raise ValueError("suffix shape/capacity mismatch / 后缀形状或容量不匹配")
            size = k.shape[2]
            valid: npt.NDArray[np.int64]
            if lengths is None:
                valid = np.full(self.batch_size, size, dtype=np.int64)
            else:
                if (
                    not isinstance(lengths, np.ndarray)
                    or lengths.shape != (self.batch_size,)
                    or not np.issubdtype(lengths.dtype, np.integer)
                    or np.any(lengths < 0)
                    or np.any(lengths > size)
                ):
                    raise ValueError("invalid suffix lengths / 后缀长度非法")
                valid = lengths.astype(np.int64, copy=True)
            self._k[:, :, :size] = k
            self._v[:, :, :size] = v
            # Unused padding is semantically absent; zero it before any later BLAS read.
            # 未使用的填充不属于有效缓存，清零以防后续矩阵运算受极大填充值影响。
            for row, length in enumerate(valid):
                self._k[row, :, int(length) :] = 0
                self._v[row, :, int(length) :] = 0
            self._lengths[:] = valid

    def append(
        self,
        k: FloatArray,
        v: FloatArray,
        indices: npt.NDArray[np.integer[Any]] | Sequence[int] | None = None,
    ) -> None:
        """Append one token per selected row, inputs (N,H,D); reject duplicates/overflow.

        向所选行各追加一个 token；索引须唯一且容量充足，错误时整个追加不生效。
        """
        with self._lock:
            self._ensure_open()
            k, v = _array(k, "append_k", 3), _array(v, "append_v", 3)
            rows: npt.NDArray[np.int64]
            if indices is None:
                rows = np.arange(self.batch_size, dtype=np.int64)
            else:
                indices = np.asarray(indices)
                if (
                    not isinstance(indices, np.ndarray)
                    or indices.ndim != 1
                    or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0)
                    or np.any(indices >= self.batch_size)
                ):
                    raise ValueError("invalid append indices / 追加索引非法")
                rows = indices.astype(np.int64, copy=True)
            if len(np.unique(rows)) != len(rows):
                raise ValueError("duplicate append indices / 追加索引重复")
            if k.shape != v.shape or k.shape != (len(rows), self.heads, self.dim):
                raise ValueError("append shape mismatch / 追加形状不匹配")
            positions = self._lengths[rows]
            if np.any(positions >= self.capacity):
                raise ValueError("suffix capacity exceeded / 超出后缀容量")
            self._k[rows, :, positions, :] = k
            self._v[rows, :, positions, :] = v
            self._lengths[rows] += 1

    def materialize(self) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]:
        """Copy contiguous dense K/V plus valid-token mask; explicitly allocates duplicates.

        返回连续稠密 K/V 副本及有效 token 掩码；此接口显式复制共享前缀。
        """
        with self._lock:
            prefix = self._ensure_open()
            p = prefix.shape[1]
            s = int(self._lengths.max())
            shape = (self.batch_size, self.heads, p + s, self.dim)
            k, v = np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32)
            k[:, :, :p], v[:, :, :p] = prefix._k, prefix._v
            k[:, :, p:], v[:, :, p:] = self._k[:, :, :s], self._v[:, :, :s]
            mask = np.arange(p + s)[None, :] < (p + self._lengths[:, None])
            return k, v, mask

    def close(self) -> None:
        """Idempotently release owned buffers and this prefix reference / 幂等释放资源。"""
        with self._lock:
            self._closed = True
            self._prefix = None
            self._k = np.empty((0, 0, 0, 0), dtype=np.float32)
            self._v = np.empty_like(self._k)
            self._lengths = np.empty(0, dtype=np.int64)

    def __enter__(self) -> DecodeBatch:
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
