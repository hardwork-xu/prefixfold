# Public Python API

[简体中文](../zh/API.md) · [Architecture](ARCHITECTURE.md) · [Home](../../README.md)

Import public names from `prefixfold`. All attention input arrays are CPU NumPy arrays with exactly `dtype=np.float32`. The API does not silently cast float64, lists, or GPU tensors to float32. Noncontiguous float32 arrays are accepted. Every supplied K/V and query array is checked for NaN and infinity, including supplied suffix padding. Integer configuration parameters reject booleans.

Notation: `B` requests, `H` heads, `D` head width, `P` shared positions, `C` reserved suffix capacity, `S` supplied suffix width, and `N` selected rows. `B,H,D` are positive; `P,C,S` may be zero where the shape contract allows it.

## Complete example

```python
import numpy as np
from prefixfold import AttentionPlan, DecodeBatch, SharedPrefix, attend, dense_reference

rng = np.random.default_rng(7)
pk = rng.normal(size=(2, 17, 8)).astype(np.float32)
pv = rng.normal(size=pk.shape).astype(np.float32)
prefix = SharedPrefix(pk, pv)
q = rng.normal(size=(2, 2, 8)).astype(np.float32)
sk = rng.normal(size=(2, 2, 2, 8)).astype(np.float32)
sv = rng.normal(size=sk.shape).astype(np.float32)

with DecodeBatch(prefix, batch_size=2, capacity=3) as batch:
    batch.load_suffix(sk, sv, lengths=np.array([1, 2], dtype=np.int64))
    output = attend(q, batch, AttentionPlan(tile_tokens=7))
    np.testing.assert_allclose(output, dense_reference(q, batch), atol=3e-5, rtol=3e-5)
    batch.append(sk[:1, :, 0], sv[:1, :, 0], indices=[0])
    np.testing.assert_array_equal(batch.lengths, [2, 2])
    updated = attend(q, batch)
    np.testing.assert_allclose(updated, dense_reference(q, batch), atol=3e-5, rtol=3e-5)
    print(updated.shape, batch.kv_nbytes)

with DecodeBatch(prefix, batch_size=1, capacity=0) as other:
    assert attend(q[:1], other).shape == (1, 2, 8)
```

The second batch reuses the same prefix after the first closes. The example performs actual attention and state mutation without model downloads.

## SharedPrefix

`SharedPrefix(k, v)` creates immutable owned copies of matching `(H,P,D)` arrays. It does not infer or verify that different requests originated from the same text or model context. Prefix positional encoding and equality are the caller's responsibility.

| Member | Result or behavior |
|---|---|
| `shape` | Tuple `(H,P,D)` |
| `nbytes` | K/V array payload bytes: `8*H*P*D` for two float32 arrays |

Wrong dtype/rank, mismatched shapes, nonfinite values, or zero `H,D` raise `ValueError`. Empty `P` is allowed. Do not mutate private underscore fields. Lifetime follows Python references; there is no explicit prefix invalidation or eviction API.

## DecodeBatch

`DecodeBatch(prefix, batch_size, capacity)` references the supplied `SharedPrefix` and allocates private K/V capacity. `batch_size > 0`, `capacity >= 0`. Existing configuration attributes `batch_size`, `heads`, `dim`, and `capacity` describe the allocated layout; assigning new values is not a supported resize operation.

| Member | Result or behavior |
|---|---|
| `lengths` | New int64 array `(B,)` with current valid suffix lengths |
| `kv_nbytes` | `prefix.nbytes + 8*B*H*C*D + 8*B`; includes reserved K/V and int64 lengths |
| `load_suffix(k, v, lengths=None)` | Replace valid suffixes from `(B,H,S,D)` arrays, `S <= C`; returns `None` |
| `append(k, v, indices=None)` | Append one token per selected row from `(N,H,D)` arrays; returns `None` |
| `materialize()` | New `(dense_k, dense_v, mask)` arrays described below |
| `close()` | Idempotently release this batch's buffers and prefix reference; returns `None` |
| `with DecodeBatch(...) as batch` | Close the batch when the context exits, including on exceptions |

`kv_nbytes` measures owned array payload, not object overhead, process RSS, or peak memory. Summing it across batches that share one prefix double-counts that prefix; subtract repeated prefix payload when calculating a combined total.

`load_suffix` requires matching K/V shapes. Without `lengths`, every row has length `S`. Otherwise supply an integer NumPy array `(B,)` with `0 <= lengths[i] <= S`. Input data and lengths are copied. Unused suffix capacity is zeroed after a valid load, preventing inactive padding from affecting later arithmetic. Even ignored padding supplied by the caller must be finite. Loading `S=0` clears all suffix lengths while preserving capacity and the prefix. Validation failure leaves the old state unchanged.

`append` defaults to all rows in order, requiring `N=B`. Explicit indices accept a one-dimensional integer NumPy array or integer sequence such as `[2, 0]`; order maps input rows to destination requests. Each index must be unique and within `[0,B)`. Every selected row must have at least one free slot. All validation precedes writes, so a full row or invalid index rejects the whole append. For an empty no-op selection, use `np.empty(0, dtype=np.int64)` with K/V shape `(0,H,D)`.

`materialize` uses `L=P+max(lengths)`, producing float32 K/V `(B,H,L,D)` and boolean mask `(B,L)`. `True` means a visible key. Prefix data are explicitly duplicated across requests, and suffix padding is not a valid key. A PyTorch caller expands this mask as `(B,1,1,L)` for queries `(B,H,1,D)` and uses `is_causal=False`, since the supplied cache already contains the visible history. `materialize` is useful for interoperability and debugging but has allocation cost.

After `close`, buffer operations, `lengths`, `kv_nbytes`, and attention reject access with `RuntimeError`. Repeated `close` is permitted. Other batches sharing the prefix continue to work.

## AttentionPlan

`AttentionPlan(tile_tokens=1024, workspace_bytes=8*1024*1024)` is a frozen dataclass. Both fields must be positive integers. `effective_tile(batch_size, dim)` returns the requested tile reduced to fit the estimate `4*B*(4*T+8*D+32)` bytes. If even `T=1` does not fit, it raises `ValueError`.

The budget estimates explicit scratch arrays. It excludes cache/input arrays, final output, BLAS internals, Python overhead, and allocator effects. It is not a process hard limit or measured peak memory. A single plan can be reused for different batches; the effective tile is computed for each batch's dimensions.

## attend

`attend(q, batch, plan=None, *, grouped=True) -> ndarray[float32]`

`q` and returned output have shape `(B,H,D)`. A missing plan selects defaults. `grouped` must be a Python boolean. `True` groups requests for shared-prefix matrix multiplication; `False` evaluates the prefix per request while retaining the same suffix path. Both modes compute the same scaled dot-product attention within floating-point error. There is no automatic dispatch based on benchmark results.

Each query attends to all prefix positions and its own valid suffix. At least one key is required for every query: `P=0` with any zero suffix length raises `ValueError`. There is no dropout, future-position mask, arbitrary mask argument, grouped-query attention, or gradient calculation.

The batch lock is held for validation and the complete computation, preventing append/load/close from interleaving. The function does not mutate `q` or cache storage. Finite float32 input is necessary but not sufficient for finite intermediate arithmetic; score or accumulation overflow raises `FloatingPointError`. Nonfinite input is rejected as `ValueError`, and a closed batch raises `RuntimeError`. Allocation failure is not suppressed. The caller must avoid concurrent writes to `q`.

## dense_reference

`dense_reference(q, batch) -> ndarray[float64]`

It uses the same valid keys and query shape as `attend`, holds the batch lock, and evaluates dense stable softmax in float64 one request/head at a time. It returns `(B,H,D)` float64 output and does not mutate the batch. Use it to check numerical behavior on manageable shapes; it is not the practical performance baseline and does not obey `AttentionPlan` scratch estimates. The declared float32 comparison uses `atol=3e-5, rtol=3e-5`.

## CLI and NPZ files

`python -m prefixfold demo` or `prefixfold demo` runs deterministic synthetic data through append, grouped attention and the FP64 comparison. Success emits JSON with `status`, `shape`, `max_abs_error`, `kv_payload_bytes` and `suffix_lengths`.

`prefixfold attend input.npz output.npz --tile-tokens 1024 --workspace-mib 8 --max-input-mib 512` reads these arrays:

| NPZ key | Shape | Type |
|---|---|---|
| `q` | `(B,H,D)` | float32 |
| `prefix_k`, `prefix_v` | `(H,P,D)` | float32 |
| `suffix_k`, `suffix_v` | `(B,H,S,D)` | float32 |
| `lengths` (optional) | `(B,)`, values 0..S | integer |

Output NPZ contains `output` `(B,H,D)` float32. Omit `lengths` to use all supplied suffix positions. Archives are loaded with `allow_pickle=False`; declared decompressed data is limited before loading. Only trusted files are in scope. The output path must be new and differ from input. Errors emit a bilingual diagnostic to stderr and exit 2; success emits JSON and exits 0. Allocation failure is not converted to a fake result.
