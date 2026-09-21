# Code walkthrough

[简体中文](../zh/WALKTHROUGH.md) · [API](API.md) · [Architecture](ARCHITECTURE.md)

I would start maintaining this project by tracing one small request through ownership, validation, reduction, and comparison with the reference. The useful question at each boundary is which invariant the next function may assume.

## 1. Enter through the public package

[__init__.py](../../src/prefixfold/__init__.py) exports `SharedPrefix`, `DecodeBatch`, `AttentionPlan`, `attend`, and `dense_reference`. The complete example in [API](API.md) uses only these names. The command-line entrypoint declared by [pyproject.toml](../../pyproject.toml) calls `prefixfold.cli:main`; the default demo eventually constructs the same cache objects and calls the same `attend` function. There is one numerical implementation for both languages and interfaces.

## 2. Establish owned, valid storage

In [cache.py](../../src/prefixfold/cache.py), `_array` rejects a wrong ndarray dtype or rank and scans for nonfinite values. `_positive_int` rejects booleans as dimensions even though Python makes `bool` a subclass of `int`.

`SharedPrefix.__init__` checks shape agreement and positive head count/dimension, takes contiguous copies, and marks them read-only. At this point the prefix's public contract is independent of the caller's original buffers. Empty prefix length is permitted because a later private suffix may supply all valid keys.

`DecodeBatch.__init__` preallocates suffix K/V and lengths. Read `load_suffix` next: it validates both K/V and every length before changing the batch. Then inspect `append`: validate selected rows and shapes, gather each next position, reject any full row, write K/V, and increment lengths. The key invariant is `0 <= lengths[i] <= capacity` for every request.

The most revealing cache tests are in [test_cache.py](../../tests/test_cache.py): changing constructor inputs must not change owned data; returned copies must not mutate the cache; an overflow in one request must not partially append another request; closing one batch must not invalidate another batch sharing its prefix.

## 3. Validate the query and choose a tile

In [attention.py](../../src/prefixfold/attention.py), `attend` first checks the batch, plan, and boolean grouping flag. Within the batch lock, `_validate_query` checks `(B,H,D)`, float32, and finite inputs. If the prefix is empty and any row also has an empty suffix, normalized attention is undefined and raises `ValueError`.

`AttentionPlan.effective_tile` converts the scratch estimate into a tile width. Inspect the estimate before assuming a memory limit: it models explicit temporary arrays and excludes input/output storage and BLAS internals. The requested tile is a ceiling, not a promise that every tile has that many positions.

## 4. Follow the online state

For each head, `attend` initializes `m=-inf`, `total=0`, and `numerator=0`. Prefix tiles use `queries @ prefix_keys.T`. Suffix tiles use per-request batched products and length masks. `_update` is the numerical center:

1. Find the maximum of the old state and the new tile's row maximum.
2. Rescale prior mass and numerator by `exp(old_max - new_max)`.
3. Subtract the new maximum from tile scores and exponentiate in place.
4. Add the tile's probability mass and probability-weighted values.

After processing a set of valid positions $A$, the state must represent

$$m=\max_{j\in A}x_j,\quad total=\sum_{j\in A}e^{x_j-m},\quad numerator=\sum_{j\in A}e^{x_j-m}v_j.$$

The final division `numerator / total` produces attention. Empty masked contributions have zero mass. The finite check and `np.errstate` make float32 overflow visible even when all input values were finite. A failed `attend` has allocated temporary output but has not mutated the cache.

The grouping switch changes prefix request width from `B` to `1`. The suffix loop is outside the grouping loop and uses the same batched computation in both modes. The ablation therefore compares prefix matrix grouping and its associated loop overhead without changing suffix batching.

## 5. Use independent correctness checks

`dense_reference` concatenates the valid prefix/suffix for one request and head at a time, converts to float64, computes a conventional stable softmax, and writes float64 output. It avoids the optimized online-state path. It is a correctness oracle, not a fair high-performance baseline.

[test_attention.py](../../tests/test_attention.py) also constructs an oracle from `materialize` and its mask. It checks tiny and irregular dimensions, ragged lengths, different tile partitions, constant values, query-zero means, key-shift invariance, repeated reads, and float32 score overflow. The practical performance comparison uses the benchmark's CPU PyTorch SDPA path; its results must not be replaced with timings of this Python reference.

## 6. Debug a mismatch in a small shape

From the repository root after installing dependencies:

```sh
uv run --frozen pytest tests/test_attention.py -q
uv run --frozen pytest tests/test_cache.py tests/test_validation.py -q
uv run --frozen pytest tests/test_attention.py -k ragged -x --pdb
```

For a new failure, save the RNG seed, `(B,H,P,S,D)`, actual `lengths`, plan, and grouping flag. First compare `materialize()` and its mask with intended inputs. Then compare `attend(q, batch, AttentionPlan(tile_tokens=1))` with `dense_reference(q, batch)`. If tile size changes the error, inspect rescaling and masking around the first differing tile; if append changes untouched rows, inspect indices and the pre-write capacity check.

Do not repair a numerical failure by simply widening tolerance. Determine whether the cause is a wrong mask, a broken state invariant, valid floating-point reordering, or finite-range overflow. Re-run the affected numerical and integration checks after a change, and re-run benchmarks when a measured execution path changes.
