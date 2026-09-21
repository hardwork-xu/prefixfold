# Experiments

[简体中文](../zh/EXPERIMENTS.md) · [Research](RESEARCH.md) · [Raw main run](../../results/benchmark.json)

I kept a fixed primary workload and published every configured case, including cases where PrefixFold is slower. The experiment asks whether an established shared-prefix decomposition is useful on this CPU; it does not evaluate a language model or generation service.

## Frozen targets and observed verdict

[targets.json](../../research/targets.json) and [benchmark.json](../../configs/benchmark.json) set the targets before tuning: float32, `B=16,H=4,P=4096,S=128,D=64`, five warmups, 30 measured calls per variant, tile 1,024, explicit scratch budget 8 MiB. The batch has one query per request/head.

| Metric | Target, fixed before main run | Measured on primary workload | Verdict |
|---|---:|---:|---|
| Owned K/V array payload reduction against dense duplicated KV | At least 80% | 90.9091%, 132 MiB to 12 MiB | Met |
| SDPA median / PrefixFold median | At least 1.25× | 3.5036×, 7.1603 ms / 2.0437 ms | Met |
| All workload/variant correctness checks | `atol=3e-5, rtol=3e-5` vs float64 | 28/28 passed | Met |

**Expected/theoretical:** storing the common prefix once reduces its payload by a factor of `B`; arithmetic complexity is unchanged. **Measured:** the actual K/V arrays used by timed variants occupy the byte counts above. Neither statement establishes a reduction of the same percentage in process memory. The benchmark excludes the int64 lengths array from K/V payload, whereas the public `DecodeBatch.kv_nbytes` property includes it.

## Environment and provenance

The main run used Apple M1 Pro, 10 logical CPUs, 16 GiB physical memory, Darwin 25.6.0 arm64, Python 3.12.2, NumPy 2.2.6, PyTorch 2.8.0, and PrefixFold 0.1.0. Both installed library build configurations identify Apple Accelerate BLAS. PyTorch's recorded intra-op thread count and the visible OpenMP pool were 1. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` requested 1; `threadpoolctl` cannot observe whether Accelerate honors its setting. These are **one-thread-requested CPU results**, not proof that every internal BLAS operation used exactly one OS thread.

Run `20260921T121914Z-56a3bc9a` started at `2026-09-21T12:19:14.085647+00:00` and completed at `2026-09-21T12:19:45.075223+00:00`. Tested Git revision: `70e7ed50e51b4de74a9d35c358068aa689a939c5`. The tested source was clean. Its source manifest SHA-256 is `c0003d3eaf2887a4b8a026e9139ea2cc783d76487fa219f9bdaeac15c9b7d500`; per-file hashes and configuration hash are in the raw record. This identifies the measured source even after later documentation commits.


The original main run and three tile runs from source `99e2958` are retained in [results/pre-resource-fix](../../results/pre-resource-fix), including all 1,200 measurements. A resource audit found that the scratch estimator did not conservatively cover integer index/mask temporaries for small dimensions. The estimate changed from `4B(3T+8D+32)` to `4B(4T+8D+32)` before this final source revision. Original targets, workload configurations, seeds and actual effective tiles remained unchanged. The complete main suite and all three sweeps were rerun, producing the 1,200 final measurements referenced here. Differences between the two sessions must not be interpreted as an optimization gain from this resource-accounting fix.

All Q/K/V arrays are **synthetic seeded normal float32 data**; there is no pretrained model, private dataset, or captured production trace. Base seed is `20260921`; workload index is added to it, making the primary fixture seed `20260925`. All four variants of a workload use the same input seed and shapes. Ragged lengths are saved explicitly.

## Baselines and timing protocol

| Variant | Implemented work and purpose |
|---|---|
| `torch_sdpa` | CPU `torch.nn.functional.scaled_dot_product_attention`, dense contiguous KV, boolean validity mask, dropout zero; practical library baseline |
| `numpy_dense` | Batched NumPy matrix products, stable softmax, same mask and float32 arithmetic; practical second baseline with the same array library as PrefixFold |
| `prefixfold` | Shared-prefix grouped matrix multiplication, tiled online reduction, private-suffix batching, validated public API |
| `ungrouped` | Same prefix ownership, plan and private-suffix code; only prefix queries are processed one request at a time |

The float64 `dense_reference` is a correctness oracle, not a performance competitor. PyTorch selects its CPU implementation; this is not CUDA FlashAttention. Differences in library dispatch, fusion, vectorization, and Python overhead remain even when both builds use Accelerate. PrefixFold's timed public call includes query validation, plan selection and its batch lock; the dense baseline closures have already validated/generated their inputs. The comparison therefore does not isolate machine instructions or assume identical implementation overhead.

Each case creates a separate persistent worker process per variant. Workers retain identical imports, including NumPy and PyTorch. The parent asks only one worker to execute at a time; other workers remain idle. Initial worker order and every warmup/measured round are seeded and shuffled. A worker times synchronous CPU computation with `perf_counter_ns`, so process communication and scheduling the request are outside the recorded interval. All valid observations are retained; the main suite contains **7 cases × 4 variants × 30 calls = 840 measured samples**, plus 140 warmups and 28 first calls. It is one session on one machine, not 840 independent hardware trials.

Fixture creation, cache construction/loading, dense KV materialization and tensor views are measured separately. First-call timing is the first compute after setup; it excludes Python process startup and import time. Correctness comparison and warmup precede steady measurements. The compute path includes its output allocation and NumPy/PyTorch return conversion where present. There is no model loading, preprocessing, host-device transfer, token generation, or full application request in the timed scope. **These are operator timings, not end-to-end generation latency.**

The benchmark rejects configurations with one dense KV pair larger than 1 GiB. Workers have a response timeout, and errors/timeouts are recorded with status and error text; they are not converted to zero latency. All 28 main variant records passed with 30 samples and successful worker exits. No main case failed, timed out, or was skipped.

## All main workloads

All rows below use `H=4,D=64`; latencies are medians in milliseconds. `S*` denotes reserved ragged suffix capacity, not equal live lengths.

| Case `(B,P,S)` | NumPy ms | SDPA ms | PrefixFold ms | Ungrouped ms | SDPA / PrefixFold | KV reduction |
|---|---:|---:|---:|---:|---:|---:|
| `no_prefix` `(16,0,128)` | 0.2029 | 0.2480 | 0.4081 | 0.4193 | 0.608× | 0.00% |
| `single_request` `(1,4096,128)` | 0.4663 | 0.5357 | 0.8819 | 0.8724 | 0.607× | 0.00% |
| `short_prefix` `(16,128,128)` | 0.4352 | 0.4766 | 0.5631 | 1.3192 | 0.846× | 46.88% |
| `shared_b4_p4096` `(4,4096,128)` | 1.6612 | 1.8942 | 1.0748 | 2.0978 | 1.762× | 72.73% |
| `shared_b16_p4096` `(16,4096,128)` | 6.6804 | 7.1603 | 2.0437 | 7.2138 | 3.504× | 90.91% |
| `shared_b32_p8192` `(32,8192,128)` | 26.2760 | 28.0105 | 4.8808 | 27.1801 | 5.739× | 95.38% |
| `ragged_suffix` `(16,4096,256*)` | 6.7725 | 7.1991 | 2.5668 | 7.1642 | 2.805× | 88.21% |

The ragged case reserves 256 suffix positions per request and actually uses lengths `[29,80,230,231,225,205,46,39,203,248,179,136,234,184,202,98]`. Its dense baseline materializes up to the maximum live length, 248; PrefixFold retains the full reserved capacity. This difference is included in the reported payload reduction.

[Full 28-row table](../../results/table_en.md), [summary.json](../../results/summary.json), and [latency plot](../../results/latency.svg) are generated by [scripts/analyze.py](../../scripts/analyze.py) from raw measurements. The plot uses a logarithmic latency axis. Throughput fields count query vectors (`B*H / seconds`), not generated tokens.

On the fixed primary case, grouping reduces the ungrouped median from 7.2138 to 2.0437 ms, a 3.5298× ratio, with identical stored KV and suffix computation. Relative to NumPy dense, the primary ratio is 3.2688×. These comparisons support prefix query grouping as the useful mechanism in this workload. They do not prove that data traffic fell by the same factor; no hardware performance counter measured memory traffic.

The unfavorable cases matter: no prefix reaches only 0.608× of SDPA speed, a single request 0.607×, and a short prefix 0.846×. Storage sharing provides no reduction for the first two, and even a 46.875% payload reduction in the short-prefix case does not yield faster attention. Python dispatch, temporary arrays, and reduction work can dominate. When `P=0` or `B=1`, grouped and ungrouped arithmetic paths are equivalent; their observed timing differences are noise and process/runtime effects, not evidence of a grouping improvement.

## Setup, cold calls, memory and variability

The following measurements are from the primary workload. They are separate timing scopes and must not be presented as a measured full request by summing their medians.

| Variant | Common fixture ms | Variant setup ms | First compute ms | Steady median ms | Whole-worker peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| `prefixfold` | 17.8804 | 0.0772 | 2.7534 | 2.0437 | 243.031 |
| `ungrouped` | 16.2873 | 0.0164 | 7.7288 | 7.2138 | 223.656 |
| `torch_sdpa` | 15.3498 | 13.6900 | 9.5335 | 7.1603 | 370.844 |
| `numpy_dense` | 14.8978 | 12.2113 | 9.9379 | 6.6804 | 380.812 |

`peak_worker_rss_bytes` is the operating system's process high-water mark over imports, fixtures, baseline materialization, float64 correctness, and timed calls. Baseline workers release the shared source before steady timing, but its earlier allocation still affects peak RSS. RSS is not attention scratch, an incremental allocation measurement, or the total of all simultaneously resident workers. The 90.9091% claim uses K/V array payload only. `AttentionPlan`'s 8 MiB budget is an estimate excluding input/output and vendor workspace, not a process hard limit.

Primary PrefixFold sample range is 1.4868–2.4598 ms, sample standard deviation 0.3023 ms; SDPA is 6.9532–7.4807 ms with standard deviation 0.1406 ms. The ragged case has a PrefixFold range of 1.8255–3.3839 ms. The largest case contains PrefixFold and SDPA observations of 13.7206 ms and 72.2465 ms. Those samples remain in the data. Thirty observations per variant do not justify p99 or strong tail-latency claims. No confidence interval, thermal stabilization, CPU pinning, or controlled multi-session repeatability claim is made.

All main PrefixFold cases passed the declared mixed absolute/relative tolerance; maximum absolute error across them is `2.0915591120163057e-7`, and the primary case is `1.0125361699087065e-7`. The raw maximum relative error can exceed `3e-5` near a nearly zero reference output; the acceptance rule is `abs(actual-reference) <= atol + rtol*abs(reference)`, not a relative-only bound.

## Post-primary tile sensitivity

After retaining the fixed main run, three additional primary-shaped runs tested tiles 256, 1,024 and 4,096. All use fixture seed `20260921`, so they share inputs with one another but differ from the main run's primary fixture. They use the same tested source hash, four variants, five warmups and 30 samples per variant, contributing another 360 measured samples. Every variant passed correctness. This is exploratory sensitivity, not a replacement of the frozen 1,024-token main result.

| Tile | PrefixFold median ms | Ungrouped median ms | SDPA median ms | NumPy median ms | Raw record |
|---:|---:|---:|---:|---:|---|
| 256 | 2.7249 | 15.9835 | 7.2366 | 6.8120 | [tile256.json](../../results/tile256.json) |
| 1,024 | 2.3479 | 7.2505 | 7.1777 | 6.6577 | [tile1024.json](../../results/tile1024.json) |
| 4,096 | 1.9245 | 4.2925 | 7.1430 | 6.5834 | [tile4096.json](../../results/tile4096.json) |

The larger tile was faster here, especially for the ungrouped loop, consistent with amortizing dispatch/reduction overhead. It uses more score workspace. The runs executed in tile order 256, then 1,024, then 4,096; tile order was not randomized, although variant order within each run was. Separate sessions, possible runtime drift and one shape do not establish an optimal tile rule; no default or original target was changed. Configuration files are [tile256.json](../../configs/tile256.json), [tile1024.json](../../configs/tile1024.json), and [tile4096.json](../../configs/tile4096.json).

## Reproduction and conclusion scope

```sh
make install
.venv/bin/python benchmark.py --config configs/benchmark.json --output results/reproduced.json
.venv/bin/python scripts/analyze.py --input results/reproduced.json --output-dir results/reproduced-analysis
.venv/bin/python benchmark.py --config configs/tile256.json --output results/reproduced-tile256.json
.venv/bin/python benchmark.py --config configs/tile1024.json --output results/reproduced-tile1024.json
.venv/bin/python benchmark.py --config configs/tile4096.json --output results/reproduced-tile4096.json
```

The recorded main invocation is `python benchmark.py --config configs/benchmark.json --output results/benchmark.json --variants prefixfold ungrouped torch_sdpa numpy_dense --timeout 120`. The commands above use new result paths to preserve checked-in evidence. `scripts/analyze.py` writes analysis artifacts by default. It updates README benchmark sections only when explicitly passed `--update-readme`; use that flag only for the intended published main run.

The evidence supports this implementation for these synthetic CPU shapes, particularly long shared prefixes with multiple requests. It does not establish serving throughput, model output quality, GPU/MPS speed, Linux container performance, or universal CPU superiority. The documented padding-overflow bug was repaired before the measured revision and is covered by a regression test; finite valid logits that exceed float32 range are still intentionally rejected. Next useful experiments would add independent session repeats, representative real projected KV traces obtained with an explicit model integration, and a dispatch rule tested on held-out shapes.
