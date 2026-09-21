# Project presentation and evidence

[简体中文](../zh/RESUME.md) · [Experiments](EXPERIMENTS.md) · [Research](RESEARCH.md)

I maintain PrefixFold as a focused systems artifact: the implementation, failure cases, and measured limits should be inspectable together. The descriptions below concern the repository's verified behavior. They make no claim of deployment, user adoption, published research, or a completed personal learning process.

## One-sentence description

**PrefixFold** is a typed Python/NumPy implementation of exact shared-prefix decode attention on CPUs, pairing reusable KV ownership and online softmax with reproducible latency, payload, and correctness experiments.

## Resume bullets

1. Implemented CPU shared-prefix decode attention using grouped matrix multiplication and stable tiled softmax reduction; on Apple M1 Pro with one thread requested, a fixed float32 case (`B=16,H=4,D=64`, 4,096 shared + 128 private positions) reduced median operator latency from **7.1603 ms with PyTorch 2.8 SDPA to 2.0437 ms**, a **3.5036×** ratio over 30 calls per variant.
2. Engineered immutable shared-prefix ownership and fixed-capacity ragged suffix storage with validated atomic appends and locking across complete attention calls; on the same workload, the actual K/V array payload was **12 MiB versus 132 MiB** for duplicated dense KV, a **90.9091% reduction**, with reusable prefixes across independently closed batches.
3. Built a reproducible CPU benchmark with **4 variants, 7 workloads and 840 raw timing samples**, source/configuration hashes, randomized variant order and separate setup/cold/RSS records; all 28 variant/workload combinations passed `atol=3e-5, rtol=3e-5` against float64, while three workloads where the optimized path was slower than SDPA remained in the published results.
4. Added offline numerical and lifecycle regression tests, a CLI and typed public API, locked dependency setup, and aligned English/Chinese maintainer documents; verified grouping against an unchanged suffix path and investigated tile sizes **256/1,024/4,096** with **360 additional measured samples**, preserving the original primary target and run.

These ratios describe CPU attention computation only. Payload reduction is not process RSS reduction, and query-vector throughput is not generated-token throughput. The mechanism reproduces established Cascade/Hydragen and online-softmax ideas; the contribution is the CPU implementation, lifecycle contract, controlled ablation, and evidence.

## Evidence index

| Claim | Implementation | Verification and raw evidence |
|---|---|---|
| Bullet 1: grouped exact computation and primary latency | [attention.py](../../src/prefixfold/attention.py), `attend`, `_update`, `AttentionPlan` | [Frozen targets](../../research/targets.json), [main configuration](../../configs/benchmark.json), [raw run](../../results/benchmark.json), [summary](../../results/summary.json), case `shared_b16_p4096` |
| Bullet 2: ownership, capacity, appends, cleanup, payload | [cache.py](../../src/prefixfold/cache.py), `SharedPrefix`, `DecodeBatch` | [Cache tests](../../tests/test_cache.py), [validation tests](../../tests/test_validation.py), raw `kv_payload_bytes`; primary payloads 12,582,912 and 138,412,032 bytes |
| Bullet 3: fair variants, full samples, correctness and unfavorable cases | [benchmark.py](../../benchmark.py), [analyze.py](../../scripts/analyze.py), [float64 oracle](../../src/prefixfold/attention.py) | [Numerical tests](../../tests/test_attention.py), [complete table](../../results/table_en.md), raw `measurement_order`, `samples_ms`, `correctness`, `setup`, `first_call_ms`, `peak_rss_bytes` |
| Bullet 4: engineering interface and sensitivity | [Public API](API.md), [CLI](../../src/prefixfold/cli.py), [lockfile](../../uv.lock), [CI configuration](../../.github/workflows/ci.yml) | [Integration tests](../../tests/test_integration.py), [tile256](../../results/tile256.json), [tile1024](../../results/tile1024.json), [tile4096](../../results/tile4096.json); final executed checks are in [Release](RELEASE.md) |

Main benchmark source revision: `70e7ed50e51b4de74a9d35c358068aa689a939c5`; source manifest hash: `c0003d3eaf2887a4b8a026e9139ea2cc783d76487fa219f9bdaeac15c9b7d500`. Main run ID: `20260921T121914Z-56a3bc9a`. CI configuration is a repository artifact; its existence alone is not evidence that remote CI ran.

## Short project talk

**Why this problem.** I chose a concrete data-reuse question that can be examined on ordinary CPU hardware. Several decode requests may share identical projected prefix KV. Storing that prefix once saves capacity; the additional question is whether exposing shared access to matrix multiplication improves attention latency.

**How it works.** Each prefix is copied once into immutable storage. A batch keeps independent bounded suffix buffers and lengths. Queries are grouped across requests for each prefix tile. A running maximum, exponential sum, and weighted-value numerator retain enough information to combine prefix and suffix attention exactly in real arithmetic. The final normalization returns one output per request/head. Input validation, locks and cleanup make this computation repeatable through the public API.

**Key tradeoff.** Grouping can improve reuse and matrix shape without reducing the number of query-key dot products. Tiling limits explicit score storage but adds loops and reductions. NumPy makes the mechanism readable; fused CPU code could reduce overhead at the cost of another implementation and build surface. Prefix equality and positional encoding remain the caller's responsibility.

**Most informative experiment.** On the fixed primary shape, grouped attention takes 2.0437 ms against 7.1603 ms for SDPA and 7.2138 ms for the same engine without prefix grouping. The latter comparison preserves suffix code and KV payload, so it connects the benefit to the intended mechanism. Correctness uses an independent float64 path, and every valid timing sample is retained.

**Failures and limitations.** The optimized path is slower for no prefix, one request, and a short prefix. An implementation review also found that extreme finite values in ignored suffix padding could overflow before masking; unused storage is now zeroed during validated loading and the case has a regression test. Valid arithmetic exceeding float32 range is still rejected explicitly. Experiments are synthetic operator workloads on one CPU session; they cannot substantiate model-quality, generated-token, production, GPU, or universal-platform claims.

**Next research questions.** Would an independently evaluated dispatch rule select dense SDPA for unfavorable shapes? How stable are ratios across thermal states and independent sessions? Does a fused CPU kernel retain the same error contract while reducing overhead? How do real projected KV distributions change the crossover point? These are future questions, not current accomplishments.

## Concise demonstration

```sh
make demo
.venv/bin/python -m pytest tests/test_attention.py tests/test_cache.py -q
.venv/bin/python benchmark.py --config configs/benchmark.json --output results/interview-run.json
```

Start with the default demo, show the state update in `attention.py`, then explain one positive and one negative workload from the raw results. The existing records are evidence of this run; a fresh run produces its own timing record rather than guaranteeing identical numbers.
