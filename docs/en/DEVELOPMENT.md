# Development record

[简体中文](../zh/DEVELOPMENT.md) · [Home](../../README.md)

I am maintaining this as a focused numerical systems project. This record describes activities that actually ran on 2026-09-21; it does not imply months of development, prior deployment, a publication, or that a future maintainer has already studied the implementation.

## Decisions and actual fixes

The workspace was empty, so the project was isolated in its own repository. Hardware checks found an M1 Pro with 16 GiB memory, Python 3.12.2, Clang and Metal. CPU experiments were selected; Docker was absent, and a model-site download probe failed. PyPI downloads and the authenticated GitHub account API succeeded. Public records omit local paths and private email.

The numerical and performance contracts were committed before timing. Literature review established that the main decomposition and grouping mechanisms belong to Cascade Attention and Hydragen; the project documents them as reproduced mechanisms with CPU engineering, not novel algorithms.

The first editable install occurred before package files existed and could not import `prefixfold`. Reinstalling the local package after writing the source fixed the installation. The first static pass found NumPy shape-type inference mismatches; explicit NDArray annotations fixed them. These were actual development errors, not skipped checks.

The initial test run had 117 passes and two failures because row-index lists were not accepted. The API was deliberately extended to integer sequences while retaining strict dtype/range/uniqueness checks. No numerical tolerance or valid assertion was weakened.

Independent review found that extremely large but inactive suffix padding could overflow BLAS before masking. `load_suffix` now zeroes all unused capacity after validation, and a real numerical regression exercises that case. Another review found that the first ungrouped path changed suffix batching too; the final implementation keeps suffix execution identical and disables only shared-prefix grouping. Prefix weak-reference release and concurrent writers/readers are exercised directly.

A direct `pytest` invocation initially could not import the root benchmark module, although `python -m pytest` worked. The pytest configuration now explicitly includes the repository root. This fixes both supported test entry points.

## Actual commit checkpoints

| Commit | Completed work and observed validation |
|---|---|
| `6d49e97` | Repository scaffold, dependency lock and frozen targets; environment probes executed |
| `7eeb671` | Core, state lifecycle, FP64 oracle and CLI; demo max absolute error about 7.85e-8; lint/type checks passed |
| `99e2958` | Offline numerical/integration tests, practical baselines, benchmark harness, CI/container configuration; 138 tests passed at this checkpoint |

The initial raw benchmark, retained in `results/pre-resource-fix/benchmark.json`, records full revision `99e2958d71a20396308e931fca53cbdcdff98773` and clean tested-source hashes. Later documentation/tooling commits do not silently replace that identity. Use `git log --oneline --reverse` for the complete current history, and compare raw `file_sha256` values with delivered source.

## Final resource and evidence hardening

A final NumPy allocation audit found that int64 token indices, a boolean mask and existing float32 scores could temporarily require `13T` bytes for B=D=1, exceeding the old `12T` leading term. The estimate now uses `4B(4T+8D+32)` and shrinks tiles accordingly. A million-token test with an 8 MiB budget tracks actual NumPy allocations with a fixed allowance for Python overhead. It is not an RSS test. Analysis now rejects incomplete/falsely passed records and makes README updates explicit. The final full local test run at this checkpoint had **181 passes**; the `70e7ed5` commit body mistakenly said 180, corrected here without rewriting history.

| Commit | Completed work |
|---|---|
| `2cd7db5` | Initial complete measurements and related-work evidence |
| `70e7ed5` | Resource-estimate regression fix, strict analysis, document/privacy validators, and retained pre-fix evidence |

The public-source checkpoint `1f66376` passed remote Python 3.11/3.12/3.13 CI and the Linux Docker build/demo job. `fe31764` added the verified public URLs and external receipt. SVG identifiers were subsequently made deterministic and repeated rendering was checked by SHA-256; this changes no measurements.

## Results and verification

The initial experiment retained all 840 measured calls across seven cases and four variants, with 90.91% K/V array reduction and 2.99× median compute speedup. After the resource-estimate fix, the complete main and tile suites were rerun on revision `70e7ed5`: the final primary result is 3.50× (7.1603 / 2.0437 ms), with the same 90.91% K/V reduction. Both sessions remain available. The configured tiles and numerical kernels are unchanged, so the latency difference must not be attributed to the resource-estimate fix. No-prefix, single-request and short-prefix regressions remain visible. Post-primary tile sensitivity uses separately recorded, matching fixtures; it does not change the original default or target.

[Local acceptance](../../results/acceptance.json) records actual commands, exits, logs and source hashes, including clean wheel installation. [Experiments](EXPERIMENTS.md) preserves timing and memory scope. [Release](RELEASE.md) distinguishes local verification from external publishing and CI.

The first complete local acceptance run exposed trailing whitespace emitted by Matplotlib SVG output. The plot writer now normalizes whitespace without changing chart geometry or measurements. The failed check and original log remain in `results/validation-first/`; the final acceptance records 21 passed, 0 failed, 0 skipped and 2 local Docker checks not run.

## Next maintenance questions

I would investigate whether a measured dispatch rule can choose SDPA for small/nonsharing inputs, whether a compiled merge loop reduces Python overhead, and how real model KV distributions affect the crossover. These are future questions, not implemented branches or promised results. A model integration would need its own model license, positional-encoding verification and generation-quality tests.
