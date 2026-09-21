# Release preparation and status

[简体中文](../zh/RELEASE.md) · [Home](../../README.md)

I maintain PrefixFold 0.1.0 as a focused, reproducible CPU numerical library. Release readiness is defined by the implementation, evidence and known boundaries, not by a production-maturity label.

## Status

**Verified public source and remote CI:** [initial successful workflow](https://github.com/hardwork-xu/prefixfold/actions/runs/35600203428) passed all three Python jobs (3.11, 3.12, 3.13) and the Linux Docker build/demo job. The [machine-readable receipt](../../results/publication.json) records the actual checked revision and job URLs. Local macOS Docker remains unexecuted because no Docker is installed. Versioned release artifacts are available from the repository Releases page after publication; package-index publication and public-service deployment are not performed.

Public repository: [hardwork-xu/prefixfold](https://github.com/hardwork-xu/prefixfold). External verification is recorded in the release receipt when completed.

The local implementation, tests and primary benchmark have run. Packaging, clean-install and document/privacy checks are recorded in [acceptance.json](../../results/acceptance.json). Docker is not installed on the measured Mac: **local container validation was not executed**. The repository contains Linux CPU container and ordinary GitHub Actions configurations; the verified external run is recorded above.

The public repository and release are prepared only from project files and privacy-safe Git identity. No private contact address, workstation path, private dataset, token or model weight belongs in the release. The personal provenance note is stored outside this repository. No paid infrastructure or public service is required.

## Installation, demonstration and build

```sh
python3 -m venv .venv
.venv/bin/python -m pip install uv==0.8.22
.venv/bin/uv sync --frozen --group bench
make demo
make test check
make build
```

The package build produces `dist/prefixfold-0.1.0-py3-none-any.whl` and `dist/prefixfold-0.1.0.tar.gz`. Build artifacts are not source-tracked. A clean interpreter can install the wheel and run `python -m prefixfold demo`; `make validate` executes that check with `-I` to avoid importing the checkout by accident.

```sh
docker build -t prefixfold:0.1.0 .
docker run --rm prefixfold:0.1.0
```

This Docker path installs only the NumPy runtime from the frozen lock and runs as an unprivileged user. It has no GPU path. Benchmark dependencies are a separate development group; host GPU availability is unrelated to this image.

## 0.1.0 release notes draft

- Add FP32 CPU shared-prefix single-token attention with grouped prefix matrix multiplication, tiled stable softmax and ragged suffix masking.
- Add immutable prefix ownership, fixed-capacity transactional suffix append, synchronized reads, explicit cleanup, Python API and offline NPZ/JSON CLI.
- Include 7-workload/4-variant main evidence and post-primary tile sensitivity. On M1 Pro, the fixed B16/H4/P4096/S128/D64 workload uses 12 MiB K/V versus 132 MiB dense and measures 3.50× median compute speedup over PyTorch CPU SDPA. All 28 main numerical comparisons pass the fixed tolerance. Small/nonsharing regressions remain documented.
- Provide bilingual research, maintenance and API documentation, MIT licensing, verified attribution, locked dependencies, CPU CI and local acceptance logs.

Known limitations: CPU FP32 operator scope only, synthetic workload evidence, one prefix level, no model-generation claim, no GPU implementation, no automatic dispatch, no hard process-memory cap, and no local Docker execution on the measured host.

## Public metadata

One sentence: **Exact shared-prefix decode attention on CPU, with bounded KV ownership and reproducible benchmarks.**

GitHub About: **CPU shared-prefix attention with immutable KV prefixes, ragged suffixes, exact softmax merging, and reproducible PyTorch baselines. CPU 共享前缀精确注意力与可复现实验。**

Suggested topics: `attention`, `cpu`, `kv-cache`, `numpy`, `pytorch`, `benchmark`, `reproducible-research`, `inference`.

[CITATION.cff](../../CITATION.cff) identifies the verified public author name and software version. Its CFF 1.2.0 schema validation is part of acceptance; there is no invented DOI or paper citation. Refer to the original papers for their algorithms. The [MIT license](../../LICENSE), [third-party notice](../../NOTICE.md), [contribution guide](../../CONTRIBUTING.md) and [security scope](../../SECURITY.md) accompany the release.

Publishing to a package index, claiming production deployment, purchasing infrastructure and downloading/integrating a model are outside this release's executed scope. A GitHub source release does not imply any of those actions.
