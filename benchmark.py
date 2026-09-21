#!/usr/bin/env python3
"""Isolated, interleaved CPU attention benchmark. / 隔离并交错执行的 CPU 注意力基准。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import selectors
import subprocess
import sys
import time
import uuid
from datetime import UTC
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parent
VARIANTS = ("prefixfold", "ungrouped", "torch_sdpa", "numpy_dense")
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def utc_now() -> str:
    """Return an actual UTC timestamp. / 返回实际 UTC 时间。"""
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def source_identity() -> dict[str, Any]:
    """Hash the tested implementation and harness. / 对被测实现与基准程序计算哈希。"""
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "benchmark.py"]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "benchmark.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "tested_source_dirty": bool(dirty.stdout.strip()),
        "source_sha256": combined,
        "file_sha256": hashes,
    }


def validate_config(config: dict[str, Any]) -> None:
    """Reject malformed or excessive workloads. / 拒绝无效或超出范围的工作负载。"""
    if not isinstance(config, dict):
        raise ValueError("config must be an object / 配置必须为对象")
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1 / schema_version 必须为 1")
    for name in ("warmup", "repetitions", "seed"):
        value = config.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer / 必须为非负整数")
    if config["repetitions"] < 1:
        raise ValueError("repetitions must be positive / 重复次数必须为正")
    if not isinstance(config.get("cases"), list) or not config["cases"]:
        raise ValueError("cases must be a nonempty list / cases 必须为非空列表")
    plan = config.get("plan", {})
    if not isinstance(plan, dict) or set(plan) - {"tile_tokens", "workspace_bytes"}:
        raise ValueError("invalid attention plan / 无效的注意力配置")
    for key, value in plan.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"invalid plan {key} / 无效的计划参数")
    names = set()
    for case in config["cases"]:
        if not isinstance(case, dict):
            raise ValueError("case must be an object / 场景必须为对象")
        if not isinstance(case.get("name"), str) or not case["name"] or case["name"] in names:
            raise ValueError("case names must be unique / 场景名必须唯一")
        names.add(case["name"])
        for key in ("batch", "heads", "prefix", "suffix", "dim"):
            value = case.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (0 if key == "prefix" else 1)
            ):
                raise ValueError(f"invalid {key} / 无效的 {key}")
        # Bound one dense K/V pair to 1 GiB; this harness targets ordinary CPU machines.
        # 单组稠密 K/V 不超过 1 GiB，限制普通 CPU 环境中的资源消耗。
        payload = (
            8 * case["batch"] * case["heads"] * (case["prefix"] + case["suffix"]) * case["dim"]
        )
        if payload > 1024**3:
            raise ValueError(
                "dense K/V exceeds 1 GiB benchmark limit / 稠密 K/V 超出基准的 1 GiB 限制"
            )


def _peak_rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False), flush=True)


def _worker(case: dict[str, Any], variant: str, seed: int, plan_config: dict[str, Any]) -> None:
    for name, value in THREAD_ENV.items():
        os.environ[name] = value
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    from prefixfold import (
        AttentionPlan,
        DecodeBatch,
        SharedPrefix,
        __version__,
        attend,
        dense_reference,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    controller = threadpool_limits(limits=1)
    rng = np.random.default_rng(seed)
    b, h, p, s, d = (case[key] for key in ("batch", "heads", "prefix", "suffix", "dim"))
    started = time.perf_counter_ns()
    prefix_k = rng.standard_normal((h, p, d), dtype=np.float32)
    prefix_v = rng.standard_normal((h, p, d), dtype=np.float32)
    suffix_k = rng.standard_normal((b, h, s, d), dtype=np.float32)
    suffix_v = rng.standard_normal((b, h, s, d), dtype=np.float32)
    q = rng.standard_normal((b, h, d), dtype=np.float32)
    lengths = rng.integers(0, s + 1, size=b) if case.get("ragged") else np.full(b, s)
    if p == 0:
        lengths = np.maximum(lengths, 1)
    prefix = SharedPrefix(prefix_k, prefix_v)
    batch = DecodeBatch(prefix, batch_size=b, capacity=s)
    batch.load_suffix(suffix_k, suffix_v, lengths=lengths)
    del prefix_k, prefix_v, suffix_k, suffix_v
    common_setup_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    mask_nbytes = 0
    plan = AttentionPlan(**plan_config)
    if variant in {"torch_sdpa", "numpy_dense"}:
        keys, values, mask = batch.materialize()
        kv_nbytes = int(keys.nbytes + values.nbytes)
        mask_nbytes = int(mask.nbytes)
        if variant == "torch_sdpa":
            qt = torch.from_numpy(q).unsqueeze(2)
            kt, vt = torch.from_numpy(keys), torch.from_numpy(values)
            mt = torch.from_numpy(mask)[:, None, None, :]

            def compute() -> Any:
                with torch.inference_mode():
                    return (
                        torch.nn.functional.scaled_dot_product_attention(
                            qt, kt, vt, attn_mask=mt, dropout_p=0.0
                        )
                        .squeeze(2)
                        .numpy()
                    )
        else:
            scale = np.float32(d**-0.5)

            def compute() -> Any:
                scores = np.matmul(q[:, :, None, :], keys.swapaxes(-2, -1)) * scale
                np.copyto(scores, -np.inf, where=~mask[:, None, None, :])
                scores -= scores.max(axis=-1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= scores.sum(axis=-1, keepdims=True)
                return np.matmul(scores, values).squeeze(2)
    else:
        kv_nbytes = int(batch.kv_nbytes - batch.lengths.nbytes)

        def compute() -> Any:
            return attend(q, batch, plan=plan, grouped=variant == "prefixfold")

    variant_setup_ms = (time.perf_counter_ns() - started) / 1e6
    rss_setup = _peak_rss()
    started = time.perf_counter_ns()
    output = compute()
    first_call_ms = (time.perf_counter_ns() - started) / 1e6
    reference = dense_reference(q, batch)
    delta = np.abs(output.astype(np.float64) - reference)
    denominator = np.maximum(np.abs(reference), 1e-12)
    passed = bool(np.allclose(output, reference, atol=3e-5, rtol=3e-5))
    correctness = {
        "passed": passed,
        "atol": 3e-5,
        "rtol": 3e-5,
        "max_abs_error": float(delta.max()),
        "max_relative_error": float((delta / denominator).max()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "reference": "dense_reference_float64",
    }
    del reference, delta, denominator, output
    # Baselines release the shared source; only their dense K/V remain during timing.
    # 基线在计时前释放共享输入，仅保留实际使用的稠密 K/V。
    if variant in {"torch_sdpa", "numpy_dense"}:
        del batch, prefix
    pools = [
        {
            k: pool[k]
            for k in ("user_api", "internal_api", "num_threads", "version", "architecture")
            if k in pool
        }
        for pool in threadpool_info()
    ]
    _emit(
        {
            "status": "ready" if passed else "failed",
            "correctness": correctness,
            "setup": {
                "common_fixture_ms": common_setup_ms,
                "variant_materialization_ms": variant_setup_ms,
            },
            "first_call_ms": first_call_ms,
            "kv_payload_bytes": kv_nbytes,
            "mask_bytes": mask_nbytes,
            "peak_rss_after_setup_bytes": rss_setup,
            "suffix_lengths": lengths.tolist(),
            "prefixfold_version": __version__,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "threadpools": pools,
        }
    )
    if not passed:
        return
    for line in sys.stdin:
        command = json.loads(line)
        if command["op"] == "close":
            _emit({"status": "closed", "peak_rss_bytes": _peak_rss()})
            controller.restore_original_limits()
            return
        if command["op"] != "run":
            raise ValueError("invalid worker command / 无效的工作进程命令")
        started = time.perf_counter_ns()
        output = compute()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        _emit({"status": "ok", "elapsed_ms": elapsed_ms, "checksum": float(output.sum())})


def _receive(stream: TextIO, timeout: float) -> dict[str, Any]:
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise TimeoutError("worker response timed out / 工作进程响应超时")
        line = stream.readline()
    if not line:
        raise RuntimeError("worker exited without response / 工作进程未返回结果即退出")
    return json.loads(line)  # type: ignore[no-any-return]


def _sanitize(text: str) -> str:
    for path in (str(ROOT), sys.prefix, str(Path.home())):
        text = text.replace(path, "<local-path>")
    return text[-6000:]


def _relative_argument(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _hardware() -> dict[str, Any]:
    if sys.platform == "darwin":
        values = {}
        for key, field in (
            ("machdep.cpu.brand_string", "cpu_model"),
            ("hw.memsize", "physical_memory_bytes"),
        ):
            probe = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=False
            )
            if probe.returncode == 0:
                value = probe.stdout.strip()
                values[field] = int(value) if field == "physical_memory_bytes" else value
        return values
    memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return {
        "cpu_model": platform.processor() or platform.machine(),
        "physical_memory_bytes": memory,
    }


def run(
    config: dict[str, Any], config_path: Path, output: Path, variants: list[str], timeout: float
) -> bool:
    """Run and persist every measurement, including failures. / 执行并保存全部测量及失败记录。"""
    validate_config(config)
    if len(variants) != len(set(variants)) or any(v not in VARIANTS for v in variants):
        raise ValueError("invalid or duplicate variants / 无效或重复的实现")
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    config_bytes = config_path.read_bytes()
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": utc_now(),
        "status": "running",
        "command": (
            f"python benchmark.py --config {_relative_argument(config_path)} "
            f"--output {_relative_argument(output)} --variants {' '.join(variants)} "
            f"--timeout {timeout}"
        ),
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source": source_identity(),
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "backend": "cpu",
            "thread_environment": THREAD_ENV,
            **_hardware(),
        },
        "protocol": {
            "timing_scope": (
                "compute_only; fixture creation and dense materialization reported separately"
            ),
            "ordering": "seeded shuffled variants within every warmup and measured round",
            "memory": (
                "whole-worker process peak RSS including imports, setup, correctness and timed "
                "calls; not variant workspace"
            ),
            "kv_payload": (
                "actual allocated K/V array bytes; excludes lengths, mask, output and workspace"
            ),
            "throughput_unit": (
                "query_vectors_per_second (batch * heads / seconds); not generated tokens"
            ),
            "data": (
                "seeded synthetic float32 normal Q/K/V; no pretrained model or real request trace"
            ),
            "tail_latency": (
                "not claimed; 30 observations are insufficient for reliable extreme tails"
            ),
            "thread_note": (
                "VECLIB_MAXIMUM_THREADS=1 requested; Accelerate enforcement is not observable "
                "through threadpoolctl"
            ),
        },
        "workloads": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    save()
    randomizer = random.Random(config["seed"])
    all_ok = True
    for case_index, case in enumerate(config["cases"]):
        workload: dict[str, Any] = {"case": case, "variants": {}, "measurement_order": []}
        result["workloads"].append(workload)
        workers: dict[str, subprocess.Popen[str]] = {}
        order = variants.copy()
        randomizer.shuffle(order)
        try:
            for variant in order:
                entry: dict[str, Any] = {"status": "starting", "samples_ms": [], "warmup_ms": []}
                workload["variants"][variant] = entry
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "benchmark.py"),
                    "--worker",
                    variant,
                    "--case-json",
                    json.dumps(case),
                    "--seed",
                    str(config["seed"] + case_index),
                    "--plan-json",
                    json.dumps(config.get("plan", {})),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env={**os.environ, **THREAD_ENV},
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                workers[variant] = process
                assert process.stdout is not None
                try:
                    entry.update(_receive(process.stdout, timeout))
                    if entry["status"] != "ready":
                        all_ok = False
                except (TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                    entry.update(
                        status="timeout" if isinstance(exc, TimeoutError) else "failed",
                        error=str(exc),
                    )
                    all_ok = False
            active = [v for v in order if workload["variants"][v]["status"] == "ready"]
            for iteration in range(config["warmup"] + config["repetitions"]):
                shuffled = active.copy()
                randomizer.shuffle(shuffled)
                phase = "warmup" if iteration < config["warmup"] else "measured"
                workload["measurement_order"].append({"phase": phase, "variants": shuffled})
                for variant in shuffled:
                    process = workers[variant]
                    entry = workload["variants"][variant]
                    assert process.stdin is not None and process.stdout is not None
                    try:
                        process.stdin.write('{"op":"run"}\n')
                        process.stdin.flush()
                        sample = _receive(process.stdout, timeout)
                        if sample["status"] != "ok":
                            raise RuntimeError(
                                sample.get("error", "worker measurement failed / 工作进程测量失败")
                            )
                        entry["warmup_ms" if phase == "warmup" else "samples_ms"].append(
                            sample["elapsed_ms"]
                        )
                    except (
                        TimeoutError,
                        RuntimeError,
                        BrokenPipeError,
                        json.JSONDecodeError,
                    ) as exc:
                        entry.update(
                            status="timeout" if isinstance(exc, TimeoutError) else "failed",
                            error=str(exc),
                        )
                        active.remove(variant)
                        all_ok = False
                save()
            for variant in active:
                process = workers[variant]
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write('{"op":"close"}\n')
                process.stdin.flush()
                entry = workload["variants"][variant]
                closed = _receive(process.stdout, timeout)
                entry.update(status="passed", peak_rss_bytes=closed["peak_rss_bytes"])
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            all_ok = False
            workload["error"] = f"{type(exc).__name__}: {_sanitize(str(exc))}"
            for entry in workload["variants"].values():
                if entry["status"] in {"starting", "ready"}:
                    entry.update(status="failed", error=workload["error"])
        finally:
            for variant, process in workers.items():
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                entry = workload["variants"][variant]
                entry["worker_exit_code"] = process.returncode
                if process.returncode != 0 and entry["status"] == "passed":
                    entry.update(status="failed", error="nonzero exit / 工作进程非零退出")
                    all_ok = False
                if process.stderr is not None:
                    error = process.stderr.read()
                    if error:
                        workload["variants"][variant]["worker_stderr"] = _sanitize(error)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            save()
        print(
            f"{case['name']}: "
            + ", ".join(f"{v}={e['status']}" for v, e in workload["variants"].items()),
            flush=True,
        )
    result.update(status="passed" if all_ok else "failed", completed_at=utc_now())
    save()
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/benchmark.json"), help="suite JSON / 基准配置"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark.json"),
        help="raw results JSON / 原始结果",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="implementations / 实现",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="worker response timeout seconds / 工作进程响应超时秒数",
    )
    parser.add_argument("--worker", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--case-json", help=argparse.SUPPRESS)
    parser.add_argument("--plan-json", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=20260921, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            _worker(json.loads(args.case_json), args.worker, args.seed, json.loads(args.plan_json))
            return 0
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be positive / 超时必须为正数")
        config = json.loads(args.config.read_text())
        return 0 if run(config, args.config, args.output, args.variants, args.timeout) else 1
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        if args.worker:
            _emit({"status": "failed", "error": f"{type(exc).__name__}: {_sanitize(str(exc))}"})
        else:
            print(
                f"Benchmark failed / 基准失败: {type(exc).__name__}: {_sanitize(str(exc))}",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
