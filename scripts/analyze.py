#!/usr/bin/env python3
"""Generate bilingual tables and plots from raw evidence. / 从原始证据生成双语表格与图表。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("prefixfold", "ungrouped", "torch_sdpa", "numpy_dense")
TOLERANCE = 3e-5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest(raw: dict[str, Any], config_path: Path | None) -> tuple[dict[str, Any], list[str]]:
    """Verify original config bytes and parse the recorded invocation, without executing it.

    校验原配置文件字节并解析记录的命令；不会执行命令。外部配置需显式传入路径。
    """
    _require(isinstance(raw.get("command"), str), "missing command / 缺少执行命令")
    tokens = shlex.split(raw["command"])
    if config_path is None:
        _require("--config" in tokens, "missing config argument / 缺少配置参数")
        index = tokens.index("--config") + 1
        _require(index < len(tokens), "missing config value / 缺少配置参数值")
        config_path = (ROOT / tokens[index]).resolve()
        _require(
            config_path.is_relative_to(ROOT),
            "external config requires --config-source / 外部配置需显式提供 --config-source",
        )
    _require(config_path.is_file(), "config source not found / 未找到原配置文件")
    original = config_path.read_bytes()
    _require(
        _hash(raw.get("config_sha256"))
        and hashlib.sha256(original).hexdigest() == raw["config_sha256"],
        "config hash mismatch / 配置文件哈希不匹配",
    )
    config = json.loads(original)
    _require(
        isinstance(config, dict) and config == raw.get("config"),
        "embedded config differs from source / 内嵌配置与原文件不一致",
    )
    variants = list(VARIANTS)
    if "--variants" in tokens:
        start = tokens.index("--variants") + 1
        variants = []
        for token in tokens[start:]:
            if token.startswith("--"):
                break
            variants.append(token)
    _require(
        bool(variants)
        and len(variants) == len(set(variants))
        and all(variant in VARIANTS for variant in variants),
        "invalid variant manifest / 实现清单无效",
    )
    return config, variants


def _passed_record(entry: dict[str, Any], config: dict[str, Any]) -> None:
    for name, expected in (("samples_ms", config["repetitions"]), ("warmup_ms", config["warmup"])):
        samples = entry.get(name)
        _require(
            isinstance(samples, list)
            and len(samples) == expected
            and all(_finite(value, positive=True) for value in samples),
            f"invalid {name} count or values / 样本数量或数值无效",
        )
    correctness = entry.get("correctness")
    _require(isinstance(correctness, dict), "missing correctness / 缺少正确性记录")
    _require(
        correctness.get("passed") is True
        and correctness.get("atol") == TOLERANCE
        and correctness.get("rtol") == TOLERANCE
        and correctness.get("reference") == "dense_reference_float64",
        "false passed status or changed tolerance / 通过状态或误差标准无效",
    )
    for name in ("max_abs_error", "max_relative_error", "rmse"):
        _require(_finite(correctness.get(name)), "invalid error metric / 误差指标无效")
    _require(
        _finite(entry.get("first_call_ms"), positive=True), "invalid first call / 首次计时无效"
    )
    for field in ("kv_payload_bytes", "peak_rss_bytes"):
        value = entry.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"invalid {field} / 字节计数无效",
        )
    setup = entry.get("setup")
    _require(
        isinstance(setup, dict)
        and all(
            _finite(setup.get(key)) for key in ("common_fixture_ms", "variant_materialization_ms")
        ),
        "invalid setup measurements / 初始化测量无效",
    )
    _require(
        entry.get("worker_exit_code", 0) == 0, "worker exit contradicts pass / 退出码与通过状态矛盾"
    )


def validate_raw(raw: dict[str, Any], config_path: Path | None = None) -> list[str]:
    """Reject incomplete or contradictory evidence; retain complete failure records.

    拒绝不完整或相互矛盾的证据；完整失败记录允许汇总，但不得被标为全部通过。
    """
    _require(
        isinstance(raw, dict) and raw.get("schema_version") == 1,
        "unsupported result schema / 不支持的结果格式",
    )
    _require(raw.get("status") in {"passed", "failed"}, "run is incomplete / 运行尚未完成")
    config, variants = _manifest(raw, config_path)
    for field, minimum in (("repetitions", 1), ("warmup", 0)):
        value = config.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"invalid {field} / 重复次数配置无效",
        )
    cases, workloads = config.get("cases"), raw.get("workloads")
    _require(isinstance(cases, list) and bool(cases), "invalid cases / 场景清单无效")
    _require(
        isinstance(workloads, list) and len(workloads) == len(cases),
        "incomplete workload manifest / 工作负载清单不完整",
    )
    expected = {}
    for case in cases:
        _require(
            isinstance(case, dict) and isinstance(case.get("name"), str),
            "invalid case entry / 场景记录无效",
        )
        _require(case["name"] not in expected, "duplicate configured case / 配置场景重复")
        expected[case["name"]] = case
    _require(config.get("primary_case") in expected, "missing primary case / 缺少主场景")
    seen = set()
    for workload in workloads:
        _require(isinstance(workload, dict), "invalid workload record / 工作负载记录无效")
        case = workload.get("case")
        _require(isinstance(case, dict), "missing workload case / 缺少工作负载场景")
        name = case.get("name")
        _require(
            isinstance(name, str)
            and name in expected
            and name not in seen
            and case == expected[name],
            "workload differs from manifest / 工作负载与配置清单不一致",
        )
        seen.add(name)
        records = workload.get("variants")
        _require(
            isinstance(records, dict) and set(records) == set(variants),
            "incomplete variant manifest / 实现清单不完整",
        )
        for entry in records.values():
            _require(
                isinstance(entry, dict) and entry.get("status") in {"passed", "failed", "timeout"},
                "incomplete variant record / 实现记录不完整",
            )
            if entry["status"] == "passed":
                _passed_record(entry, config)
            else:
                _require(
                    raw["status"] == "failed", "failure contradicts run pass / 失败与总通过矛盾"
                )
    source = raw.get("source")
    _require(isinstance(source, dict), "missing source identity / 缺少源码标识")
    files = source.get("file_sha256")
    _require(
        isinstance(files, dict) and bool(files) and all(_hash(value) for value in files.values()),
        "invalid source file hashes / 源文件哈希无效",
    )
    _require(
        hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        == source.get("source_sha256"),
        "source aggregate hash mismatch / 源码集合哈希不匹配",
    )
    return variants


def summarize(raw: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    """Summarize every valid sample after evidence validation. / 校验证据后汇总全部有效样本。"""
    variants_manifest = validate_raw(raw, config_path)
    rows: list[dict[str, Any]] = []
    for workload in raw["workloads"]:
        case = workload["case"]
        variants = workload["variants"]
        valid = {name: value for name, value in variants.items() if value["status"] == "passed"}
        sdpa = (
            statistics.median(valid["torch_sdpa"]["samples_ms"]) if "torch_sdpa" in valid else None
        )
        dense_bytes = valid.get("torch_sdpa", valid.get("numpy_dense", {})).get("kv_payload_bytes")
        for name, entry in variants.items():
            row: dict[str, Any] = {"case": case["name"], "variant": name, "status": entry["status"]}
            if entry["status"] == "passed":
                samples = entry["samples_ms"]
                if not samples or any(not isinstance(x, (int, float)) or x <= 0 for x in samples):
                    raise ValueError("invalid latency sample / 无效的延迟样本")
                median = statistics.median(samples)
                row.update(
                    samples=len(samples),
                    median_ms=median,
                    min_ms=min(samples),
                    max_ms=max(samples),
                    mean_ms=statistics.mean(samples),
                    stdev_ms=statistics.stdev(samples) if len(samples) > 1 else None,
                    first_call_ms=entry["first_call_ms"],
                    kv_payload_bytes=entry["kv_payload_bytes"],
                    peak_worker_rss_bytes=entry["peak_rss_bytes"],
                    query_vectors_per_second=case["batch"] * case["heads"] * 1000 / median,
                    torch_sdpa_speedup=sdpa / median if sdpa is not None else None,
                    kv_reduction_fraction=1 - entry["kv_payload_bytes"] / dense_bytes
                    if dense_bytes
                    else None,
                    correctness=entry["correctness"],
                    setup=entry["setup"],
                )
            else:
                row["error"] = entry.get("error", "see raw worker result / 见原始工作进程结果")
            rows.append(row)
    primary = next(
        (
            r
            for r in rows
            if r["case"] == raw["config"]["primary_case"] and r["variant"] == "prefixfold"
        ),
        None,
    )
    targets = raw["config"]["targets"]
    verdict: dict[str, Any] = {
        "workload": raw["config"]["primary_case"],
        "targets": targets,
        "measured": {},
        "met": {},
    }
    if primary is not None and primary["status"] == "passed":
        for metric, target in targets.items():
            measured = primary.get(metric)
            verdict["measured"][metric] = measured
            verdict["met"][metric] = measured >= target if measured is not None else None
    return {
        "schema_version": 1,
        "run_id": raw["run_id"],
        "raw_status": raw["status"],
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": raw["config_sha256"],
        "validated_variants": variants_manifest,
        "source": raw["source"],
        "environment": raw["environment"],
        "rows": rows,
        "primary_verdict": verdict,
        "all_correct": raw["status"] == "passed"
        and bool(rows)
        and all(r["status"] == "passed" and r["correctness"]["passed"] for r in rows),
    }


def table(summary: dict[str, Any], language: str) -> str:
    """Render measurements with shared numbers. / 双语表格共享同一组数值。"""
    headings = {
        "en": (
            "| Workload | Variant | Median ms | SDPA / variant | KV MiB | "
            "KV reduction | Max abs error | n |"
        ),
        "zh": (
            "| 工作负载 | 实现 | 中位数 ms | SDPA / 当前实现 | KV MiB | "
            "KV 减少比例 | 最大绝对误差 | n |"
        ),
    }
    lines = [headings[language], "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        if row["status"] != "passed":
            lines.append(
                f"| {row['case']} | {row['variant']} | {row['status']} | — | — | — | — | — |"
            )
            continue
        speedup = (
            f"{row['torch_sdpa_speedup']:.3f}×" if row["torch_sdpa_speedup"] is not None else "—"
        )
        reduction = (
            f"{row['kv_reduction_fraction']:.2%}"
            if row["kv_reduction_fraction"] is not None
            else "—"
        )
        lines.append(
            f"| {row['case']} | {row['variant']} | {row['median_ms']:.4f} | {speedup} | "
            f"{row['kv_payload_bytes'] / 1024**2:.3f} | {reduction} | "
            f"{row['correctness']['max_abs_error']:.2e} | {row['samples']} |"
        )
    note = {
        "en": (
            "Measured CPU attention compute only; setup is separate. KV bytes are allocated "
            "array payload, not process RSS. All samples, cold calls, setup and whole-worker "
            "RSS remain in raw JSON. Throughput counts query vectors, not generated tokens. "
            "Extreme tail latency is not claimed."
        ),
        "zh": (
            "实测范围仅为 CPU 注意力计算；初始化单独记录。KV 字节是已分配数组的有效载荷，"
            "不是进程 RSS。原始 JSON 保留全部样本、首次调用、初始化和整个工作进程 RSS。"
            "吞吐按查询向量计数，不是生成 token。未声称极端尾延迟。"
        ),
    }
    return "\n".join(lines) + "\n\n" + note[language] + "\n"


def plot(summary: dict[str, Any], output: Path) -> None:
    """Save a standalone SVG without display dependencies. / 无需显示设备即可保存独立 SVG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cases = list(dict.fromkeys(row["case"] for row in summary["rows"]))
    variants = ["prefixfold", "ungrouped", "torch_sdpa", "numpy_dense"]
    colors = ["#087e8b", "#69b2be", "#b34b35", "#d6a24a"]
    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
    width = 0.19
    positions = np.arange(len(cases))
    for index, variant in enumerate(variants):
        selected = {
            (row["case"]): row
            for row in summary["rows"]
            if row["variant"] == variant and row["status"] == "passed"
        }
        values = [
            selected[case]["median_ms"] if case in selected else float("nan") for case in cases
        ]
        ax.bar(positions + (index - 1.5) * width, values, width, label=variant, color=colors[index])
    ax.set_xticks(positions, cases, rotation=20, ha="right")
    if any(row["status"] == "passed" for row in summary["rows"]):
        ax.set_yscale("log")
        ax.set_ylabel("Median CPU attention latency (ms, log scale)")
        ax.set_title("PrefixFold: all measured workloads; lower is better")
    else:
        ax.set_yticks([])
        ax.set_title("No successful measurements; see raw failure records")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(ncols=4, frameon=False)
    fig.savefig(
        output,
        metadata={
            "Date": None,
            "Description": "CPU attention latency / CPU 注意力延迟；完整数值见双语表格",
        },
    )
    plt.close(fig)
    if output.suffix == ".svg":
        # Whitespace-only normalization preserves the vector graphic and clean Git diffs.
        # 仅规范化空白，保留矢量图形并使 Git 差异检查通过。
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/benchmark.json"), help="raw JSON / 原始 JSON"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"), help="artifact directory / 产物目录"
    )
    parser.add_argument(
        "--config-source",
        type=Path,
        help="original config file if relocated / 原配置文件迁移后的路径",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="update README tables in the current directory / 更新当前目录 README 表格",
    )
    args = parser.parse_args()
    try:
        summary = summarize(json.loads(args.input.read_text()), args.config_source)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        for language in ("en", "zh"):
            content = table(summary, language)
            (args.output_dir / f"table_{language}.md").write_text(content)
            readme = Path("README.md" if language == "en" else "README_zh.md")
            if args.update_readme and readme.exists():
                original = readme.read_text()
                begin, end = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"
                if begin in original and end in original:
                    prefix, remainder = original.split(begin, 1)
                    _, suffix = remainder.split(end, 1)
                    readme.write_text(prefix + begin + "\n\n" + content + "\n" + end + suffix)
        plot(summary, args.output_dir / "latency.svg")
        print("Analysis complete / 结果分析完成")
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"Analysis failed / 分析失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
