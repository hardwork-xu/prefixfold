#!/usr/bin/env python3
"""Generate bilingual tables and plots from raw evidence. / 从原始证据生成双语表格与图表。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    """Summarize all valid samples without filtering. / 汇总全部有效样本，不筛除慢样本。"""
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported result schema / 不支持的结果格式")
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
        "source": raw["source"],
        "environment": raw["environment"],
        "rows": rows,
        "primary_verdict": verdict,
        "all_correct": bool(rows)
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
    ax.set_yscale("log")
    ax.set_ylabel("Median CPU attention latency (ms, log scale)")
    ax.set_title("PrefixFold: all measured workloads; lower is better")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/benchmark.json"), help="raw JSON / 原始 JSON"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"), help="artifact directory / 产物目录"
    )
    args = parser.parse_args()
    try:
        summary = summarize(json.loads(args.input.read_text()))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        for language in ("en", "zh"):
            content = table(summary, language)
            (args.output_dir / f"table_{language}.md").write_text(content)
            readme = Path("README.md" if language == "en" else "README_zh.md")
            if readme.exists():
                original = readme.read_text()
                begin, end = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"
                if begin in original and end in original:
                    prefix, remainder = original.split(begin, 1)
                    _, suffix = remainder.split(end, 1)
                    readme.write_text(prefix + begin + "\n\n" + content + "\n" + end + suffix)
        plot(summary, args.output_dir / "latency.svg")
        print("Analysis complete / 结果分析完成")
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"Analysis failed / 分析失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
