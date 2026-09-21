"""Offline JSON/NPZ command line / 离线 JSON/NPZ 命令行。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from .attention import AttentionPlan, attend, dense_reference
from .cache import DecodeBatch, SharedPrefix


def _demo() -> dict[str, object]:
    rng = np.random.default_rng(7)
    shape = (2, 257, 32)
    prefix = SharedPrefix(
        rng.standard_normal(shape, dtype=np.float32), rng.standard_normal(shape, dtype=np.float32)
    )
    with DecodeBatch(prefix, batch_size=4, capacity=9) as batch:
        batch.load_suffix(
            rng.standard_normal((4, 2, 7, 32), dtype=np.float32),
            rng.standard_normal((4, 2, 7, 32), dtype=np.float32),
            np.array([7, 3, 0, 5], dtype=np.int64),
        )
        batch.append(
            rng.standard_normal((4, 2, 32), dtype=np.float32),
            rng.standard_normal((4, 2, 32), dtype=np.float32),
        )
        q = rng.standard_normal((4, 2, 32), dtype=np.float32)
        actual = attend(q, batch, AttentionPlan(tile_tokens=64))
        expected = dense_reference(q, batch)
        error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, atol=3e-5, rtol=3e-5):
            raise ArithmeticError("demo correctness check failed / 示例正确性检查失败")
        return {
            "status": "ok",
            "workload": "synthetic_fp32_attention_operator",
            "shape": list(actual.shape),
            "max_abs_error": error,
            "kv_payload_bytes": batch.kv_nbytes,
            "suffix_lengths": batch.lengths.tolist(),
            "message": "Grouped decode and append verified / 分组解码与追加已验证",
        }


def _run_npz(args: argparse.Namespace) -> dict[str, object]:
    source, destination = Path(args.input), Path(args.output)
    if source.resolve() == destination.resolve():
        raise ValueError("output must differ from input / 输出不得覆盖输入")
    if destination.exists():
        raise ValueError("output already exists / 输出文件已存在")
    # Limit archive expansion before NumPy allocates arrays. No pickle is accepted.
    # 在 NumPy 分配前限制压缩包展开大小，并禁止 pickle。
    with ZipFile(source) as archive:
        if sum(item.file_size for item in archive.infolist()) > args.max_input_mib * 1024 * 1024:
            raise ValueError("input exceeds uncompressed byte limit / 输入超过解压字节限制")
    with np.load(source, allow_pickle=False) as data:
        required = {"q", "prefix_k", "prefix_v", "suffix_k", "suffix_v"}
        if not required.issubset(data.files):
            raise ValueError("missing required NPZ arrays / NPZ 缺少必要数组")
        q = data["q"]
        prefix = SharedPrefix(data["prefix_k"], data["prefix_v"])
        suffix_k, suffix_v = data["suffix_k"], data["suffix_v"]
        if q.ndim != 3 or suffix_k.ndim != 4:
            raise ValueError("invalid NPZ query/suffix dimensions / NPZ query 或后缀维数非法")
        with DecodeBatch(prefix, q.shape[0], suffix_k.shape[2]) as batch:
            batch.load_suffix(suffix_k, suffix_v, data["lengths"] if "lengths" in data else None)
            output = attend(
                q, batch, AttentionPlan(args.tile_tokens, args.workspace_mib * 1024 * 1024)
            )
    # Exclusive creation prevents races overwriting existing artifacts.
    # 排他创建，避免并发覆盖已有文件。
    with destination.open("xb") as handle:
        np.savez(handle, output=output)
    return {"status": "ok", "shape": list(output.shape)}


def main(argv: list[str] | None = None) -> int:
    """Run CLI; errors exit 2 with bilingual diagnostics / 运行 CLI，错误退出码为 2。"""
    parser = argparse.ArgumentParser(
        description="Shared-prefix CPU attention / 共享前缀 CPU 注意力"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run offline numerical demo / 运行离线数值示例")
    run = sub.add_parser("attend", help="read input NPZ and write output NPZ / 读取并输出 NPZ")
    run.add_argument("input", help="input NPZ / 输入 NPZ")
    run.add_argument("output", help="new output NPZ / 新输出 NPZ")
    run.add_argument(
        "--tile-tokens", type=int, default=1024, help="maximum tile tokens / 最大分块长度"
    )
    run.add_argument(
        "--workspace-mib", type=int, default=8, help="array scratch budget MiB / 临时数组预算"
    )
    run.add_argument(
        "--max-input-mib", type=int, default=512, help="archive size limit MiB / 解压大小上限"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "attend" and args.max_input_mib < 1:
            raise ValueError("max input MiB must be positive / 输入限制必须为正")
        result = _demo() if args.command == "demo" else _run_npz(args)
    except (ValueError, RuntimeError, ArithmeticError, OSError, BadZipFile) as exc:
        print(f"error / 错误: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0
