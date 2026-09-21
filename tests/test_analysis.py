"""Evidence validation using retained real runs / 使用保留的真实实验验证证据完整性。"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analyze import plot, summarize, table

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw():
    return json.loads((ROOT / "results/benchmark.json").read_text())


def passed_entry(raw):
    return raw["workloads"][0]["variants"]["prefixfold"]


def test_real_run_validates_all_cases_and_samples(raw):
    summary = summarize(raw)
    assert summary["all_correct"] is True
    assert len(summary["rows"]) == len(raw["config"]["cases"]) * 4
    assert all(row["samples"] == 30 for row in summary["rows"])
    assert summary["config_sha256"] == raw["config_sha256"]
    assert (
        summary["analyzer_sha256"]
        == hashlib.sha256((ROOT / "scripts/analyze.py").read_bytes()).hexdigest()
    )


def test_config_hash_must_match_original_file(raw):
    raw["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="config hash"):
        summarize(raw)


def test_embedded_config_cannot_be_changed_without_source(raw):
    raw["config"]["repetitions"] = 1
    with pytest.raises(ValueError, match="embedded config"):
        summarize(raw)


def test_relocated_original_config_requires_explicit_path(raw, tmp_path):
    config_path = tmp_path / "relocated.json"
    config_path.write_bytes((ROOT / "configs/benchmark.json").read_bytes())
    raw["command"] = raw["command"].replace("configs/benchmark.json", str(config_path))
    with pytest.raises(ValueError, match="external config"):
        summarize(raw)
    assert summarize(raw, config_path)["all_correct"] is True


def test_missing_workload_is_rejected(raw):
    raw["workloads"].pop()
    with pytest.raises(ValueError, match="workload manifest"):
        summarize(raw)


def test_duplicate_workload_cannot_replace_missing_case(raw):
    raw["workloads"][0] = raw["workloads"][1]
    with pytest.raises(ValueError, match="workload differs"):
        summarize(raw)


def test_workload_dimensions_must_match_config(raw):
    raw["workloads"][0]["case"]["batch"] = 8
    with pytest.raises(ValueError, match="workload differs"):
        summarize(raw)


def test_missing_variant_is_rejected(raw):
    del raw["workloads"][0]["variants"]["torch_sdpa"]
    with pytest.raises(ValueError, match="variant manifest"):
        summarize(raw)


def test_incomplete_run_is_rejected(raw):
    raw["status"] = "running"
    with pytest.raises(ValueError, match="incomplete"):
        summarize(raw)


@pytest.mark.parametrize("field", ["samples_ms", "warmup_ms"])
def test_passed_sample_counts_must_match_config(raw, field):
    passed_entry(raw)[field].pop()
    with pytest.raises(ValueError, match="count or values"):
        summarize(raw)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1.0"])
def test_nonfinite_or_invalid_latencies_rejected(raw, value):
    passed_entry(raw)["samples_ms"][0] = value
    with pytest.raises(ValueError, match="count or values"):
        summarize(raw)


@pytest.mark.parametrize("field,value", [("passed", False), ("atol", 1e-3), ("rtol", 1e-3)])
def test_false_passed_correctness_or_changed_tolerance_rejected(raw, field, value):
    passed_entry(raw)["correctness"][field] = value
    with pytest.raises(ValueError, match="false passed"):
        summarize(raw)


def test_nonfinite_error_metric_rejected(raw):
    passed_entry(raw)["correctness"]["max_abs_error"] = float("nan")
    with pytest.raises(ValueError, match="error metric"):
        summarize(raw)


def test_source_aggregate_hash_must_match_manifest(raw):
    raw["source"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="aggregate hash"):
        summarize(raw)


def test_failed_record_is_retained_without_claiming_correctness(raw):
    raw["status"] = "failed"
    passed_entry(raw).update(
        status="failed", error="recorded worker failure / 已记录的工作进程失败"
    )
    summary = summarize(raw)
    assert summary["all_correct"] is False
    failed = next(row for row in summary["rows"] if row["status"] == "failed")
    assert failed["error"] == "recorded worker failure / 已记录的工作进程失败"
    assert "failed" in table(summary, "en")
    assert len(summary["rows"]) == 28


def test_failed_variant_cannot_be_hidden_behind_passed_run(raw):
    passed_entry(raw)["status"] = "timeout"
    with pytest.raises(ValueError, match="contradicts run pass"):
        summarize(raw)


def test_all_failed_run_retains_failures_without_fabricated_latencies(raw, tmp_path):
    raw["status"] = "failed"
    for workload in raw["workloads"]:
        for entry in workload["variants"].values():
            entry["status"] = "failed"
    summary = summarize(raw)
    assert summary["all_correct"] is False
    assert all("median_ms" not in row for row in summary["rows"])
    output = tmp_path / "all-failed.svg"
    plot(summary, output)
    assert output.is_file()


@pytest.mark.integration
@pytest.mark.parametrize("update_readme", [False, True])
def test_readme_mutation_requires_explicit_option(tmp_path, update_readme):
    before = "title\n<!-- BENCHMARK:START -->\noriginal\n<!-- BENCHMARK:END -->\n"
    for name in ("README.md", "README_zh.md"):
        (tmp_path / name).write_text(before)
    command = [
        sys.executable,
        str(ROOT / "scripts/analyze.py"),
        "--input",
        str(ROOT / "results/benchmark.json"),
        "--output-dir",
        str(tmp_path / "analysis"),
    ]
    if update_readme:
        command.append("--update-readme")
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for name in ("README.md", "README_zh.md"):
        after = (tmp_path / name).read_text()
        assert (after != before) is update_readme
    assert (tmp_path / "analysis/summary.json").is_file()
    assert (tmp_path / "analysis/latency.svg").is_file()


@pytest.mark.integration
def test_corruption_fails_without_rewriting_evidence(raw, tmp_path):
    raw["workloads"].pop()
    source = tmp_path / "incomplete.json"
    source.write_text(json.dumps(raw))
    before = source.read_bytes()
    output = tmp_path / "analysis"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze.py"),
            "--input",
            str(source),
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "incomplete workload manifest" in completed.stdout
    assert source.read_bytes() == before
    assert not output.exists()
