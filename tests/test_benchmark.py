"""Benchmark contracts and actual worker execution / 基准契约与真实进程执行。"""

import copy
import json
from pathlib import Path

import pytest

from benchmark import run, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs" / "smoke.json").read_text())


def test_smoke_configuration_is_valid(config):
    validate_config(config)


@pytest.mark.parametrize(
    "field,value",
    [("schema_version", 2), ("warmup", -1), ("repetitions", 0), ("seed", True), ("cases", [])],
)
def test_invalid_configuration_fields(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("value", [None, "invalid-case"])
def test_malformed_case_has_diagnostic_error(config, value):
    config["cases"] = [value]
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    "field,value", [("batch", 0), ("heads", True), ("dim", -1), ("prefix", -1)]
)
def test_invalid_workload_dimensions(config, field, value):
    config["cases"][0][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_duplicate_workload_names_are_rejected(config):
    config["cases"].append(copy.deepcopy(config["cases"][0]))
    with pytest.raises(ValueError):
        validate_config(config)


def test_excessive_dense_allocation_is_rejected(config):
    config["cases"][0].update(batch=128, heads=32, prefix=65536, suffix=64, dim=128)
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.integration
def test_real_benchmark_preserves_samples_and_provenance(config, tmp_path):
    source, output = tmp_path / "config.json", tmp_path / "raw.json"
    source.write_text(json.dumps(config))
    assert run(config, source, output, ["prefixfold", "torch_sdpa"], timeout=30)
    raw = json.loads(output.read_text())
    assert raw["schema_version"] == 1
    assert raw["status"] == "passed"
    assert raw["run_id"] and raw["started_at"] and raw["completed_at"]
    assert len(raw["source"]["source_sha256"]) == 64
    assert raw["source"]["file_sha256"]
    assert raw["environment"]["backend"] == "cpu"
    assert raw["config"] == config
    assert len(raw["workloads"]) == 1
    workload = raw["workloads"][0]
    assert len(workload["measurement_order"]) == config["warmup"] + config["repetitions"]
    for variant in ["prefixfold", "torch_sdpa"]:
        evidence = workload["variants"][variant]
        assert evidence["status"] == "passed"
        assert evidence["correctness"]["passed"]
        assert evidence["correctness"]["atol"] == evidence["correctness"]["rtol"] == 3e-5
        assert len(evidence["samples_ms"]) == config["repetitions"]
        assert len(evidence["warmup_ms"]) == config["warmup"]
        assert all(sample > 0 for sample in evidence["samples_ms"])
        assert evidence["kv_payload_bytes"] > 0
        assert evidence["peak_rss_bytes"] > 0
