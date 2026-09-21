"""Real CLI and repeated decode integration / 真实 CLI 与连续解码集成。"""

import json
import subprocess
import sys

import numpy as np
import pytest

from prefixfold import DecodeBatch, SharedPrefix, attend, dense_reference

pytestmark = pytest.mark.integration


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "prefixfold", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cli_demo():
    result = run_cli("demo")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["max_abs_error"] <= 3e-5
    assert len(payload["shape"]) == 3


def test_cli_npz_end_to_end(tmp_path, rng):
    q = rng.normal(size=(3, 2, 5)).astype(np.float32)
    pk = rng.normal(size=(2, 13, 5)).astype(np.float32)
    pv = rng.normal(size=pk.shape).astype(np.float32)
    sk = rng.normal(size=(3, 2, 4, 5)).astype(np.float32)
    sv = rng.normal(size=sk.shape).astype(np.float32)
    lengths = np.array([0, 2, 4])
    source, destination = tmp_path / "input.npz", tmp_path / "output.npz"
    np.savez(source, q=q, prefix_k=pk, prefix_v=pv, suffix_k=sk, suffix_v=sv, lengths=lengths)
    result = run_cli("attend", source, destination, "--tile-tokens", "7")
    assert result.returncode == 0, result.stderr
    with DecodeBatch(SharedPrefix(pk, pv), 3, 4) as batch:
        batch.load_suffix(sk, sv, lengths)
        expected = dense_reference(q, batch)
    with np.load(destination, allow_pickle=False) as output:
        assert "output" in output.files
        np.testing.assert_allclose(output["output"], expected, atol=3e-5, rtol=3e-5)


def test_cli_npz_default_lengths(tmp_path):
    source, destination = tmp_path / "input.npz", tmp_path / "output.npz"
    np.savez(
        source,
        q=np.ones((1, 1, 2), np.float32),
        prefix_k=np.ones((1, 2, 2), np.float32),
        prefix_v=np.ones((1, 2, 2), np.float32),
        suffix_k=np.ones((1, 1, 1, 2), np.float32),
        suffix_v=np.ones((1, 1, 1, 2), np.float32),
    )
    result = run_cli("attend", source, destination)
    assert result.returncode == 0, result.stderr
    with np.load(destination, allow_pickle=False) as output:
        np.testing.assert_allclose(output["output"], 1)


@pytest.mark.parametrize("args", [("unknown-command",), ("attend",), ("demo", "--unknown")])
def test_cli_usage_errors(args):
    result = run_cli(*args)
    assert result.returncode == 2
    assert result.stderr


def test_cli_rejects_invalid_npz_without_output(tmp_path):
    source, destination = tmp_path / "bad.npz", tmp_path / "output.npz"
    np.savez(source, q=np.zeros((1, 1, 2), np.float32))
    result = run_cli("attend", source, destination)
    assert result.returncode == 2
    assert result.stderr
    assert not destination.exists()


def test_cli_help_is_bilingual():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "demo" in result.stdout
    assert any("\u4e00" <= character <= "\u9fff" for character in result.stdout)


def test_append_attention_sequence(rng):
    pk = rng.normal(size=(2, 11, 7)).astype(np.float32)
    pv = rng.normal(size=pk.shape).astype(np.float32)
    with DecodeBatch(SharedPrefix(pk, pv), 4, 6) as batch:
        for step in range(6):
            k = rng.normal(size=(4, 2, 7)).astype(np.float32)
            v = rng.normal(size=k.shape).astype(np.float32)
            batch.append(k, v)
            q = rng.normal(size=k.shape).astype(np.float32)
            np.testing.assert_allclose(
                attend(q, batch), dense_reference(q, batch), atol=3e-5, rtol=3e-5
            )
            np.testing.assert_array_equal(batch.lengths, step + 1)
