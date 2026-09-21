"""Numerical oracle and partition invariants / 数值判定与分块不变量。"""

import tracemalloc

import numpy as np
import pytest

from prefixfold import AttentionPlan, DecodeBatch, SharedPrefix, attend, dense_reference


def independent_oracle(q, batch):
    """Evaluate each row separately in float64 / 逐行使用 float64 计算。"""
    k, v, mask = batch.materialize()
    output = np.empty(q.shape, dtype=np.float64)
    for row in range(q.shape[0]):
        for head in range(q.shape[1]):
            keys = k[row, head, mask[row]].astype(np.float64)
            values = v[row, head, mask[row]].astype(np.float64)
            scores = keys @ q[row, head].astype(np.float64) / np.sqrt(q.shape[-1])
            weights = np.exp(scores - scores.max())
            output[row, head] = weights @ values / weights.sum()
    return output


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1, 0, 1), (1, 2, 0, 7, 3), (3, 2, 13, 5, 7), (5, 1, 17, 0, 9)],
)
@pytest.mark.parametrize("tile", [1, 7, 1024])
@pytest.mark.parametrize("grouped", [False, True])
def test_matches_independent_float64_oracle(batch_factory, rng, shape, tile, grouped):
    batch_size, heads, prefix_tokens, suffix_tokens, dim = shape
    batch = batch_factory(*shape)
    q = rng.normal(size=(batch_size, heads, dim)).astype(np.float32)
    result = attend(q, batch, AttentionPlan(tile_tokens=tile), grouped=grouped)
    expected = independent_oracle(q, batch)
    assert result.shape == q.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(dense_reference(q, batch), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("prefix_tokens", [0, 9])
@pytest.mark.parametrize("grouped", [False, True])
def test_ragged_suffix_matches_oracle(batch_factory, rng, prefix_tokens, grouped):
    batch = batch_factory(4, 3, prefix_tokens, 8, 5)
    k = rng.normal(size=(4, 3, 8, 5)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    lengths = np.array([1 if prefix_tokens == 0 else 0, 3, 6, 8])
    batch.load_suffix(k, v, lengths)
    q = rng.normal(size=(4, 3, 5)).astype(np.float32)
    actual = attend(q, batch, AttentionPlan(tile_tokens=3), grouped=grouped)
    np.testing.assert_allclose(actual, independent_oracle(q, batch), rtol=3e-5, atol=3e-5)


def test_masked_suffix_values_do_not_contribute(batch_factory, rng):
    batch = batch_factory(2, 1, 3, 7, 4)
    k = np.zeros((2, 1, 7, 4), dtype=np.float32)
    v = np.ones_like(k)
    lengths = np.array([1, 4])
    batch.load_suffix(k, v, lengths)
    q = rng.normal(size=(2, 1, 4)).astype(np.float32)
    expected = attend(q, batch)
    for row, length in enumerate(lengths):
        k[row, :, length:] = 100
        v[row, :, length:] = 10000
    batch.load_suffix(k, v, lengths)
    np.testing.assert_array_equal(attend(q, batch), expected)


def test_constant_values_and_large_scores_are_stable(rng):
    k = (rng.normal(size=(2, 19, 7)) * 100).astype(np.float32)
    v = np.full(k.shape, 2.75, dtype=np.float32)
    with DecodeBatch(SharedPrefix(k, v), 3, 0) as batch:
        q = (rng.normal(size=(3, 2, 7)) * 100).astype(np.float32)
        output = attend(q, batch, AttentionPlan(tile_tokens=3))
        assert np.isfinite(output).all()
        np.testing.assert_allclose(output, 2.75, rtol=3e-5, atol=3e-5)


def test_uniform_query_is_mean_of_valid_values(batch_factory):
    batch = batch_factory(3, 2, 5, 4, 7)
    q = np.zeros((3, 2, 7), dtype=np.float32)
    _, values, mask = batch.materialize()
    expected = np.stack([values[i, :, mask[i], :].mean(axis=0) for i in range(3)])
    np.testing.assert_allclose(attend(q, batch), expected, rtol=3e-5, atol=3e-5)


def test_shared_key_shift_preserves_attention(rng):
    k = rng.normal(size=(2, 17, 5)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    shift = rng.normal(size=(2, 1, 5)).astype(np.float32)
    q = rng.normal(size=(4, 2, 5)).astype(np.float32)
    with DecodeBatch(SharedPrefix(k, v), 4, 0) as before:
        with DecodeBatch(SharedPrefix(k + shift, v), 4, 0) as after:
            np.testing.assert_allclose(attend(q, before), attend(q, after), rtol=3e-5, atol=3e-5)


def test_noncontiguous_inputs_are_valid(rng):
    k = rng.normal(size=(2, 11, 10)).astype(np.float32)[..., ::2]
    v = rng.normal(size=(2, 11, 10)).astype(np.float32)[..., ::2]
    q = rng.normal(size=(3, 2, 10)).astype(np.float32)[..., ::2]
    assert not k.flags.c_contiguous
    with DecodeBatch(SharedPrefix(k, v), 3, 0) as batch:
        np.testing.assert_allclose(
            attend(q, batch), independent_oracle(q, batch), rtol=3e-5, atol=3e-5
        )


def test_repeat_reads_do_not_mutate_state(batch_factory, rng):
    batch = batch_factory()
    q = rng.normal(size=(3, 2, 7)).astype(np.float32)
    original = batch.materialize()
    expected = attend(q, batch)
    for _ in range(3):
        np.testing.assert_array_equal(attend(q, batch), expected)
    for actual, saved in zip(batch.materialize(), original, strict=True):
        np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize("grouped", [False, True])
def test_float32_score_overflow_is_reported(grouped):
    k = np.full((1, 2, 2), np.finfo(np.float32).max, dtype=np.float32)
    with DecodeBatch(SharedPrefix(k, np.ones_like(k)), 1, 0) as batch:
        q = np.full((1, 1, 2), np.finfo(np.float32).max, dtype=np.float32)
        with pytest.raises(FloatingPointError):
            attend(q, batch, grouped=grouped)


def test_completely_empty_attention_is_rejected():
    prefix = SharedPrefix(np.empty((1, 0, 2), np.float32), np.empty((1, 0, 2), np.float32))
    with DecodeBatch(prefix, 2, 1) as batch:
        batch.append(np.ones((1, 1, 2), np.float32), np.ones((1, 1, 2), np.float32), indices=[0])
        with pytest.raises(ValueError):
            attend(np.ones((2, 1, 2), np.float32), batch)


@pytest.mark.parametrize("grouped", [False, True])
def test_partition_maximum_changes_rescale_previous_mass(grouped):
    pk = np.full((1, 9, 3), -100, np.float32)
    pv = np.full_like(pk, 7)
    sk = np.full((2, 1, 5, 3), 100, np.float32)
    sv = np.full_like(sk, -11)
    q = np.array([[[100, 100, 100]], [[-100, -100, -100]]], np.float32)
    with DecodeBatch(SharedPrefix(pk, pv), 2, 5) as batch:
        batch.load_suffix(sk, sv)
        result = attend(q, batch, AttentionPlan(tile_tokens=2), grouped=grouped)
        np.testing.assert_allclose(result[0], -11, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(result[1], 7, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("grouped", [False, True])
def test_inactive_padding_cannot_overflow_logits(grouped):
    pk = np.zeros((1, 1, 2), np.float32)
    pv = np.full_like(pk, 3)
    sk = np.zeros((2, 1, 1, 2), np.float32)
    sv = np.full_like(sk, 9)
    sk[0] = 1e30
    q = np.full((2, 1, 2), 1e30, np.float32)
    with DecodeBatch(SharedPrefix(pk, pv), 2, 1) as batch:
        batch.load_suffix(sk, sv, np.array([0, 1]))
        actual = attend(q, batch, grouped=grouped)
        np.testing.assert_allclose(actual[0], 3, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(actual[1], 6, atol=3e-5, rtol=3e-5)


def test_large_single_row_respects_array_scratch_budget():
    """Track NumPy temporaries plus Python overhead, not RSS / 跟踪临时分配，不测进程内存。"""
    tokens, budget = 1024 * 1024, 8 * 1024 * 1024
    pk = np.empty((1, 0, 1), np.float32)
    q = np.zeros((1, 1, 1), np.float32)
    sk = np.zeros((1, 1, tokens, 1), np.float32)
    sv = np.ones_like(sk)
    plan = AttentionPlan(tile_tokens=tokens, workspace_bytes=budget)
    with DecodeBatch(SharedPrefix(pk, pk), 1, tokens) as batch:
        batch.load_suffix(sk, sv)
        attend(q, batch, plan)
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        try:
            actual = attend(q, batch, plan)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not already_tracing:
                tracemalloc.stop()
        # Allow fixed Python/profiler overhead; input/cache allocation precedes tracking.
        # 为 Python 与跟踪器留固定余量；输入与缓存均在跟踪前分配。
        assert peak - baseline <= budget + 64 * 1024
        np.testing.assert_array_equal(actual, 1)
