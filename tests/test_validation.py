"""Public input contracts and failures / 公共输入契约与失败处理。"""

import numpy as np
import pytest

from prefixfold import AttentionPlan, DecodeBatch, SharedPrefix, attend


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4), (0, 2, 3), (1, 2, 0)])
def test_prefix_rejects_invalid_shape(shape):
    x = np.ones(shape, np.float32)
    with pytest.raises(ValueError):
        SharedPrefix(x, x)


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int32])
def test_prefix_rejects_non_float32(dtype):
    x = np.ones((1, 3, 2), dtype)
    with pytest.raises((TypeError, ValueError)):
        SharedPrefix(x, x)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("which", ["key", "value"])
def test_prefix_rejects_nonfinite(invalid, which):
    k = np.ones((1, 3, 2), np.float32)
    v = k.copy()
    (k if which == "key" else v)[0, 0, 0] = invalid
    with pytest.raises(ValueError):
        SharedPrefix(k, v)


def test_prefix_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        SharedPrefix(np.ones((1, 3, 2), np.float32), np.ones((1, 4, 2), np.float32))


@pytest.mark.parametrize("batch_size,capacity", [(0, 1), (-1, 1), (1, -1), (1.5, 1), (1, 2.5)])
def test_invalid_batch_dimensions(batch_size, capacity):
    x = np.ones((1, 3, 2), np.float32)
    with pytest.raises((TypeError, ValueError)):
        DecodeBatch(SharedPrefix(x, x), batch_size, capacity)


@pytest.mark.parametrize("tile", [0, -1, 1.5, "7"])
def test_invalid_tile_configuration(tile):
    with pytest.raises((TypeError, ValueError)):
        AttentionPlan(tile_tokens=tile)


@pytest.mark.parametrize("workspace", [0, -1, 1.5])
def test_invalid_workspace_configuration(workspace):
    with pytest.raises((TypeError, ValueError)):
        AttentionPlan(workspace_bytes=workspace)


def test_insufficient_workspace_is_reported(batch_factory):
    batch = batch_factory()
    with pytest.raises(ValueError):
        attend(np.ones((3, 2, 7), np.float32), batch, AttentionPlan(workspace_bytes=1))


@pytest.mark.parametrize("shape", [(3, 7), (2, 2, 7), (3, 1, 7), (3, 2, 8)])
def test_query_shape_validation(batch_factory, shape):
    with pytest.raises(ValueError):
        attend(np.ones(shape, np.float32), batch_factory())


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_query_finite_validation(batch_factory, invalid):
    q = np.ones((3, 2, 7), np.float32)
    q[1, 1, 3] = invalid
    with pytest.raises(ValueError):
        attend(q, batch_factory())


def test_query_dtype_validation(batch_factory):
    with pytest.raises((TypeError, ValueError)):
        attend(np.ones((3, 2, 7), np.float64), batch_factory())


@pytest.mark.parametrize("shape", [(2, 1, 5, 2), (1, 1, 2, 2), (2, 2, 2, 2), (2, 1, 2, 3)])
def test_suffix_shape_and_capacity_validation(batch_factory, shape):
    batch = batch_factory(2, 1, 3, 4, 2)
    with pytest.raises(ValueError):
        batch.load_suffix(np.ones(shape, np.float32), np.ones(shape, np.float32))


@pytest.mark.parametrize("lengths", [[0], [0, 5], [-1, 2], [0.5, 2.0], [[1, 2]]])
def test_invalid_lengths_are_rejected(batch_factory, lengths):
    batch = batch_factory(2, 1, 3, 4, 2)
    suffix = np.ones((2, 1, 4, 2), np.float32)
    with pytest.raises((TypeError, ValueError)):
        batch.load_suffix(suffix, suffix, np.asarray(lengths))


@pytest.mark.parametrize("indices", [[0, 0], [-1, 1], [0, 3], [0], [0.5, 1.0], [[0, 1]]])
def test_invalid_append_indices_are_transactional(batch_factory, indices):
    batch = batch_factory(3, 1, 3, 2, 2)
    empty = np.empty((3, 1, 0, 2), np.float32)
    batch.load_suffix(empty, empty)
    before = batch.materialize()
    kv = np.ones((2, 1, 2), np.float32)
    with pytest.raises((TypeError, ValueError)):
        batch.append(kv, kv, indices=indices)
    np.testing.assert_array_equal(batch.lengths, 0)
    for actual, saved in zip(batch.materialize(), before, strict=True):
        np.testing.assert_array_equal(actual, saved)


def test_invalid_append_values_are_transactional(batch_factory):
    batch = batch_factory(2, 1, 3, 2, 2)
    empty = np.empty((2, 1, 0, 2), np.float32)
    batch.load_suffix(empty, empty)
    before = batch.materialize()
    k = np.ones((2, 1, 2), np.float32)
    v = k.copy()
    v[1, 0, 1] = np.nan
    with pytest.raises(ValueError):
        batch.append(k, v)
    for actual, saved in zip(batch.materialize(), before, strict=True):
        np.testing.assert_array_equal(actual, saved)


def test_nonfinite_suffix_padding_is_rejected(batch_factory):
    batch = batch_factory(2, 1, 3, 2, 2)
    k = np.ones((2, 1, 2, 2), np.float32)
    v = k.copy()
    v[1, 0, 1, 1] = np.inf
    with pytest.raises(ValueError):
        batch.load_suffix(k, v, lengths=np.array([1, 1]))


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int64])
def test_suffix_dtype_validation(batch_factory, dtype):
    batch = batch_factory(2, 1, 3, 2, 2)
    kv = np.ones((2, 1, 2, 2), dtype)
    with pytest.raises((TypeError, ValueError)):
        batch.load_suffix(kv, kv)


def test_suffix_key_value_shape_mismatch(batch_factory):
    batch = batch_factory(2, 1, 3, 4, 2)
    with pytest.raises(ValueError):
        batch.load_suffix(np.ones((2, 1, 2, 2), np.float32), np.ones((2, 1, 3, 2), np.float32))


@pytest.mark.parametrize("shape", [(2, 2), (1, 1, 2), (2, 2, 2), (2, 1, 3)])
def test_append_shape_validation(batch_factory, shape):
    batch = batch_factory(2, 1, 3, 2, 2)
    empty = np.empty((2, 1, 0, 2), np.float32)
    batch.load_suffix(empty, empty)
    kv = np.ones(shape, np.float32)
    with pytest.raises(ValueError):
        batch.append(kv, kv)
    np.testing.assert_array_equal(batch.lengths, 0)


def test_append_dtype_validation(batch_factory):
    batch = batch_factory(2, 1, 3, 2, 2)
    empty = np.empty((2, 1, 0, 2), np.float32)
    batch.load_suffix(empty, empty)
    kv = np.ones((2, 1, 2), np.float64)
    with pytest.raises((TypeError, ValueError)):
        batch.append(kv, kv)
    np.testing.assert_array_equal(batch.lengths, 0)
