"""Cache ownership and transactional updates / 缓存所有权与原子更新。"""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from prefixfold import DecodeBatch, SharedPrefix, attend


def test_prefix_owns_input_and_materialization_is_a_copy():
    k = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    v = -k.copy()
    saved_k, saved_v = k.copy(), v.copy()
    prefix = SharedPrefix(k, v)
    assert prefix.shape == (1, 4, 3)
    assert prefix.nbytes == k.nbytes + v.nbytes
    k.fill(99)
    v.fill(99)
    with DecodeBatch(prefix, 2, 0) as batch:
        dense_k, dense_v, mask = batch.materialize()
        np.testing.assert_array_equal(dense_k[0], saved_k)
        np.testing.assert_array_equal(dense_v[1], saved_v)
        assert mask.all()
        dense_k.fill(0)
        dense_v.fill(0)
        np.testing.assert_array_equal(batch.materialize()[0][0], saved_k)


def test_reserved_payload_accounting(batch_factory):
    batch = batch_factory(3, 2, 11, 5, 7)
    expected = 2 * 2 * 11 * 7 * 4 + 2 * 3 * 2 * 5 * 7 * 4 + batch.lengths.nbytes
    assert batch.kv_nbytes == expected


def test_suffix_load_owns_inputs_and_lengths(batch_factory, rng):
    batch = batch_factory(2, 1, 3, 4, 2)
    k = rng.normal(size=(2, 1, 4, 2)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    lengths = np.array([1, 3])
    batch.load_suffix(k, v, lengths)
    expected = batch.materialize()
    k.fill(100)
    v.fill(100)
    lengths[:] = 0
    returned_lengths = batch.lengths
    returned_lengths[:] = 0
    np.testing.assert_array_equal(batch.lengths, [1, 3])
    for actual, saved in zip(batch.materialize(), expected, strict=True):
        np.testing.assert_array_equal(actual, saved)


def test_append_routes_rows_and_preserves_prefix(batch_factory):
    batch = batch_factory(3, 1, 2, 4, 2)
    batch.load_suffix(np.zeros((3, 1, 0, 2), np.float32), np.zeros((3, 1, 0, 2), np.float32))
    old_prefix = batch.materialize()[0].copy()
    keys = np.array([[[10, 20]], [[30, 40]]], dtype=np.float32)
    values = -keys
    batch.append(keys, values, indices=[2, 0])
    np.testing.assert_array_equal(batch.lengths, [1, 0, 1])
    k, v, mask = batch.materialize()
    np.testing.assert_array_equal(k[:, :, :2], old_prefix)
    np.testing.assert_array_equal(k[2, :, 2], keys[0])
    np.testing.assert_array_equal(k[0, :, 2], keys[1])
    np.testing.assert_array_equal(v[2, :, 2], values[0])
    np.testing.assert_array_equal(
        mask, [[True, True, True], [True, True, False], [True, True, True]]
    )


def test_append_full_default_batch_and_reuse(batch_factory):
    batch = batch_factory(2, 1, 3, 2, 2)
    empty = np.empty((2, 1, 0, 2), np.float32)
    batch.load_suffix(empty, empty)
    keys = np.ones((2, 1, 2), np.float32)
    batch.append(keys, keys)
    batch.append(keys * 2, keys * 3)
    np.testing.assert_array_equal(batch.lengths, [2, 2])
    with pytest.raises(ValueError):
        batch.append(keys, keys)
    batch.load_suffix(empty, empty)
    batch.append(keys * 4, keys * 5)
    np.testing.assert_array_equal(batch.lengths, [1, 1])
    np.testing.assert_array_equal(batch.materialize()[1][:, :, 3], keys * 5)


def test_overflow_of_one_row_does_not_change_other_rows(batch_factory):
    batch = batch_factory(2, 1, 2, 2, 3)
    suffix = np.ones((2, 1, 2, 3), np.float32)
    batch.load_suffix(suffix, suffix, np.array([0, 2]))
    before = batch.materialize()
    with pytest.raises(ValueError):
        batch.append(np.ones((2, 1, 3), np.float32), np.ones((2, 1, 3), np.float32))
    np.testing.assert_array_equal(batch.lengths, [0, 2])
    for actual, saved in zip(batch.materialize(), before, strict=True):
        np.testing.assert_array_equal(actual, saved)


def test_invalid_load_is_transactional(batch_factory):
    batch = batch_factory(2, 1, 3, 2, 2)
    before = batch.materialize()
    bad = np.ones((2, 1, 2, 2), np.float32)
    bad[1, 0, 1, 1] = np.nan
    with pytest.raises(ValueError):
        batch.load_suffix(np.zeros_like(bad), bad)
    for actual, saved in zip(batch.materialize(), before, strict=True):
        np.testing.assert_array_equal(actual, saved)


def test_empty_prefix_and_suffix_materialize_without_attention():
    empty = np.empty((1, 0, 2), dtype=np.float32)
    with DecodeBatch(SharedPrefix(empty, empty), 2, 0) as batch:
        k, v, mask = batch.materialize()
        assert k.shape == v.shape == (2, 1, 0, 2)
        assert mask.shape == (2, 0)
        assert mask.dtype == np.bool_


def test_context_manager_and_close_reject_use(batch_factory):
    batch = batch_factory(2, 1, 3, 2, 2)
    with batch:
        assert batch.materialize()[0].shape == (2, 1, 5, 2)
    batch.close()
    with pytest.raises(RuntimeError):
        batch.materialize()
    with pytest.raises(RuntimeError):
        _ = batch.lengths
    with pytest.raises(RuntimeError):
        batch.load_suffix(np.empty((2, 1, 0, 2), np.float32), np.empty((2, 1, 0, 2), np.float32))
    with pytest.raises(RuntimeError):
        batch.append(np.ones((2, 1, 2), np.float32), np.ones((2, 1, 2), np.float32))
    with pytest.raises(RuntimeError):
        attend(np.ones((2, 1, 2), np.float32), batch)


def test_closing_one_batch_preserves_shared_prefix():
    k = np.ones((1, 3, 2), np.float32)
    prefix = SharedPrefix(k, k)
    first = DecodeBatch(prefix, 1, 0)
    with DecodeBatch(prefix, 2, 0) as second:
        first.close()
        np.testing.assert_array_equal(attend(np.ones((2, 1, 2), np.float32), second), 1)


def test_close_releases_its_last_prefix_reference():
    kv = np.ones((1, 3, 2), np.float32)
    prefix = SharedPrefix(kv, kv)
    reference = weakref.ref(prefix)
    batch = DecodeBatch(prefix, 1, 1)
    del prefix
    gc.collect()
    assert reference() is not None
    batch.close()
    gc.collect()
    assert reference() is None


def test_concurrent_append_and_reads_are_consistent():
    pk = np.zeros((1, 1, 2), np.float32)
    gate = Barrier(5)
    with DecodeBatch(SharedPrefix(pk, pk), 2, 32) as batch:

        def writer(worker):
            gate.wait(timeout=5)
            for sequence in range(8):
                values = np.full((2, 1, 2), worker * 8 + sequence + 1, np.float32)
                batch.append(np.zeros_like(values), values)

        def reader():
            gate.wait(timeout=5)
            for _ in range(32):
                _, values, mask = batch.materialize()
                np.testing.assert_array_equal(mask[0], mask[1])
                np.testing.assert_array_equal(values[0], values[1])
                output = attend(np.zeros((2, 1, 2), np.float32), batch)
                np.testing.assert_array_equal(output[0], output[1])

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(writer, worker) for worker in range(4)]
            futures.append(pool.submit(reader))
            for future in futures:
                future.result(timeout=10)
        np.testing.assert_array_equal(batch.lengths, 32)
        _, values, _ = batch.materialize()
        np.testing.assert_array_equal(np.sort(values[0, 0, 1:, 0]), np.arange(1, 33))
