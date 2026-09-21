"""Offline fixtures / 离线测试夹具。"""

import numpy as np
import pytest

from prefixfold import DecodeBatch, SharedPrefix


@pytest.fixture
def rng():
    """Return a deterministic generator / 返回确定性随机数生成器。"""
    return np.random.default_rng(7021)


@pytest.fixture
def batch_factory(rng):
    """Create batches and release their storage / 创建批次并释放其存储。"""
    allocated = []

    def make(batch_size=3, heads=2, prefix_tokens=11, suffix_tokens=5, dim=7):
        prefix_k = rng.normal(size=(heads, prefix_tokens, dim)).astype(np.float32)
        prefix_v = rng.normal(size=(heads, prefix_tokens, dim)).astype(np.float32)
        batch = DecodeBatch(SharedPrefix(prefix_k, prefix_v), batch_size, suffix_tokens)
        suffix_k = rng.normal(size=(batch_size, heads, suffix_tokens, dim)).astype(np.float32)
        suffix_v = rng.normal(size=(batch_size, heads, suffix_tokens, dim)).astype(np.float32)
        batch.load_suffix(suffix_k, suffix_v)
        allocated.append(batch)
        return batch

    yield make
    for batch in allocated:
        batch.close()
