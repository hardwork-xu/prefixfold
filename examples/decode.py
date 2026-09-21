"""Repeated decode with shared prefix and ragged append / 共享前缀与不等长追加。"""

import numpy as np

from prefixfold import DecodeBatch, SharedPrefix, attend

rng = np.random.default_rng(42)
prefix = SharedPrefix(
    rng.standard_normal((2, 1024, 32), dtype=np.float32),
    rng.standard_normal((2, 1024, 32), dtype=np.float32),
)
with DecodeBatch(prefix, batch_size=4, capacity=8) as batch:
    for step in range(8):
        batch.append(
            rng.standard_normal((4, 2, 32), dtype=np.float32),
            rng.standard_normal((4, 2, 32), dtype=np.float32),
        )
        output = attend(rng.standard_normal((4, 2, 32), dtype=np.float32), batch)
        print(f"step / 步骤 {step}: {output.shape}; bytes / 字节 {batch.kv_nbytes}")
