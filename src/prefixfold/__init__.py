"""Exact shared-prefix CPU attention / 共享前缀 CPU 精确注意力。"""

from .attention import AttentionPlan, attend, dense_reference
from .cache import DecodeBatch, SharedPrefix

__all__ = ["AttentionPlan", "DecodeBatch", "SharedPrefix", "attend", "dense_reference"]
__version__ = "0.1.0"
