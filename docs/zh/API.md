# Python 公开 API

[English](../en/API.md) · [架构](ARCHITECTURE.md) · [主页](../../README_zh.md)

从 `prefixfold` 导入公开名称。全部注意力输入均为 CPU NumPy 数组，且必须严格满足 `dtype=np.float32`。API 不会静默把 float64、列表或 GPU 张量转换为 float32。接受非连续 float32 数组。传入的全部 K/V 和查询数组均检查 NaN 与无穷值，包括提供的后缀填充。整数配置参数拒绝布尔值。

记号：`B` 为请求数，`H` 为头数，`D` 为头维度，`P` 为共享位置数，`C` 为预留后缀容量，`S` 为提供的后缀宽度，`N` 为选中行数。`B,H,D` 为正；在形状契约允许时，`P,C,S` 可以为零。

## 完整示例

```python
import numpy as np
from prefixfold import AttentionPlan, DecodeBatch, SharedPrefix, attend, dense_reference

rng = np.random.default_rng(7)
pk = rng.normal(size=(2, 17, 8)).astype(np.float32)
pv = rng.normal(size=pk.shape).astype(np.float32)
prefix = SharedPrefix(pk, pv)
q = rng.normal(size=(2, 2, 8)).astype(np.float32)
sk = rng.normal(size=(2, 2, 2, 8)).astype(np.float32)
sv = rng.normal(size=sk.shape).astype(np.float32)

with DecodeBatch(prefix, batch_size=2, capacity=3) as batch:
    batch.load_suffix(sk, sv, lengths=np.array([1, 2], dtype=np.int64))
    output = attend(q, batch, AttentionPlan(tile_tokens=7))
    np.testing.assert_allclose(output, dense_reference(q, batch), atol=3e-5, rtol=3e-5)
    batch.append(sk[:1, :, 0], sv[:1, :, 0], indices=[0])
    np.testing.assert_array_equal(batch.lengths, [2, 2])
    updated = attend(q, batch)
    np.testing.assert_allclose(updated, dense_reference(q, batch), atol=3e-5, rtol=3e-5)
    print(updated.shape, batch.kv_nbytes)

with DecodeBatch(prefix, batch_size=1, capacity=0) as other:
    assert attend(q[:1], other).shape == (1, 2, 8)
```

第一个批次关闭后，第二个批次继续复用同一前缀。示例真实执行注意力计算和状态变更，无须下载模型。

## SharedPrefix

`SharedPrefix(k, v)` 为形状一致的 `(H,P,D)` 数组创建不可变自有副本。它不推断或验证不同请求是否来自相同文本或模型上下文；前缀位置编码和相等性由调用者负责。

| 成员 | 返回值或行为 |
|---|---|
| `shape` | 元组 `(H,P,D)` |
| `nbytes` | K/V 数组有效载荷字节：两个 float32 数组共 `8*H*P*D` |

错误的数据类型/维数、形状不一致、非有限值或 `H,D` 为零时抛出 `ValueError`。允许 `P` 为空。不得修改下划线私有字段。生命周期由 Python 引用管理，没有显式前缀失效或淘汰 API。

## DecodeBatch

`DecodeBatch(prefix, batch_size, capacity)` 引用传入的 `SharedPrefix` 并分配私有 K/V 容量。要求 `batch_size > 0`、`capacity >= 0`。配置属性 `batch_size`、`heads`、`dim`、`capacity` 描述已分配的布局；直接为它们重新赋值不属于支持的调整容量操作。

| 成员 | 返回值或行为 |
|---|---|
| `lengths` | 新的 `(B,)` int64 数组，表示当前有效后缀长度 |
| `kv_nbytes` | `prefix.nbytes + 8*B*H*C*D + 8*B`，包含预留 K/V 与 int64 长度 |
| `load_suffix(k, v, lengths=None)` | 从 `(B,H,S,D)` 数组替换有效后缀，要求 `S <= C`；返回 `None` |
| `append(k, v, indices=None)` | 从 `(N,H,D)` 数组为每个选中行追加一个 Token；返回 `None` |
| `materialize()` | 新的 `(dense_k, dense_v, mask)` 数组，详见下文 |
| `close()` | 幂等释放本批次缓冲区与前缀引用；返回 `None` |
| `with DecodeBatch(...) as batch` | 上下文退出时关闭批次，异常退出也会关闭 |

`kv_nbytes` 统计自有数组有效载荷，不是对象开销、进程 RSS 或峰值内存。跨多个共享同一前缀的批次直接相加会重复计算前缀；计算合计时应减去重复前缀字节。

`load_suffix` 要求 K/V 形状一致。不传 `lengths` 时，每行长度均为 `S`；否则必须提供 `(B,)` 整数 NumPy 数组，且 `0 <= lengths[i] <= S`。输入数据与长度均被复制。合法加载后清零未使用后缀容量，避免无效填充影响后续算术。调用者提供的填充即使被忽略，也必须是有限值。加载 `S=0` 可清空所有后缀长度，同时保留容量和前缀。验证失败时旧状态不变。

`append` 默认按顺序选择全部行，要求 `N=B`。显式索引可使用一维整数 NumPy 数组或 `[2, 0]` 这样的整数序列，顺序决定输入行映射到哪个请求。每个索引必须唯一且位于 `[0,B)`，所有选中行至少有一个空闲槽位。全部验证在写入前完成，因此某行已满或索引非法会拒绝整次追加。空选择的无操作调用请使用 `np.empty(0, dtype=np.int64)`，配合形状为 `(0,H,D)` 的 K/V。

`materialize` 使用 `L=P+max(lengths)`，返回 `(B,H,L,D)` float32 K/V 和 `(B,L)` 布尔掩码。`True` 表示可见键。前缀在各请求间被显式复制，后缀填充不是有效键。对 `(B,H,1,D)` 的 PyTorch 查询，应将掩码扩展为 `(B,1,1,L)`，并使用 `is_causal=False`，因为传入缓存已经是可见历史。该接口适合互操作与调试，但有分配成本。

`close` 后，缓冲区操作、`lengths`、`kv_nbytes` 与注意力计算均以 `RuntimeError` 拒绝访问。允许重复 `close`，共享同一前缀的其他批次仍可正常使用。

## AttentionPlan

`AttentionPlan(tile_tokens=1024, workspace_bytes=8*1024*1024)` 是不可变 dataclass，两个字段必须为正整数。`effective_tile(batch_size, dim)` 根据 `4*B*(4*T+8*D+32)` 字节估计缩小请求的块宽；如果连 `T=1` 都无法容纳，则抛出 `ValueError`。

预算估计显式临时数组，不包含缓存/输入、最终输出、BLAS 内部空间、Python 开销或分配器效果。它不是进程内存硬限制或实测峰值内存。同一计划对象可复用于不同批次，每次根据该批次维度计算有效块宽。

## attend

`attend(q, batch, plan=None, *, grouped=True) -> ndarray[float32]`

`q` 与输出形状均为 `(B,H,D)`。省略计划时使用默认值。`grouped` 必须为 Python 布尔值；`True` 在共享前缀矩阵乘法中合并请求；`False` 逐请求计算前缀，同时保留相同后缀路径。两种模式在浮点误差范围内计算相同的缩放点积注意力，不会根据基准结果自动选择实现。

每个查询关注全部前缀位置及自身有效后缀。每个查询至少需要一个键；`P=0` 且任意后缀长度为零时抛出 `ValueError`。接口不包含 dropout、未来位置掩码、任意掩码参数、分组查询注意力或梯度计算。

验证与完整计算都持有批次锁，防止追加、加载或关闭交错执行。函数不改变 `q` 或缓存。float32 输入有限是必要条件，但不能保证中间算术有限；分数或累加溢出会抛出 `FloatingPointError`。非有限输入以 `ValueError` 拒绝，关闭的批次抛出 `RuntimeError`。分配失败不会被隐藏。调用者必须避免并发写入 `q`。

## dense_reference

`dense_reference(q, batch) -> ndarray[float64]`

它使用与 `attend` 相同的有效键和查询形状，持有批次锁，逐请求、逐头用 float64 计算稠密稳定 softmax。返回 `(B,H,D)` float64 输出，不修改批次。适用于可管理形状的数值验证；它不是实用性能基线，也不受 `AttentionPlan` 临时空间估计约束。声明的 float32 对照容差为 `atol=3e-5, rtol=3e-5`。

## CLI 与 NPZ 文件

`python -m prefixfold demo` 或 `prefixfold demo` 使用确定性合成数据执行追加、分组注意力及 FP64 对比。成功时输出 JSON，包含 `status`、`shape`、`max_abs_error`、`kv_payload_bytes` 和 `suffix_lengths`。

`prefixfold attend input.npz output.npz --tile-tokens 1024 --workspace-mib 8 --max-input-mib 512` 读取下列数组：

| NPZ 键 | 形状 | 类型 |
|---|---|---|
| `q` | `(B,H,D)` | float32 |
| `prefix_k`, `prefix_v` | `(H,P,D)` | float32 |
| `suffix_k`, `suffix_v` | `(B,H,S,D)` | float32 |
| `lengths`（可选） | `(B,)`，数值 0..S | integer |

输出 NPZ 包含 `(B,H,D)` float32 的 `output`。省略 `lengths` 时使用全部提供的后缀位置。加载使用 `allow_pickle=False`，并在读取前限制声明的解压大小，仅面向可信文件。输出路径必须是新文件且与输入不同。错误向 stderr 输出双语诊断并退出 2；成功输出 JSON 并退出 0。内存分配失败不会被转换为虚假结果。
