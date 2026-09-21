# 代码导读

[English](../en/WALKTHROUGH.md) · [API](API.md) · [架构](ARCHITECTURE.md)

我会从一个小请求入手，依次追踪所有权、验证、归约以及参考实现对照，以此维护项目。每个边界最值得理解的问题是：下一步可以依赖哪些不变量？

## 1. 从公开包入口开始

[__init__.py](../../src/prefixfold/__init__.py) 导出 `SharedPrefix`、`DecodeBatch`、`AttentionPlan`、`attend` 与 `dense_reference`。[API](API.md) 中的完整示例只使用这些名称。[pyproject.toml](../../pyproject.toml) 声明的命令行入口调用 `prefixfold.cli:main`；默认示例最终构造相同缓存对象，调用同一个 `attend` 函数。两种语言和接口共用一份数值实现。

## 2. 建立自有、合法的存储

[cache.py](../../src/prefixfold/cache.py) 中的 `_array` 拒绝错误的 ndarray 类型或维数，并扫描非有限值。虽然 Python 中 `bool` 是 `int` 的子类，`_positive_int` 仍拒绝把布尔值作为维度。

`SharedPrefix.__init__` 检查形状一致、头数和维度为正，复制为连续数组并标记只读。此后前缀的公开契约不再依赖调用者原始缓冲区。允许空前缀，是因为之后的私有后缀可能提供全部有效键。

`DecodeBatch.__init__` 预分配后缀 K/V 与长度。接着阅读 `load_suffix`：它在改变批次前验证两份 K/V 与全部长度。然后阅读 `append`：验证选中行和形状，获取各行下一个位置，拒绝任何已满行，写入 K/V，再增加长度。关键不变量是每个请求都满足 `0 <= lengths[i] <= capacity`。

[test_cache.py](../../tests/test_cache.py) 中最能说明契约的测试包括：修改构造输入不能改变自有数据；返回副本不能修改缓存；一个请求容量溢出不能导致另一个请求部分追加；关闭批次不能让共享同一前缀的另一个批次失效。

## 3. 验证查询并选择分块

[attention.py](../../src/prefixfold/attention.py) 的 `attend` 先检查批次、计划对象与布尔分组开关。在批次锁内，`_validate_query` 检查 `(B,H,D)`、float32 与有限输入。如果前缀为空且任意行的后缀也为空，归一化注意力没有定义，抛出 `ValueError`。

`AttentionPlan.effective_tile` 把临时空间估计换算为块宽。不能直接把预算当作进程内存上限：估计只针对显式临时数组，不包括输入/输出或 BLAS 内部空间。请求的分块大小只是上限，并不保证每个块都有这么多位置。

## 4. 跟踪在线状态

对每个头，`attend` 初始化 `m=-inf`、`total=0`、`numerator=0`。前缀块使用 `queries @ prefix_keys.T`；后缀块使用逐请求批量乘法和长度掩码。`_update` 是数值核心：

1. 取旧状态最大值与新块逐行最大值的较大者。
2. 用 `exp(old_max - new_max)` 重新缩放已有质量与分子。
3. 从块分数中减去新最大值，再原地取指数。
4. 累加块的概率质量以及概率加权的 Value。

处理有效位置集合 $A$ 后，状态必须表示：

$$m=\max_{j\in A}x_j,\quad total=\sum_{j\in A}e^{x_j-m},\quad numerator=\sum_{j\in A}e^{x_j-m}v_j.$$

最终用 `numerator / total` 得到注意力。空掩码贡献的质量为零。有限检查与 `np.errstate` 会显式暴露 float32 溢出，即使输入本身全部有限。`attend` 失败时可能已分配临时输出，但不会改变缓存。

分组开关把前缀请求宽度从 `B` 改为 `1`。后缀循环位于分组循环之外，在两种模式中使用相同的批量计算。因此消融对比前缀矩阵分组及其循环开销，同时保持后缀批处理不变。

## 5. 使用独立正确性检查

`dense_reference` 逐请求、逐头拼接有效前缀/后缀，转换为 float64，计算常规稳定 softmax，再写入 float64 输出。它不使用优化实现的在线状态路径，适合作为正确性参考，不能充当公平的高性能基线。

[test_attention.py](../../tests/test_attention.py) 还从 `materialize` 与掩码构造独立参考，检查极小或不规则维度、不等长后缀、不同分块、常量 Value、零查询对应均值、Key 平移不变性、重复读取与 float32 分数溢出。性能对比使用基准中的 CPU PyTorch SDPA 路径，不能用这个 Python 参考实现的耗时替代。

## 6. 用小形状调试不一致

安装依赖后，在仓库根目录运行：

```sh
uv run --frozen pytest tests/test_attention.py -q
uv run --frozen pytest tests/test_cache.py tests/test_validation.py -q
uv run --frozen pytest tests/test_attention.py -k ragged -x --pdb
```

出现新失败时，保存随机种子、`(B,H,P,S,D)`、实际 `lengths`、计划对象和分组开关。先比较 `materialize()` 及其掩码与预期输入；再比较 `attend(q, batch, AttentionPlan(tile_tokens=1))` 和 `dense_reference(q, batch)`。如果块大小改变误差，检查首个出现差异的块附近的重缩放和掩码；如果追加改变了未选中的行，检查索引以及写入前的容量验证。

不要只扩大容差来修复数值失败。先确定原因是错误掩码、状态不变量破坏、正常浮点重排还是有限数值范围导致的溢出。修改后重新运行受影响的数值和集成检查；已测执行路径变化时，必须重跑基准。
