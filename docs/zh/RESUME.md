# 项目展示与证据

[English](../en/RESUME.md) · [实验](EXPERIMENTS.md) · [研究](RESEARCH.md)

我把 PrefixFold 作为一个聚焦的系统作品维护，希望实现、失败情况与实测边界可以一起接受检查。以下材料描述仓库已经验证的行为，不声称实际部署、用户采用、论文发表或个人学习过程已经完成。

## 一句话介绍

**PrefixFold** 是一个使用 Python 类型标注与 NumPy 的 CPU 共享前缀精确解码注意力实现，将可复用 KV 所有权、在线 softmax 与可复现的延迟、载荷和正确性实验结合起来。

## 简历条目

1. 实现基于分组矩阵乘法和稳定分块 softmax 归约的 CPU 共享前缀解码注意力；在 Apple M1 Pro 请求单线程、固定 float32 场景（`B=16,H=4,D=64`，4,096 个共享位置与 128 个私有位置）中，每种实现测量 30 次，将算子中位延迟从 **PyTorch 2.8 SDPA 的 7.1603 ms 降至 2.0437 ms**，比值为 **3.5036 倍**。
2. 设计不可变共享前缀所有权与固定容量的不等长后缀存储，实现验证后的原子追加和覆盖完整注意力调用的锁；同一场景中，实际 K/V 数组载荷为 **12 MiB，对比复制前缀的稠密 KV 为 132 MiB，减少 90.9091%**，且多个可独立关闭的批次能够复用同一前缀。
3. 构建包含 **4 种实现、7 个工作负载、840 个原始耗时样本** 的可复现 CPU 基准，记录源码/配置哈希、随机实现顺序及独立准备/首次调用/RSS 数据；全部 28 个实现与场景组合通过相对 float64 的 `atol=3e-5, rtol=3e-5` 检查，并公开保留优化路径慢于 SDPA 的三个工作负载。
4. 补充离线数值与生命周期回归测试、CLI、带类型的公开 API、锁定依赖和语义一致的中英文维护文档；在保持后缀路径不变的条件下验证分组机制，并通过 **360 个额外实测样本** 研究 **256/1,024/4,096** 三种块宽，保留原始主目标与主实验。

上述比值仅描述 CPU 注意力计算。载荷减少不等于进程 RSS 减少，查询向量吞吐不等于生成 Token 吞吐。机制复现已有 Cascade/Hydragen 与在线 softmax 思路；贡献在于 CPU 实现、生命周期契约、受控消融和实验证据。

## 证据索引

| 结论 | 实现 | 验证与原始证据 |
|---|---|---|
| 条目 1：分组精确计算与主实验延迟 | [attention.py](../../src/prefixfold/attention.py)，`attend`、`_update`、`AttentionPlan` | [冻结目标](../../research/targets.json)、[主配置](../../configs/benchmark.json)、[原始运行](../../results/benchmark.json)、[汇总](../../results/summary.json)，场景 `shared_b16_p4096` |
| 条目 2：所有权、容量、追加、清理与载荷 | [cache.py](../../src/prefixfold/cache.py)，`SharedPrefix`、`DecodeBatch` | [缓存测试](../../tests/test_cache.py)、[验证测试](../../tests/test_validation.py)、原始 `kv_payload_bytes`；主场景分别为 12,582,912 与 138,412,032 字节 |
| 条目 3：实用基线、全部样本、正确性与不利情况 | [benchmark.py](../../benchmark.py)、[analyze.py](../../scripts/analyze.py)、[float64 参考](../../src/prefixfold/attention.py) | [数值测试](../../tests/test_attention.py)、[完整表格](../../results/table_zh.md)、原始 `measurement_order`、`samples_ms`、`correctness`、`setup`、`first_call_ms`、`peak_rss_bytes` |
| 条目 4：工程接口与敏感性实验 | [公开 API](API.md)、[CLI](../../src/prefixfold/cli.py)、[锁文件](../../uv.lock)、[CI 配置](../../.github/workflows/ci.yml) | [集成测试](../../tests/test_integration.py)、[tile256](../../results/tile256.json)、[tile1024](../../results/tile1024.json)、[tile4096](../../results/tile4096.json)；最终实际检查见[发布文档](RELEASE.md) |

主基准源码 revision：`70e7ed50e51b4de74a9d35c358068aa689a939c5`；源码清单哈希：`c0003d3eaf2887a4b8a026e9139ea2cc783d76487fa219f9bdaeac15c9b7d500`。主运行 ID：`20260921T121914Z-56a3bc9a`。CI 配置是仓库文件，配置存在本身不证明远端 CI 已运行。

## 简短项目讲解

**为什么选这个问题。** 我选择了一个能在普通 CPU 硬件上验证的数据复用问题。多个解码请求可能共享相同的投影前缀 KV；前缀只存一份可以节省容量，进一步的问题是让矩阵乘法利用共享访问是否能降低注意力延迟。

**核心实现怎样工作。** 前缀复制一次后保持不可变；批次维护独立的有界后缀缓冲区与长度。在每个前缀块上合并请求查询，使用运行最大值、指数和与加权 Value 分子保留足够信息，在实数算术下精确合并前缀和后缀注意力。最终归一化返回每个请求/头的一个输出。输入验证、锁与清理让公开 API 支持重复使用。

**关键取舍。** 分组可以改善复用与矩阵形状，但不减少查询键点积数量。分块限制显式分数存储，同时增加循环和归约。NumPy 让机制便于阅读；融合 CPU 代码可能降低开销，但增加实现与构建维护面。前缀相等性与位置编码仍由调用者保证。

**最有解释力的实验。** 固定主形状中，分组注意力为 2.0437 ms，SDPA 为 7.1603 ms，而同一引擎关闭前缀分组后为 7.2138 ms。后一个对照保持后缀代码和 KV 载荷不变，因此能把收益联系到预定机制。正确性通过独立 float64 路径检查，并保留每个有效计时样本。

**失败与局限。** 没有前缀、只有一个请求或前缀很短时，优化路径更慢。代码审查还发现，无效后缀填充中的极大有限值可能在应用掩码前就引发溢出；现在验证加载时会清零未使用存储，并有对应回归测试。有效算术超出 float32 范围时仍明确拒绝。实验是一台 CPU 上的合成算子工作负载，不能支持模型质量、生成 Token、生产、GPU 或普遍平台收益的结论。

**下一步研究问题。** 经独立评估的实现选择规则能否在不利形状下切换至稠密 SDPA？不同热状态和独立会话中的比值有多稳定？融合 CPU 内核能否在保持相同误差契约的同时降低开销？真实投影 KV 分布如何影响性能交叉点？这些是后续问题，不是当前成果。

## 简短演示

```sh
make demo
.venv/bin/python -m pytest tests/test_attention.py tests/test_cache.py -q
.venv/bin/python benchmark.py --config configs/benchmark.json --output results/interview-run.json
```

先运行默认示例，展示 `attention.py` 中的状态更新，再从原始结果解释一个正面场景和一个负面场景。已有记录证明的是本次运行；新的运行会生成自己的记录，不保证得到完全相同的耗时数字。
