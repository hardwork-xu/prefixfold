# 发布准备与状态

[English](../en/RELEASE.md) · [主页](../../README_zh.md)

我将 PrefixFold 0.1.0 作为聚焦、可复现的 CPU 数值库维护。发布准备以实现、证据和已知边界判断，不使用“生产成熟”标签代替验收。

## 状态

**已核验公开源码及远端 CI：**[首次成功工作流](https://github.com/hardwork-xu/shared-prefix-attention/actions/runs/35600203428)的 Python 3.11、3.12、3.13 三项任务及 Linux Docker 构建/演示任务全部通过。[机器可读记录](../../results/publication.json)保存实际被测 revision 和任务链接。本地 macOS 因无 Docker 仍未执行容器验证。版本化构建产物发布后可从仓库 Releases 页面获取；未执行包索引发布或公网服务部署。

公开仓库：[hardwork-xu/shared-prefix-attention](https://github.com/hardwork-xu/shared-prefix-attention)。外部核验完成后写入发布记录。

本地实现、测试和主基准已经运行。打包、干净安装和文档/隐私检查记录于 [acceptance.json](../../results/acceptance.json)。实测 Mac 未安装 Docker：**本地容器验证未执行**。仓库包含 Linux CPU 容器和普通 GitHub Actions 配置，已核验的外部运行记录见上文。

公开仓库与发布只使用项目文件及隐私安全的 Git 身份。私人联系地址、工作站路径、私有数据集、令牌和模型权重不应进入发布。个人溯源记录存放在本仓库之外。项目不需要付费基础设施或公网服务。

## 安装、演示与构建

```sh
python3 -m venv .venv
.venv/bin/python -m pip install uv==0.8.22
.venv/bin/uv sync --frozen --group bench
make demo
make test check
make build
```

构建生成 `dist/prefixfold-0.1.0-py3-none-any.whl` 和 `dist/prefixfold-0.1.0.tar.gz`，不将构建产物纳入源码追踪。干净解释器安装 wheel 后可运行 `python -m prefixfold demo`；`make validate` 使用 `-I` 验证，避免意外导入当前检出源码。

```sh
docker build -t prefixfold:0.1.0 .
docker run --rm prefixfold:0.1.0
```

Docker 路径只从冻结锁文件安装 NumPy 运行时，并以非特权用户运行，不含 GPU 路径。基准依赖属于独立开发分组，宿主机 GPU 可用性与该镜像无关。

## 0.1.0 发布说明草稿

- 新增 FP32 CPU 共享前缀单 token 注意力，包含分组前缀矩阵乘法、稳定分块 softmax 和不等长后缀掩码。
- 新增不可变前缀所有权、固定容量原子后缀追加、同步读取、显式清理、Python API 和离线 NPZ/JSON CLI。
- 提供七个工作负载、四种实现的主实验及后续分块敏感性分析。M1 Pro 上固定 B16/H4/P4096/S128/D64 场景的 K/V 为 12 MiB，稠密基线为 132 MiB，计算中位数相对 PyTorch CPU SDPA 加速 3.50 倍；主实验 28 个数值比较均通过冻结容差。保留小规模/低共享场景退化。
- 提供双语研究、维护和 API 文档、MIT 许可、核验归属、锁定依赖、CPU CI 及本地验收日志。

已知限制：仅 CPU FP32 算子、合成工作负载证据、单层前缀、不声称模型生成性能、无 GPU 实现、无自动路径选择、无进程硬内存限制，实测宿主机未执行本地 Docker。

## 公开元数据

一句话介绍：**CPU 共享前缀精确解码注意力，具备有界 KV 所有权管理与可复现基准。**

GitHub About：**CPU shared-prefix attention with immutable KV prefixes, ragged suffixes, exact softmax merging, and reproducible PyTorch baselines. CPU 共享前缀精确注意力与可复现实验。**

建议 Topics：`attention`、`cpu`、`kv-cache`、`numpy`、`pytorch`、`benchmark`、`reproducible-research`、`inference`。

[CITATION.cff](../../CITATION.cff)使用核实的公开作者名称与软件版本。CFF 1.2.0 格式验证属于验收，不虚构 DOI 或论文。算法应归属原始论文。发布包含 [MIT 许可证](../../LICENSE)、[第三方说明](../../NOTICE.md)、[贡献指南](../../CONTRIBUTING_zh.md)与[安全范围](../../SECURITY_zh.md)。

包索引发布、生产部署声明、购买基础设施，以及下载/集成模型不属于本次发布已执行范围。GitHub 源码发布不意味着这些操作也已完成。
