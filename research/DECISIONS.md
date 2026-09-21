# Decisions / 决策

1. The primary problem is CPU decode attention over a long shared prefix. Speculative decoding would require a draft/target model pair and statistically valid generation tests; quantization adds model-quality calibration. This operator can be verified directly against float64 and practical CPU SDPA on the available M1 Pro.
   主问题是长共享前缀的 CPU 解码注意力。投机解码需要草稿/目标模型和采样分布验证，量化需要模型质量校准；当前算子可直接用 float64 与实际 CPU SDPA 验证。
2. Reproduce known Cascade/Hydragen grouping and online softmax. The project contribution is a CPU library with immutable prefix ownership, bounded ragged suffix lifecycle, configurable scratch planning, ablation and reproducible measurements. It is not a new attention algorithm.
   复现已有 Cascade/Hydragen 分组及在线 softmax；贡献是 CPU 库、不可变前缀所有权、有限容量可变后缀、临时空间规划、消融与可复现实验，不宣称新注意力算法。
3. Freeze `targets.json` before timing. The primary target is >=80% allocated KV payload reduction; >=1.25x latency improvement is a separate target. Failure remains a valid published finding.
   计时前冻结目标；主目标是 KV 分配字节减少至少 80%，至少 1.25 倍加速为独立目标；未达标也保留结论。
4. Use NumPy 2.2.6 / Python 3.11–3.13 and PyTorch 2.8.0 CPU baseline. NumPy supplies BLAS, vector arithmetic and storage; PrefixFold supplies partitioning, grouping, state merge and lifecycle. Linux CPU containers are configured but cannot be tested locally without Docker.
   使用固定版本 NumPy 和 CPU PyTorch 基线；底层矩阵乘法由 BLAS 提供，本项目实现分区、分组、状态合并与生命周期。宿主机无 Docker，不能声称本地容器已通过验证。
5. I publish only project files under a verified public identity. Repository-local GitHub noreply email keeps personal contact details out of commits. Local paths and private provenance stay outside the repository; global Git identity is left unchanged.
   我只公开项目文件并使用核实的公开身份。仓库本地 GitHub noreply 邮箱避免在提交中暴露私人联系方式；本地路径与私人溯源记录保留在仓库之外，不修改全局 Git 身份。
