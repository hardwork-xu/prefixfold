# Maintenance / 维护约定

- Core: `src/prefixfold/{cache,attention,cli}.py`; evidence: `results/`; protocols: `research/` and `configs/`.
  核心代码、原始证据与实验协议分开维护。
- Install: `uv sync --frozen --group bench`; demo: `uv run prefixfold demo`.
  使用锁定的隔离环境，默认示例离线运行。
- Check: `make check`; test: `make test`; benchmark: `make benchmark`; build: `make build`.
  修改后运行受影响的检查。
- Keep README.md / README_zh.md and docs/en / docs/zh semantically aligned.
  公共说明、API 和核心注释提供中英双语；配置键统一英文。
- Preserve all valid benchmark samples, failures, targets, provenance and source hashes. Generated tables derive from raw JSON.
  不删除不利结果，不把理论、目标或未运行状态写成实测。
- Do not publish secrets, private paths, personal email, model weights or unrelated files. Do not change global identity, rewrite history, buy services, or deploy without explicit authorization.
  未获明确授权不得推送、公开发布、创建付费资源或部署；本地提交仅暂存相关文件。
- A change is complete when behavior, tests, bilingual docs, source-linked acceptance and relevant benchmarks agree.
  完成条件：实现、测试、双语文档、验收证据与必要实验一致。
