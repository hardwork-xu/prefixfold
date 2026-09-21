# Contributing

[简体中文](CONTRIBUTING_zh.md)

I maintain PrefixFold as a focused CPU attention project. Contributions should connect a reproducible problem to a tested change.

1. Install with the locked commands in the README. Keep runtime dependencies small.
2. Add a real numerical or state-invariant regression test for algorithm/lifecycle changes. Run `make test check build`.
3. Keep English and Chinese documentation aligned. CLI diagnostics and public API descriptions are bilingual.
4. For performance claims, save the configuration, full samples, source hashes and failures. Preserve frozen targets and rerun affected cases. Do not replace an unfavorable run silently.
5. Use Conventional Commits with an English summary and Chinese body. Explain the problem, behavior change and actual validation in a pull request.

Please exclude personal data, tokens, local paths, proprietary tensors and large weights. An untested backend must be marked unverified. Read [security and limits](SECURITY.md) before handling untrusted input.
