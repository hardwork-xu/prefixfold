# Security and limits

[简体中文](SECURITY_zh.md)

I support the documented local CPU operator scope. PrefixFold is not a multi-tenant service, sandbox, cryptographic cache or production LLM runtime.

- Only finite FP32 arrays and supported shapes are accepted. Arithmetic overflow is an error. Callers remain responsible for semantic prefix equality, positional encoding, memory budgets and keeping supplied query arrays stable during a call.
- Suffix capacity is fixed. The attention workspace setting bounds a conservative explicit-array estimate; Python, NumPy, BLAS and process RSS are outside that bound.
- NPZ input disables pickle and checks declared decompressed archive size (512 MiB default). Use trusted files: this is not a complete defense against malicious archive headers or denial of service. The CLI refuses to overwrite input or existing output.
- Batch locks serialize updates and attention. Independent processes and external array mutation are outside that lock. No persistent cache or network listener is created.

For a sensitive report, use the repository's private vulnerability-reporting channel if enabled; otherwise open a minimal issue requesting private coordination, without publishing exploit details or personal data. No private contact address is included here.
