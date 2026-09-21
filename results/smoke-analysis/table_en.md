| Workload | Variant | Median ms | SDPA / variant | KV MiB | KV reduction | Max abs error | n |
|---|---|---:|---:|---:|---:|---:|---:|
| smoke | numpy_dense | 0.0500 | 2.196× | 0.069 | 0.00% | 4.48e-07 | 2 |
| smoke | torch_sdpa | 0.1097 | 1.000× | 0.069 | 0.00% | 1.82e-07 | 2 |
| smoke | prefixfold | 0.2427 | 0.452× | 0.023 | 67.25% | 9.85e-08 | 2 |
| smoke | ungrouped | 0.5059 | 0.217× | 0.023 | 67.25% | 1.77e-07 | 2 |

Measured CPU attention compute only; setup is separate. KV bytes are allocated array payload, not process RSS. All samples, cold calls, setup and whole-worker RSS remain in raw JSON. Throughput counts query vectors, not generated tokens. Extreme tail latency is not claimed.
