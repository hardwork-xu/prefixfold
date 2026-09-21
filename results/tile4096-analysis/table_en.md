| Workload | Variant | Median ms | SDPA / variant | KV MiB | KV reduction | Max abs error | n |
|---|---|---:|---:|---:|---:|---:|---:|
| shared_b16_p4096 | numpy_dense | 6.4156 | 1.066× | 132.000 | 0.00% | 4.56e-08 | 30 |
| shared_b16_p4096 | torch_sdpa | 6.8422 | 1.000× | 132.000 | 0.00% | 3.73e-08 | 30 |
| shared_b16_p4096 | prefixfold | 1.9347 | 3.537× | 12.000 | 90.91% | 2.11e-07 | 30 |
| shared_b16_p4096 | ungrouped | 3.9919 | 1.714× | 12.000 | 90.91% | 4.23e-08 | 30 |

Measured CPU attention compute only; setup is separate. KV bytes are allocated array payload, not process RSS. All samples, cold calls, setup and whole-worker RSS remain in raw JSON. Throughput counts query vectors, not generated tokens. Extreme tail latency is not claimed.
