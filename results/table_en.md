| Workload | Variant | Median ms | SDPA / variant | KV MiB | KV reduction | Max abs error | n |
|---|---|---:|---:|---:|---:|---:|---:|
| no_prefix | numpy_dense | 0.1814 | 1.216× | 4.000 | 0.00% | 2.30e-07 | 30 |
| no_prefix | torch_sdpa | 0.2206 | 1.000× | 4.000 | 0.00% | 2.13e-07 | 30 |
| no_prefix | prefixfold | 0.3561 | 0.620× | 4.000 | 0.00% | 2.09e-07 | 30 |
| no_prefix | ungrouped | 0.3852 | 0.573× | 4.000 | 0.00% | 2.09e-07 | 30 |
| single_request | torch_sdpa | 0.5598 | 1.000× | 8.250 | 0.00% | 3.14e-08 | 30 |
| single_request | ungrouped | 0.9013 | 0.621× | 8.250 | 0.00% | 2.84e-08 | 30 |
| single_request | prefixfold | 0.9730 | 0.575× | 8.250 | 0.00% | 2.84e-08 | 30 |
| single_request | numpy_dense | 0.4833 | 1.158× | 8.250 | 0.00% | 2.54e-08 | 30 |
| short_prefix | torch_sdpa | 0.4391 | 1.000× | 8.000 | 0.00% | 2.77e-07 | 30 |
| short_prefix | prefixfold | 0.5213 | 0.842× | 4.250 | 46.88% | 1.82e-07 | 30 |
| short_prefix | ungrouped | 1.2910 | 0.340× | 4.250 | 46.88% | 1.64e-07 | 30 |
| short_prefix | numpy_dense | 0.3959 | 1.109× | 8.000 | 0.00% | 2.70e-07 | 30 |
| shared_b4_p4096 | ungrouped | 2.1107 | 0.856× | 9.000 | 72.73% | 3.52e-08 | 30 |
| shared_b4_p4096 | torch_sdpa | 1.8058 | 1.000× | 33.000 | 0.00% | 3.71e-08 | 30 |
| shared_b4_p4096 | prefixfold | 1.0796 | 1.673× | 9.000 | 72.73% | 6.77e-08 | 30 |
| shared_b4_p4096 | numpy_dense | 1.6257 | 1.111× | 33.000 | 0.00% | 2.84e-08 | 30 |
| shared_b16_p4096 | numpy_dense | 6.4978 | 1.085× | 132.000 | 0.00% | 4.14e-08 | 30 |
| shared_b16_p4096 | torch_sdpa | 7.0512 | 1.000× | 132.000 | 0.00% | 4.16e-08 | 30 |
| shared_b16_p4096 | ungrouped | 6.9511 | 1.014× | 12.000 | 90.91% | 3.79e-08 | 30 |
| shared_b16_p4096 | prefixfold | 2.3603 | 2.987× | 12.000 | 90.91% | 1.01e-07 | 30 |
| shared_b32_p8192 | prefixfold | 5.0535 | 5.893× | 24.000 | 95.38% | 5.66e-08 | 30 |
| shared_b32_p8192 | ungrouped | 27.9303 | 1.066× | 24.000 | 95.38% | 2.42e-08 | 30 |
| shared_b32_p8192 | torch_sdpa | 29.7789 | 1.000× | 520.000 | 0.00% | 2.52e-08 | 30 |
| shared_b32_p8192 | numpy_dense | 26.6417 | 1.118× | 520.000 | 0.00% | 3.14e-08 | 30 |
| ragged_suffix | prefixfold | 2.6143 | 2.850× | 16.000 | 88.21% | 1.13e-07 | 30 |
| ragged_suffix | torch_sdpa | 7.4516 | 1.000× | 135.750 | 0.00% | 3.79e-08 | 30 |
| ragged_suffix | numpy_dense | 6.8509 | 1.088× | 135.750 | 0.00% | 3.92e-08 | 30 |
| ragged_suffix | ungrouped | 7.7640 | 0.960× | 16.000 | 88.21% | 4.34e-08 | 30 |

Measured CPU attention compute only; setup is separate. KV bytes are allocated array payload, not process RSS. All samples, cold calls, setup and whole-worker RSS remain in raw JSON. Throughput counts query vectors, not generated tokens. Extreme tail latency is not claimed.
