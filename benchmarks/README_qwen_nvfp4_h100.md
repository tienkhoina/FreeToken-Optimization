# Qwen NVFP4 microbenchmark on H100

Measured 2026-10-09 on an NVIDIA H100 80GB HBM3 with CUDA 12.9 and an
Intel Xeon Platinum 8462Y+. The process was allowed 16 CPUs across two NUMA
nodes. A vLLM process occupied 76,674 MiB of GPU memory during the run; GPU
utilization was 0% in snapshots immediately before and after the measurements.
These are results from this shared host, not isolated-device peak claims.

The Qwen expert operator benchmark uses synthetic NVFP4 banks with
Qwen3.8-Flash-Next dimensions. It does not load checkpoint weights or measure
whole-model inference. CPU and GPU GEMM plus host/device copy measurements use
separate microbenchmarks.

## GEMM, RAM and copy rates

CPU and GPU GEMM use the same 6144 x 6144 shape. Rates use
`2*N^3 / median_time`; CPU uses 16 PyTorch threads and five timed repetitions.
GPU uses CUDA events after warmup and five samples, each containing five
matmuls.

| Measurement | CPU | GPU |
|---|---:|---:|
| GEMM FP32, 6144 x 6144 | 1.812 TFLOP/s | 51.06 TFLOP/s |
| GEMM BF16, 6144 x 6144 | 1.274 TFLOP/s | 800.54 TFLOP/s |
| STREAM-style RAM read, 2 GiB buffer, 8 passes | 181.0 GB/s | n/a |
| Pinned RAM -> VRAM, 128 MiB per copy | n/a | 55.32 GB/s |
| VRAM -> pinned RAM, 128 MiB per copy | n/a | 55.31 GB/s |

The copy test performs 20 asynchronous copies per sample and reports the median
of five samples. It measures pinned-memory transfer throughput, not pageable-
memory staging. The CPU RAM read is a bandwidth-oriented streaming reduction,
not model inference. These GEMM and copy results are short microbenchmarks under
the host's background load.

The GEMM rows above are general FP32/BF16 compute baselines. They are not NVFP4
expert throughput measurements. The NVFP4-specific GPU measurements are below.

## NVFP4 expert operator

Ran `benchmarks/bench_nvfp4_route_merge.py` with H=2560, I=640, top-k=10 and
16 synthetic expert rows. CUDA graph compilation and capture are outside the
timed region. The split case routes half of the selected experts through each
branch; weight transfer is not included.

| Query tokens | Unsplit reference | Hit/fetch, ordered reduction | Hit/fetch, separate partial sums | Effective expert GEMM rate, unsplit |
|---:|---:|---:|---:|---:|
| 1 | 37.536 us | 50.304 us | 52.224 us | 2.62 TFLOP/s |
| 39 | 527.232 us | 695.680 us | 710.624 us | 7.27 TFLOP/s |
| 64 | 740.128 us | 769.824 us | 777.792 us | 8.50 TFLOP/s |

The effective expert rate estimates `6 * H * I * top_k` operations per token
for the two gate/up projections and down projection. It is a derived operator
rate, not a hardware counter measurement. The ordered reduction matches the
unsplit output byte-for-byte for all tested shapes. Separate partial sums
differ, with maximum absolute error 0.001953125. Split-path times include
their branch and reduction work and are slower than the unsplit reference in
this test.

### NVFP4 decode copy and CPU/GPU timing

For H=2560 and I=640, the packed weights, block scales and global scales occupy
2,772,480 bytes (2.644 MiB) per expert in the FreeToken bank layout. The pinned
copy bandwidth above implies about 0.050 ms per expert and 0.501 ms for ten
experts, before gather and launch overhead.

The schedule calibration measured the following one-token synthetic decode
cases with 16 CPU workers. CPU graph time includes the host/device handoff and
is separate from the CPU task time. GPU copy and math are timed separately.

| Missing experts | CPU task | CPU branch graph | GPU copy graph | GPU NVFP4 math |
|---:|---:|---:|---:|---:|
| 1 | 1.095 ms | 3.000 ms | 0.064 ms | 0.0207 ms |
| 10 | 2.221 ms | 3.773 ms | 0.560 ms | 0.0394 ms |

These CPU times are diagnostic only: CPU and GPU outputs from this synthetic
calibration had relative L2 error about 0.99999. Do not interpret them as
validated CPU NVFP4 FLOP/s or use them to choose a CPU/GPU miss split. A
checkpoint-backed CPU comparison with verified output parity is still needed.
The pinned-copy throughput and GPU NVFP4 operator timings are independently
measured; neither depends on accepting the CPU output.

Reproduce the NVFP4 operator run from the repository root:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python benchmarks/bench_nvfp4_route_merge.py \
  --output results/qwen_nvfp4_route_merge.json --repeats 21
```

The CPU/GPU MoE schedule calibration on synthetic banks reported relative L2
output error about 0.99999, so its suggested miss split is not validated and
must not be used as a CPU/GPU schedule. The measurements above establish that
the GPU NVFP4 operator runs on the H100; they do not establish CPU/GPU output
parity, checkpoint-level accuracy, or end-to-end serving speed.

The checkpoint downloaded for this investigation was
`RedHatAI/Qwen3.8-Flash-Next-NVFP4`, revision
`c8f2fb1b9869f686b214782036123b10ff96d14a`. The benchmark uses synthetic
banks, not those checkpoint weights. The server was not launched because the
existing GPU process left only 4,397 MiB free.

Raw tensors and timing samples from the local run are in the ignored
`results/qwen_redhat_trial_2026-10-09/` directory; they are not part of this
tracked report.
