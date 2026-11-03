# Fourier derivative-training benchmark

This benchmark measures the Fourier example's derivative evaluation, normalized
squared-error loss, and gradients with respect to every network weight and bias.
The main table in the repository README excludes optimizer updates, data transfers
and production training diagnostics. It is a computation benchmark at fixed weights,
not a comparison of convergence rates or final approximation errors.

## Measured results

Hardware: NVIDIA GeForce RTX 3090 (24576 MiB); Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz.
NVIDIA driver 610.74; Windows; Python 3.11.15;
PyTorch 2.11.0+cu128; Triton Windows 3.6.0.post26.

Clean production server compilation/linking: **19.3 s**;
including CMake configuration: **26.5 s**.
The compiler used CUDA 13.0.88, MSVC 19.44 and the sm_86 Release target.

| Order | PyTorch, no JIT (ms/epoch) | PyTorch + JIT (ms/epoch) | cPDE CUDA (ms/epoch) | JIT compilation time (s) |
|---:|---:|---:|---:|---:|
| 0 | 1.45 | 0.84 | 0.41 | 8.8 |
| 1 | 2.73 | 1.50 | 0.68 | 11.8 |
| 2 | 8.06 | 3.09 | 1.10 | 18.5 |
| 3 | 26.71 | 7.51 | 1.72 | 44.3 |
| 4 | 90.09 | 25.74 | 2.40 | 175.9 |
| 5 | 317.55 | 47.49 | 3.59 | 509.9 |
| 6 | 1,092.37 | 161.08 | 5.65 | 2,649.7 |

[CSV table](fourier-rtx3090.csv) · [Raw batch times, errors and source hashes](fourier-rtx3090.json) ·
[Portable benchmark script](../../scripts/benchmark_fourier_performance.py) ·
[Timing-only server patch](fourier-timing.patch).

The largest relative L2 parameter-gradient difference between cPDE and eager
PyTorch was 1.33e-07.

## Public training API

These supplemental timings use the unmodified production `net.train` with RPROP,
finite-gradient checks, per-epoch synchronization and final evaluation. They
include optimizer work and therefore have a different scope from the main table.

| Order | `net.train` (ms/epoch) |
|---:|---:|
| 0 | 0.85 |
| 1 | 0.98 |
| 2 | 1.41 |
| 3 | 2.05 |
| 4 | 2.79 |
| 5 | 3.96 |
| 6 | 5.98 |

## Workload

- A 27 × 27 grid (729 points) in `[-1, 1]²`.
- Architecture `2 → 128 → 128 → 128 → 128 → 1`, with sigmoid hidden layers and
  a linear output.
- All distinct mixed derivatives of total order zero through the selected maximum
  order. Orders 0–6 therefore use 1, 3, 6, 10, 15, 21 and 28 derivative rows.
- The Fourier example's 10 × 10 coefficient grid, coefficient seed 2026.
  Each squared-error row is weighted by the inverse population variance of its
  target values. The loss sums over all rows and samples.
- Initialization seed 3407. cPDE's exact initial weights, points, targets and
  cost weights are saved and reused by both PyTorch implementations.
- Float32 network arithmetic; PyTorch TF32 is disabled.

## What each column means

**PyTorch, no JIT** uses standard separate weight and bias tensors,
`torch.nn.functional.linear`, sigmoid, and nested `torch.autograd.grad` calls.
Lower-order derivative graphs are reused. A final reverse pass differentiates the
complete loss with respect to all parameters.

**PyTorch + JIT** captures that same complete derivative-and-parameter-gradient
computation with `make_fx`, then compiles it with
`torch.compile(..., backend='inductor', fullgraph=True)`. The default Inductor mode
is used; CUDA Graph replay is not enabled. It compiles the entire computation,
not just the original MLP. Neither PyTorch implementation uses hand-written
activation derivative formulas.

**cPDE CUDA** uses the release's existing matching-cost forward/backward path,
with a separate timing-only command to repeat the computation inside the server.
The numerical engine and kernels are unchanged. This avoids counting Python IPC,
gradient downloads or optimizer-state snapshots in the computation timing.
The timing command is provided as a patch for a separate benchmark build; it is
not part of the production server protocol.

**JIT compilation time** measures graph capture, Inductor/Triton compilation and
the first execution after CUDA initialization and data upload. Each order runs in
a fresh process with empty Inductor and Triton cache directories. Subsequent epochs
reuse the compiled graph. This timing excludes Python imports and package installation.

The cPDE build time reported before the README table is a clean, unmodified
production CUDA server build with NVRTC enabled, Release configuration and
`CMAKE_CUDA_ARCHITECTURES=86`. Compilation/linking and CMake configuration are
recorded separately. The server is built once for all derivative orders; this
array-matching objective needs no per-order runtime compilation. Scalar expression
objectives use a separate NVRTC path and are outside this benchmark.

## Timing and validation

The primary numbers are medians of seven synchronized wall-clock batches. cPDE
uses 20 warm-up computations and 100 computations per measured batch. PyTorch
performs untimed validation and warm-up, then chooses 1–100 computations per batch
to target roughly 250 ms. CUDA event intervals are also recorded, but are not used
as the main table values; they include gaps between GPU launches, rather than only
the sum of kernel durations.

The loss, output derivative rows and parameter gradients are checked at identical
weights before accepting timings. The JIT results are also checked against
uncompiled PyTorch. Individual batch times and numerical errors are retained in
the accompanying JSON records.

Actual `net.train` calls additionally perform finite-gradient checks, per-epoch
synchronization, optimizer updates, reporting and a final objective evaluation.
Supplemental public-API timings include those costs, use RPROP with initial step
`2e-4`, and amortize each call over 100 epochs. The original network state is
restored outside each measured batch. These supplemental numbers have a different
scope from the main computation-only comparison.

## Reproduce the computation benchmark

Use a separate source checkout for the timing patch:

```sh
git apply docs/benchmarks/fourier-timing.patch
cmake -S . -B build-benchmark -DCPDE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCPDE_ENABLE_NVRTC=OFF
cmake --build build-benchmark --config Release --parallel 2
```

Install NumPy and a CUDA-enabled PyTorch build. For Windows, use a compatible
[Triton Windows version](https://github.com/triton-lang/triton-windows). The measured
environment used PyTorch `2.11.0+cu128` and Triton Windows `3.6.0.post26`.
Select the same GPU for both libraries with `CUDA_VISIBLE_DEVICES` before launching
Python. On the measured machine, GPU UUID selection was used because CUDA and
`nvidia-smi` enumerated the two cards in different orders.

For each order, first save cPDE's data and weights, then run the two PyTorch modes:

```sh
python scripts/benchmark_fourier_performance.py cpde 3 --server /path/to/timing-patched/cpde_server
python scripts/benchmark_fourier_performance.py eager 3
python scripts/benchmark_fourier_performance.py compiled 3
```

Replace `3` with each integer from `0` through `6`. Use the same `--output` directory
for all three modes; the default is `output/fourier-performance`. On Windows the
server filename ends in `.exe`, and Visual Studio generators place it under
`Release/`. Run modes sequentially to avoid GPU contention. High-order JIT
compilation can take substantially longer than the actual training computations.

## Scope

These measurements apply to this network, batch size, derivative strategy,
precision and Windows machine. The host CPU affects both JIT compilation and
kernel-launch overhead. They do not establish the fastest possible implementation
in PyTorch, performance on other GPUs, or a convergence advantage.

See the [PyTorch compiler documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html)
for compilation and graph reuse behavior.
