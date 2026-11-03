# Validation for cPDE 0.2.0

Release checks completed on 3 November 2026. No prebuilt executable is shipped;
these checks validate sources that users compile locally.

After trimming the release to its eight documented examples, the retained suite
passed all 74 tests on macOS ARM64 with the production Accelerate CPU backend;
all eight example smoke runs also completed. Six tests specific to retired
experiments were archived with those experiments. No library or engine code
changed in this packaging cleanup.

The following earlier validation used the broader development snapshot (80 tests
and twelve smoke runs, including four experimental geometries):

| Check | Result | Environment / scope |
|---|---|---|
| Production CPU, Accelerate | 80 passed | Native macOS ARM64, Python 3.12, NumPy/SciPy; built with the macOS script |
| Production CPU, portable matrix routines | 80 passed | Native macOS ARM64, `CPDE_CPU_BLAS=NONE` |
| Production CPU, portable matrix routines | 79 passed, 1 skipped | Windows x64, MSVC 19.44; built with the Windows script |
| CUDA server | 79 passed, 1 skipped | Windows x64, CUDA 13.0.88, MSVC 19.44, RTX 3090, architecture 86, NVRTC enabled |
| Native engine/storage and reduction probes | All passed | Two probes per backend/build: macOS Accelerate CPU, macOS portable CPU, Windows CPU and Windows CUDA |
| Independent derivative/gradient oracle | All passed | macOS Accelerate CPU, Windows portable CPU and Windows CUDA; four derivative schemas plus 10,003-sample chunk-tail stress |
| Example smoke runs | 12 completed on CPU; 12 on CUDA | Eight delivered scripts plus four reference geometries; shortened training |
| Independent serial host-probe suite | 78 passed, 2 skipped | macOS; the two production-CPU integration checks skip when no CPU executable is supplied |

The single Windows skip is the host-only scheduling test, which needs `g++` for
forward-count instrumentation. It passed in both macOS production-CPU suite runs.
The four added backend tests cover selection/discovery, no fallback, handshake
mismatch, and an analytic expression value/gradient with training/state roundtrip.
The independent derivative oracle includes up-to-sixth-order and mixed derivative
schemas; its largest scaled forward/gradient error across the three tested backends
was below 1.1e-7. It is an arithmetic correctness check, not a speed benchmark.

After publication, [GitHub Actions CPU checks](https://github.com/Avrutskiy/cPDE/actions/runs/41581112923) passed on Linux
with both system CBLAS and portable routines, macOS Accelerate, and Windows
portable CPU. Each of the four jobs built its production CPU server, passed both
native probes, and passed the Python suite.

The [Linux host-probe and packaging checks](https://github.com/Avrutskiy/cPDE/actions/runs/41581113025) also passed for
Python 3.9 on Ubuntu 24.04 and Python 3.13 on Ubuntu 26.04: tests, all twelve
example smoke runs, Python wheel/sdist builds, and source/samples packaging.
These runs verified commit `18047c1` on 3 November 2026.

The original host workflow failed before executing tests because `ubuntu-latest`
had selected Ubuntu 26.04, which has no setup-python build for Python 3.9. Its
runner is now explicitly Ubuntu 24.04; Python 3.13 uses Ubuntu 26.04. Matrix jobs
finish independently so a failure does not cancel the other version's checks.
GPU concurrency and stream behavior still require the native CUDA path; CPU
checks cannot establish them.

## Recorded example results

The fresh explicit-fitting CUDA run completed the current 27 × 27 example for
1500 updates, with value relative L2 error 0.31621% on the validation grid. Circle
figures use saved 2000-update checkpoints; their residuals were reevaluated on a
regular grid of 22,589 points without retraining. Reevaluated function values
matched the saved fields exactly. Other full-run results and their provenance
are documented in the example guide and `examples/results/provenance.json`.
CPU smoke tests establish execution, not reproduction of those convergence curves.

## Fourier performance measurements

A separate RTX 3090 benchmark compares cPDE CUDA with PyTorch 2.11.0 and its
Inductor/Triton JIT for all mixed derivatives through orders 0–6. Numerical losses,
output rows and parameter gradients are compared at identical inputs and weights.
The README table times derivative evaluation, loss and parameter gradients;
optimizer updates and production safety checks are outside that table's scope.
Compilation is measured separately. The [benchmark methodology and records](https://github.com/Avrutskiy/cPDE/blob/main/docs/benchmarks/fourier-rtx3090.md)
include repeated batch timings and a supplemental public `net.train` measurement.
This benchmark does not establish training convergence or performance on other
networks, precisions or machines.

## Production CPU versus the test adapter

The production `cpde_cpu_server` compiles shared sources directly as C++17. The
adapters under `cpu/` provide host memory, system or portable BLAS, and serial
reductions. Scalar expressions use an interpreter. `Session(backend='cpu')`
selects this executable; CUDA is neither searched for nor linked in CPU-only builds.

Separately, `tests/build_host_probe.py` copies sources to a temporary directory,
removes CUDA launch syntax, and compiles with `g++` using independent serial test
shims. Its optional forward counters verify scheduling and accumulated gradients.
This adapter reports `host-probe` and cannot satisfy explicit CPU selection.
Neither adapter generation nor tests run when installing/using the Python package.
Tests are included in source ZIP/sdist, not the wheel or samples archive. No
compiled production or test executable is included in any asset.

## Reproduce

Install Python dependencies with `python -m pip install '.[geometry]'`.
For macOS CPU, build the server and both probes, then supply their paths:

```sh
sh scripts/build-macos.sh -DCPDE_BUILD_ENGINE_TESTS=ON -DCPDE_BUILD_REDUCTION_TESTS=ON
ctest --test-dir build-cpu -C Release --output-on-failure
export CPDE_TEST_EXECUTABLE="$PWD/build-cpu/cpde_cpu_server"
export CPDE_TEST_CPU_EXECUTABLE="$CPDE_TEST_EXECUTABLE"
export CPDE_TEST_REDUCTION_EXECUTABLE="$PWD/build-cpu/cpde_cpu_reduction_probe"
python -B -m unittest discover -s tests -v
python tests/run_examples.py "$CPDE_TEST_EXECUTABLE"
python scripts/validate_cuda_fixes.py "$CPDE_TEST_EXECUTABLE"
```

For Linux, replace the first command with `sh scripts/build-linux.sh cpu` and the
same CMake options. To exercise portable routines, also pass `-DCPDE_CPU_BLAS=NONE`
using a separate build directory. See [build instructions](BUILDING.md).

For CUDA, build with `CPDE_BACKEND=CUDA` (default) and both probe options, then set
`CPDE_TEST_EXECUTABLE` to `cpde_server` and `CPDE_TEST_REDUCTION_EXECUTABLE` to
`cpde_reduction_probe`. Also provide a CPU executable to run CPU-specific tests;
otherwise those two integration tests skip. The Windows script accepts `both` to
build both backends. Visual Studio outputs use `.exe` under `Release/`; Ninja
outputs are directly in the build directory. Set variables with PowerShell or
`cmd` on Windows.

Without supplied executables, `python -B -m unittest discover -s tests -v` builds
the independent host probes and requires `g++`; CPU integration tests skip unless
`CPDE_TEST_CPU_EXECUTABLE` is set.

## Release packaging checks

The upload kit contains source and samples ZIPs, a pure-Python wheel, a Python
sdist, release notes, upload instructions and SHA-256 checksums. Packaging checks
verify ZIP integrity, manifests, local guide links, Python excerpts and figure
provenance, including archived documentation. An isolated wheel install is checked
against the production CPU server. No local output directory, checkpoint, cache,
compiled executable or CUDA runtime is distributed.

See [known issues](KNOWN_ISSUES.md) for retained numerical limitations.
