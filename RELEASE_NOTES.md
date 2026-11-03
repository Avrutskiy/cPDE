# cPDE 0.2.0

**cPDE: CUDA-Efficient High-Order Derivative Training**

*An easy-to-use Python interface for forward and backward propagation with derivatives through sixth order, including mixed derivatives. CUDA and CPU backends.*

Currently, **only fully connected multilayer perceptrons (MLPs)** are supported.

Forward mode computes network values and their requested input
derivatives; backward mode computes weight gradients for objectives involving
those quantities. This supports both function approximation and PDE residual
minimization, with CUDA and CPU execution.

The derivative-training method is described in V. I. Avrutskiy,
[**Enhancing Function Approximation Abilities of Neural Networks by Training
Derivatives**](https://doi.org/10.1109/TNNLS.2020.2979706),
*IEEE Transactions on Neural Networks and Learning Systems*.
An [open-access preprint](https://arxiv.org/abs/1712.04473) is also available.

Python describes data, derivatives, quadrature and costs; the selected persistent
server runs complete training calls.

## AI-use disclosure

This release combines a supplied numerical core with substantial OpenAI Codex
assistance ("vibecoding"). The sigmoid/high-order derivative and backpropagation
formulas came from the supplied implementation; the retained engine also received
targeted AI-assisted fixes. Much of the Python wrapper, persistent server,
cost/expression infrastructure, CPU backend, examples, tests, build tooling and
documentation was implemented or revised with Codex under the owner's direction.
`CODE_PROVENANCE.md` provides the detailed breakdown and the limits of this
attribution. Tests also received AI assistance and are not an independent human
audit.

## Included

- Explicit CPU/CUDA session selection. The CPU backend runs natively on Apple
  Silicon with Accelerate, or uses system CBLAS/portable serial routines elsewhere.
  macOS, Linux and Windows compile scripts are included.
- Array matching and scalar expression costs, quadrature integration, composed
  scalar objectives, frozen components and attached first variations. CUDA uses
  NVRTC expression kernels; CPU interprets the same validated instructions.
- Spatial derivatives through order six, RPROP/Adam state, checkpoints,
  stop/resume, gradient diagnostics and periodic RPROP step restoration.
- Accepted wrapper/CUDA audit fixes, including Python-generated initial weights,
  matrix ownership, shared metadata, forward-only allocation, gradient-history
  resets and corrected division adjoints.
- Eight examples with detailed, linked walkthroughs of Fourier fitting, disk Poisson, circle
  Schrödinger and the moving-ellipse variation. Vanilla scripts contain only
  the derivatives needed by their formulations.
- Rendered example results, eigenvalue-elimination equations, and numerical
  energy/gradient comparisons for the attached variation.

## Files

| Asset | Contents |
|---|---|
| `cpde-0.2.0-source.zip` | Repository source, CPU/CUDA build scripts, Python package, tests, documentation and examples. Extract this archive to populate the GitHub repository. |
| `cpde_interface-0.2.0-py3-none-any.whl` | Installable Python wrapper. No server executable is included. |
| `cpde_interface-0.2.0.tar.gz` | Python source distribution, including CPU/CUDA sources, build scripts and example documentation. |
| `cpde-0.2.0-samples.zip` | Standalone examples, detailed guide, figures and reference; install the library and build a server separately. |
| `SHA256SUMS` | SHA-256 checksums for the release assets. |

## Build and run

Build locally with CMake and a C++17 compiler. CUDA additionally needs the local
CUDA Toolkit, cuBLAS and NVRTC; CPU builds have no CUDA dependency.

| Platform | CPU | CUDA |
|---|---|---|
| macOS, including Apple Silicon | `sh scripts/build-macos.sh` | — |
| Linux | `sh scripts/build-linux.sh cpu` | `sh scripts/build-linux.sh cuda` |
| Windows developer PowerShell | `./scripts/build-windows.ps1 -Backend cpu` | `./scripts/build-windows.ps1 -Backend cuda` |

Install the wrapper with `python -m pip install '.[geometry]'` from the source
checkout, or install the wheel. Pass the built server path explicitly:

```python
from cpde import Session

with Session('/path/to/cpde_cpu_server', backend='cpu') as session:
    print(session.backend)
```

`Session(backend='cpu')` discovers `CPDE_CPU_SERVER` or `cpde_cpu_server` on PATH.
CUDA uses `backend='cuda'`, `CPDE_SERVER` and `cpde_server`. Explicit selections
are verified; there is no silent fallback. See `BUILDING.md` for output paths,
BLAS options and using existing examples on either backend.
Use the matching 0.2.0 wrapper and server together. The updated server requires
wrapper-supplied initial weights; the wrapper also supports older servers for
their supported operations. No prebuilt server or CUDA runtime is distributed.

## Validation and limitations

The release contains eight documented examples; older experimental variants and
their dedicated tests are excluded. After cleanup, all 74 retained tests and all
eight example smoke runs passed on Apple Silicon with Accelerate.

Before that packaging cleanup, both the Accelerate and portable CPU builds passed
the broader 80-test development suite on Apple Silicon.
On Windows, CPU and CUDA each passed 79 tests, with one host instrumentation test
skipped because `g++` was unavailable; that test passed on macOS. Native probes,
independent high-order derivative/gradient checks and example smoke runs also
passed. [GitHub Actions CPU checks](https://github.com/Avrutskiy/cPDE/actions/runs/41581112923) subsequently passed on Linux
(system BLAS and portable), macOS and Windows. [Linux host tests, example smoke
runs and packaging](https://github.com/Avrutskiy/cPDE/actions/runs/41581113025) passed with Python 3.9 on Ubuntu 24.04 and
Python 3.13 on Ubuntu 26.04. See `VALIDATION.md` for scope and reproduction.

See `KNOWN_ISSUES.md` for retained numerical limitations: Adam second-moment
overflow, sigmoid-tail derivatives, tiny-gradient RPROP comparisons, and
cancellation in the circle cost `A-B**2/C`. Geometry remains fixed during each
training call. CPU derivative propagation, reductions and expression interpretation
are serial; matrix operations may use multithreaded system BLAS. There is no Metal
backend; no comparative CPU/GPU speed claim is made.

The README includes a measured RTX 3090 Fourier comparison with PyTorch and
PyTorch JIT, covering derivative orders 0–6 and separate compilation times.
See the [benchmark methodology](https://github.com/Avrutskiy/cPDE/blob/main/docs/benchmarks/fourier-rtx3090.md)
for the workload, timing scope and reproducibility.

Included example figures describe recorded CUDA runs, not convergence guarantees.
The production CPU backend is separate from the test-only host instrumentation.
