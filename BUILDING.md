# Build CUDA or CPU execution

Python always connects to a separately built executable. It never compiles a
server or substitutes a different backend when the requested one is unavailable.
Both executables use the same network, derivative and optimizer implementation.

## Platform scripts

Run from the source checkout. CMake and a C++17 compiler are required. CUDA
builds additionally require locally installed CUDA, cuBLAS and NVRTC.

| Platform | CPU | CUDA |
|---|---|---|
| macOS, Apple Silicon or Intel | `sh scripts/build-macos.sh` | Not supported by this macOS build |
| Linux | `sh scripts/build-linux.sh cpu` | `sh scripts/build-linux.sh cuda` |
| Windows | `./scripts/build-windows.ps1 -Backend cpu` | `./scripts/build-windows.ps1 -Backend cuda` |

Linux and Windows also accept `both`. On Windows use an x64 Visual Studio
developer PowerShell with CMake on PATH. On macOS install the command-line
compiler tools. Builds target the host architecture by default: an Apple Silicon
Mac produces a native ARM64 executable.

The scripts create `build-cpu`, `build-cuda` or `build-both`. Executables are named
`cpde_cpu_server` and `cpde_server` (with `.exe` on Windows). Visual Studio builds
place them in the build directory's `Release/` subdirectory; Ninja/Make builds
place them directly in the build directory.

Extra CMake options can be passed after the Unix script's backend argument:

```sh
sh scripts/build-linux.sh cuda -DCMAKE_CUDA_ARCHITECTURES=86
sh scripts/build-macos.sh -DCPDE_CPU_BLAS=SYSTEM
```

For Windows, use `-CMakeArgs @('-DCMAKE_CUDA_ARCHITECTURES=86')` or the desired
list of options. Unix scripts honor `CPDE_BUILD_DIR` and `CPDE_BUILD_JOBS`;
Windows accepts `-BuildDirectory` and `-Jobs`.

## Direct CMake commands

CPU only, without searching for or linking CUDA:

```sh
cmake -S . -B build-cpu -DCPDE_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release
```

Use `-DCPDE_BACKEND=CUDA` for CUDA or `BOTH` to build both. The default remains
CUDA for compatibility. `cmake --install build-cpu --prefix /your/prefix` installs
the CPU executable under `bin`; add that directory to PATH if desired.

## CPU matrix operations

`CPDE_CPU_BLAS=AUTO` is the default. On macOS it uses Apple's Accelerate framework.
On Linux/Windows it uses a discoverable system BLAS library with a working CBLAS
header/interface. If none is found, it uses the bundled serial matrix routines.
The configure log reports which implementation was selected.

- `-DCPDE_CPU_BLAS=SYSTEM` requires system CBLAS/Accelerate and fails configuration
  if it cannot be found.
- `-DCPDE_CPU_BLAS=NONE` selects the portable serial implementation deliberately.

System BLAS may run matrix operations on multiple CPU threads. Derivative
propagation, scalar reductions and the expression interpreter are serial in this
version. The CPU option is functional, but no CUDA-equivalent speed is promised.
There is no Metal backend or Apple GPU execution.

## Select a session

Explicit executable paths work on every platform:

```python
from cpde import Session

with Session('/path/to/cpde_cpu_server', backend='cpu') as session:
    assert session.backend == 'cpu'
```

For the shorter form, set `CPDE_CPU_SERVER` to the CPU executable path or put
`cpde_cpu_server` on PATH:

```python
with Session(backend='cpu') as session:
    print(session.backend)  # cpu
```

`Session(backend='cuda')` similarly uses `CPDE_SERVER` or `cpde_server` on PATH.
`Session('/path/to/server')` retains the existing API and detects the backend from
the executable handshake. An explicit backend must match the executable;
a mismatch raises an error. CPU accepts only `device=0`; it is not a CPU-core
selector. `Session()` without arguments selects CUDA discovery.

To run an existing example on CPU, set its `CPDE_SERVER` environment variable
to `cpde_cpu_server`; the examples pass that path explicitly and detect CPU.
For new CPU applications, the dedicated `CPDE_CPU_SERVER` variable keeps the two
backend paths separate.

## CPU execution and the test adapter

The production CPU target compiles shared sources directly as standard C++.
A launch macro selects serial grid-stride execution; no regular-expression source
rewriting is involved. CPU memory, BLAS and reduction adapters are under `cpu/`.
The same validated scalar expression instructions execute in a CPU interpreter;
there is no NVRTC dependency. `compile()` prepares the expression on CPU, and
`inspect()['compile_log']` identifies CPU interpretation. The displayed generated
CUDA source remains a reference for the same expression, not code run on CPU.

The test-only host probe is still separate and reports `session.backend ==
'host-probe'`. It uses independent serial shims and optional forward counters.
It cannot satisfy an explicit `backend='cpu'` request. Production CPU execution
and the test adapter share arithmetic but have different purposes and build paths.

## Verification

Enable `CPDE_BUILD_ENGINE_TESTS` and `CPDE_BUILD_REDUCTION_TESTS` for native
probes. CPU targets are `cpde_cpu_engine_storage_probe` and
`cpde_cpu_reduction_probe`; CUDA target names retain their existing spelling.
Run `ctest --test-dir build-cpu -C Release --output-on-failure`.

For the Python suite, set `CPDE_TEST_EXECUTABLE` to the server under test and
`CPDE_TEST_REDUCTION_EXECUTABLE` to its corresponding reduction probe. Also set
`CPDE_TEST_CPU_EXECUTABLE` to a production CPU server to enable the CPU selection
and integration checks. See [validation](VALIDATION.md) for measured coverage.
