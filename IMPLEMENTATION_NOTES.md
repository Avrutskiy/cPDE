# Scope and verification

This document retains development-stage notes and test counts. For the final
0.2.0 CPU/CUDA build and validation status, see [BUILDING.md](BUILDING.md) and
[VALIDATION.md](VALIDATION.md). The CPU backend was added after the stages below.

The source is based on the supplied `cpde_fourier_experiment_bundle_v3.zip`, with
the previously agreed RPROP zero-gradient correction. The Fourier generator, runner,
README, capacity-test script, specialized error kernel, and experiment `main()` are
not part of this package. No exclusion schedule is built into the executable.

The original numerical derivative propagation kernels, sigmoid activation and matrix
multiplications are retained. Optimizer-kernel changes are the agreed RPROP
zero-gradient guard and the bound-rejection reset: an unapplied update resets
that weight's step to `0.1` and clears its gradient sign history, preventing a
false rollback on the next iteration. Accepted-step rollback is retained. Adam's
first and second moments are initialized correctly by the new optimizer-state layer.
No extra activation, optimizer, or mixed-memory API was added. Version 0.2.0 adds
scalar functions of integrals as cross-sample objectives and runtime compilation of scalar
cost kernels only; it does not recompile the retained numerical engine.

The retained engine is `cuda/cpde_engine.cuh`. The targeted changes to it are the
removal of the Fourier include and specialized cost method; directing diagnostics
to stderr so stdout can carry the binary protocol; avoiding leaked duplicate
contexts during repeated preparation; clearing context pointers when growing the
bank; including the output width in cache sizing; and releasing bias-view wrappers,
weight-pointer arrays, and the generator when a persistent network is closed.
These lifetime/capacity changes are needed by repeated evaluations and training
calls in a long-lived process. Legacy loading, printing, and mixed-memory routines
otherwise remain in place and are unused by the new interface.

`cuda/cpde_server.cu` owns the persistent GPU objects, context preparation, shared
bank capacity, sample slicing, optimizer state, command loop, and RAM snapshots.
`cuda/cost_kernels.cuh` provides matching and sparse polynomial residual/gradient
operations; supplied-vector contractions compile to linear polynomial terms.
`cpde/_prepare.py` performs one-time expression and derivative preparation. Python
sends fixed coefficients and numerical arrays, never per-epoch arithmetic commands.

The CUB update adds `cuda/reduction_kernels.cuh` and fuses block reductions into the
cost kernels. Signed double sums and invalid-value locations reduce on-device with
reused scratch; the old per-chunk `cublasSasum` host result is removed. Layer gradient
summaries provide L2 norm, maximum absolute value, non-finite count, and first bad
index through CPDE/2 and Python's `diagnostics_every`/`last_gradient_report` API.
Every iteration validates local costs, `dE/dY`, and weight gradients before updating.
This update makes no changes to `cpde_engine.cuh`, its bias-gradient `cublasSgemv`,
or `derivative_kernels_v4.cu`. Legacy loss allocations remain in the engine but are
unused by the new cost execution path.

## Verified in this workspace

Sixteen integration tests and one standalone reduction test pass with the host
probe. They exercise
matching weights across outputs/derivatives/samples and target-row mappings;
linear/quadratic equations and repeated-component contractions; the sum of all
three costs and its common weight gradient; full-data versus padded-chunk updates
for both optimizers; nonlinear-network weight gradients against finite differences;
GPU-object ownership semantics represented by independent probe allocations;
public derivative-row preservation and supplied non-coordinate derivatives; host
edits and data save/load; sparse/value-only closures; zero objective behavior;
optimizer continuation, full reset, moment reset, and optional gradient history;
periodic reporting/checkpoint events; output diagnostics across context sizes; and
stopping at an update boundary followed by resume. Added coverage checks reporting
independence of updates, gradient summaries against exported buffers, large sample
counts with partial chunks, overflow in objective/`dE/dY`/weight gradients, rejected
updates and session recovery, interval validation, and legacy capability rejection.
The standalone probe checks signed sums, double accumulation, scratch reuse,
gradient statistics and error indices, fused matching costs, and ignored padding.

The three-cost example also runs against that probe. Python files compile, and the
C++ command loop plus retained arithmetic compiles as host C++17 after test-only
launch substitution. The probe uses one simulated thread, scalar column-major BLAS,
and serial substitutes for CUB. It cannot validate real CUB compilation or block
collectives, GPU races, asynchronous copy errors, or differences in CUDA/cuBLAS
arithmetic. No GPU performance claim has been measured.

Remote Windows validation completed on 2 November 2026 using CUDA Toolkit 13.0.88,
Visual Studio 2022 / MSVC 19.44, and an RTX 3090 selected by GPU UUID. The Release
build includes SM 75 and SM 86 targets. All 17 tests pass against the real CUDA
executables, and the standalone reduction probe also passes through CTest. Python
is the installed Miniconda Python 3.13.11 with NumPy. No speedup benchmark or
Compute Sanitizer run has been performed.

The real build exposed the legacy derivative engine's global `E` macro colliding
with a CUB template parameter. The reduction header now saves, undefines, and
restores that macro around CUB includes. The standalone probe covers inclusion with
the macro present; the original engine and derivative files remain unchanged.

To repeat GPU validation, set
`CPDE_TEST_EXECUTABLE` to use a supplied real server and
`CPDE_TEST_REDUCTION_EXECUTABLE` for the standalone CUDA probe. CMake's
`CPDE_BUILD_REDUCTION_TESTS=ON` builds that probe and registers it with CTest.

The Python runtime owns no CUDA compiler dependency. The CMake file is solely for
the external provider of the executable. Ordinary training without checkpointing
returns metadata and scalars without downloading weights; evaluations stay resident
until explicitly downloaded or saved.

## Integral API in 0.2.0

The Python quadrature and expression modules prepare geometry and a restricted
instruction graph. The server generates CUDA source from those validated
instructions, compiles only the scalar kernel through NVRTC, and caches the loaded
module for its device. Numeric fields and weights remain runtime inputs. CUB
reduces generated densities; a precompiled GPU scalar-graph kernel supplies leaf
multipliers, then the retained backpropagation computes weight gradients in a
second pass. The old additive cost path is retained.

The new tests cover simplex measures and orientation, immutable/replaced geometry,
parser rejection, signed integrals, normal-direction fields, nonlinear functions
and network gradients, shared/nested variation graphs, freeze, empty integrals,
zero-weight singular samples and padding, non-mutating evaluation, optimizer
continuation across geometry changes, and numerical-error recovery. A moving
paraboloid drop footprint verifies the attached contact-line gradient against its
analytic derivative and finite differences after geometry reconstruction. This is
a variation check, not a drop energy-minimization or accuracy experiment.

The reviewed implementation passes all 32 tests on the RTX 3090 with the real
NVRTC/CUB path, including optional SciPy Delaunay tests. CTest's standalone reduction
probe also passes. The Mac host-probe run passes 31 tests and skips the optional
SciPy test because that dependency is absent in its runtime. The original engine
and derivative-kernel files compare byte-for-byte with the pre-CUB snapshot.
At that API-validation stage, no automatic geometry updater or point follower
had been implemented. Direct calls keep geometry fixed throughout their epochs.
The subsequent example implementation is described below; the earlier API tests
do not establish its physical accuracy.

## First 2D example attempt (2 November 2026; superseded)

The earlier introductory scripts have been replaced with the requested Fourier,
circular-Poisson and oval-drop examples; see `examples/README.md`. Earlier scope
statements above describe the API delivery before this replacement.

Fourier uses a 13x13 grid and the paper's 2,128,128,128,128,1 capacity, analytic
derivatives and explicit exclusion stages. Poisson uses unit RHS in a unit disk
with zero Dirichlet data. The drop uses one correction network over an exact oval
initial height, a volume penalty, an attached contact-line variation and an
example-level star-shaped geometry tracker between server updates. This does not
add automatic or general topology tracking to the core API.

Both expression parsers now accept indented multiline expressions. Parameter
underscores remain prohibited because they are reserved for derivative notation;
the README's invalid `contact_cos` expression parameter was corrected to
`contactcos`. New mathematical checks cover analytic targets, quadrature and
reconstructed-domain finite differences. Host validation of these examples is
separate from the prior GPU validation above; these new experiments have not yet
been validated for scientific accuracy on CUDA.

## Current two-example delivery (2 November 2026)

The current scripts have no cost wrappers, auxiliary data-generation functions,
small-network mode or post-training error-report pipeline. They show derivative
requests explicitly and use the public `table(requests, variables=...)` API to
include dependencies. Data arrays are generated with ordinary Python/NumPy.

The explicit fitting case requests x and y. The bulk fitting case lists only
xxx, xxy, xyy and yyy and matches the ten resulting rows of target arrays with
the existing matching kernel. Poisson explicitly prepares its interior and
boundary derivative sets and supplies T for the tangential boundary cost.
Every script uses one train call, then prints a net.eval result as a NumPy array.

The drop draft and its experiment-specific tests are retained under
`experiments/deferred_drop/` for reference, outside the launcher and source bundle.
The missing drop capability is input-set refresh during server-side training;
this delivery does not add another optimizer, solver loop, or network.

See `CODE_REVIEW.md` for the review scope, confirmed open findings and validation
limits. The earlier CUDA validation does not validate these new example runs.

## Conditional forward recomputation (2 November 2026)

Cost leaves keep independent input sets, derivative closures and chunk widths.
Compiled costs now prune extra Data rows to the dependencies referenced by their
expressions, retaining frozen components needed for values.

The server analyzes the scalar graph before execution to identify fixed leaf
multipliers. For each such leaf it runs forward then backward, accumulating the
weight gradient before proceeding to the next leaf. Leaves with value-dependent
multipliers initially run forward only. Once the graph has evaluated all scalar
costs and multipliers, those leaves repeat forward and immediately run backward.
This preserves shared context storage without retaining forward activations.
Weights remain fixed until all contributions have accumulated; one optimizer
update follows. Attached variations are included in this dependency analysis.

The user chooses grouping through net.cost calls. Multiline expressions combine
local terms on shared points; scalar graph arithmetic retains separate contexts.
The README documents pass counts and the tradeoff between sharing pointwise work
and keeping large point sets independent of costly derivative closures.

Poisson has two groups: the interior PDE and a combined boundary value/tangent
cost. Its interior uses random vertices with Delaunay triangles and centroid/area
integration, and its circular boundary uses uniformly spaced samples. NumPy plus
the geometry extra (SciPy) supplies these arrays, with no example helper functions.

Validation: all 46 host tests pass, including actual forward-call counts using
test-only instrumentation, mixed fixed/dynamic multipliers, variation gradients,
and derivative-pruning regressions. All three standalone scripts also complete
one test epoch at their declared network sizes. These checks do not validate CUDA execution,
NVRTC compilation, GPU performance, or numerical convergence.

## Third problem and energy update loop (2 November 2026)

The active Schrodinger example uses a real network in a deformed solid torus,
explicit second derivatives in the bulk and first derivatives on the wall.
Its boundary formula combines u**2 with |grad(u) cross N|**2, which is the sum of
both squared tangential derivatives for a unit normal. Grid weights and normals
are generated inline from a positive, near-1 deformation of reference torus samples.

The user explicitly requested a Python loop for this case. Each call trains 100
epochs at fixed E, then a quotient cost computes integral(-u*Delta(u))/integral(u**2)
and equation.set_parameters updates E. Adam moments and step count persist across
blocks. The same quadrature is used for the quotient and PDE residual. No original
CUDA engine or server changes were needed for this example.

The forward-retention proposal is saved as a future section of IMPLEMENTATION_PLAN.md.
Validation now comprises 48 passing host tests. The example harness executes the
new script for two shortened one-epoch blocks at its declared network/grid sizes;
normalization, PDE and wall values are reported by train. These tests validate
geometry, the quotient update and optimizer continuation, not eigenpair convergence
or CUDA execution. Drop minimization remains deferred.

## Reusing the last live forward (2 November 2026)

The server now runs the most expensive eligible value-dependent context last,
using a work estimate based on execution width, layer sizes and derivative rows
and orders. After the scalar graph computes multipliers, backward immediately
consumes that context's final live chunk without another forward. Other affected
chunks still recompute forward. Fixed-multiplier costs remain sequential
forward/backward pairs. No extra activation storage or original engine changes
are required; the shifted-memory retention proposal remains future work.

The estimate targets saved work per chunk, not total cost size. Without chunking,
one whole value-dependent cost avoids recomputation. With chunking, only its last
chunk does. The default torus example has one value-dependent normalization cost
and no chunking, so all its costs now need one forward per gradient evaluation.
Shared leaves, reporting order and a single accumulated-gradient optimizer update
are preserved. The tests check actual context traces, including reversed operand
order, values, gradients and continued optimization; GPU timing is unmeasured.
