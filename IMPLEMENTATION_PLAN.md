# Quadrature and composable costs implementation plan

Status, 2 November 2026: the fixed-quadrature integral API below is implemented in
0.2.0, including NVRTC expressions, scalar graphs, freeze/variation, read-only
gradient evaluation, and explicit geometry/parameter replacement. The retained
Data-based API remains supported. See README.md for the current executable contract
and IMPLEMENTATION_NOTES.md for validation. Automatic geometry updates, contour
following, persistent compiler caches, and the final physical examples remain
separate work. This document retains the original detailed design and future
extension boundaries; it is not a claim that every proposed extension is complete.

The extension keeps the network engine and optimizers, makes Quadrature the default
input to costs, compiles general scalar densities, and supports scalar functions of
costs. A cost can attach another cost as its variation. The attachment adds a gradient
without adding its value. The keyword for suppressing selected local derivatives is
`freeze`.

## Agreed scope

- Users create points in Python. There are no built-in shape or point generators.
- Provide basic Delaunay triangulation in the plane and tetrahedralization in 3D.
  Users supply connectivity for curves, embedded surfaces, nonconvex regions, holes,
  and other specialized domains.
- Do not provide constrained triangulation, surface reconstruction, adaptive mesh
  refinement, or automatic extraction of level sets in this implementation.
- Accept user-supplied connectivity or complete quadrature points and weights.
- There is no required public Mesh object and no required Data object for ordinary
  coordinate-input costs.
- Use `variables` consistently. Use `T.x` and `N.x` in expressions for tangent and
  normal components.
- Keep existing Data-based calls and their numerical meanings. Different costs may
  use different point sets but must belong to the same network and session.
- Preserve the existing derivative propagation limit of order 6, sigmoid/linear
  network layers, RPROP, Adam, chunking, and persistent executable model.
- Changing geometry is supplied by the user; an explicit refresh mechanism makes
  it usable during training. `variation` does not itself move any points.

Confirmed follow-up decisions: compile only expression kernels with server-side
NVRTC, retain centroid as a simple default, and treat frequent geometry updates as
a separate extension preferably executed on GPU. Do not make a per-update Python
callback part of the initial implementation. The proposed GPU update contract below
is a design boundary for that extension, not a commitment to a geometry algorithm.

## Public API

### Quadrature construction

```python
# Basic automatic triangulation of a convex planar region.
q = Quadrature(
    points=points,
    variables="xy",
    dimension=2,
    triangulation="delaunay",
    rule="centroid",
)

# User-supplied triangles embedded in 3D.
surface = Quadrature(
    points=points,
    cells=triangles,
    variables="xyz",
    dimension=2,
    rule="centroid",
)

# Fully supplied quadrature, without any meshing.
boundary = Quadrature(
    points=evaluation_points,
    weights=weights,
    variables="xy",
    dimension=1,
    N=normals,
    T=tangents,
)
```

Exactly one construction mode is selected: triangulation, supplied cells, or supplied
weights. Reject conflicting modes instead of inferring precedence. Require `rule`
to be omitted in the supplied-weights mode. Default to `centroid` in mesh modes.

### Costs and expressions

```python
energy = net.cost(
    quadrature=q,
    expression="0.5*nu*(u_x**2 + u_y**2) + 0.25*(u**2 - 1)**2",
    parameters={"nu": 0.01},
    name="energy",
)
mass = net.cost(quadrature=q, expression="u", name="mass")
area = net.cost(quadrature=q, expression="1", name="area")
objective = energy + 100.0*(mass/area - 0.3)**2
net.train(cost=objective, epochs=1000, optimizer="adam")
```

Retain the existing `targets`, `equations`, and `vector_a` forms as alternatives to
`expression`. Exactly one form is selected. Existing forms retain their implicit
squared-residual semantics; `expression` is integrated directly, without a square.
All forms can use quadrature. Existing `weights` continue to mean density weights,
not geometric quadrature weights. For expression costs, put density coefficients
explicitly in the expression; reject redundant `weights` or `rhs` arguments.

Preserve `net.cost(inputs=data, ...)` with unit sample weights. Permit
`inputs=data, quadrature=q` as an advanced path for supplied non-coordinate input
features. Require sample alignment and compatible variable schemas. The ordinary
coordinate path only passes `quadrature`.

### Freeze and attached variations

```python
boundary_change = net.cost(
    quadrature=boundary,
    expression="-u*v / sqrt(v_x**2 + v_y**2)",
    freeze=["u", "v_x", "v_y"],
)
mass = net.cost(
    quadrature=q,
    expression="u",
    variation=boundary_change,
)
```

An attached variation can be any cost expression, including a sum or nonlinear
composition. Provide a functional attachment method for an already constructed
composite: `combined.with_variation(correction)` returns a new cost graph node and
does not mutate `combined`. `net.cost(..., variation=...)` is shorthand for this.

Arithmetic includes reflected scalar operations, unary minus, addition,
subtraction, multiplication, division, and real constant powers. Implement
`__radd__` so `sum(costs)` works. Provide cost-aware `cpde.exp`, `log`, `sqrt`, `sin`,
`cos`, and `tanh`. A cost cannot be used as a Python boolean or implicitly converted
to a float. Support cost-to-cost powers later if needed; initial powers have fixed
scalar exponents and explicit real-domain checks.

### Explicit evaluation without training

Add `cost.evaluate()` returning a scalar, and `cost.evaluate(gradient=True)` returning
the value and weight gradients with the same layer layout as network state arrays.
These calls do not change weights, optimizer moments/steps, gradient history, or
training progress. Use scratch weight-gradient storage or save/restore the relevant
buffers. Do not implement evaluation by performing a training step with zero rate.

An optional inspection method returns generated source, component dependencies,
freeze symbols, and compile diagnostics. It should not upload or train merely to
inspect an expression.

Extend `net.eval` to accept a Quadrature as well as existing Data. For a quadrature,
accept an explicit derivative list and generate its coordinate-input closure; return
the existing independent Data result. For example,
`net.eval(q, derivatives=["", "x", "y"])`. Evaluation without a derivative list
requests values only on the quadrature path; the existing Data path retains its
current full-schema behavior. This also lets geometry preparation callbacks inspect
the network without requiring users to construct internal coordinate matrices.

## Quadrature representation

### Shape and dimensional rules

| Quantity | Representation |
| --- | --- |
| Evaluation points | finite array `(N, d)`, ambient dimension `d` in 1..3 |
| Weights | finite array `(N,)`, no implicit normalization |
| Integration dimension | integer `k` in 0..d |
| Variables | `d` unique single-letter names; preserve existing naming constraints |
| Tangents and normals | optional unit-vector arrays `(N, d)` |
| Segment cells | integer array `(M, 2)` of point indices |
| Triangle cells | integer array `(M, 3)` of point indices |
| Tetrahedron cells | integer array `(M, 4)` of point indices |

Infer the cell dimension from its column count and validate it against `dimension`.
Require `variables` and `dimension` explicitly in the first release. Dimension zero
is accepted for direct point quadratures, including endpoint sums; it has no mesh
construction path. Count weights must be explicit, such as an array of ones.

Internally copy constructor arrays and expose read-only views. Geometry preparation
uses float64. Coordinate network inputs retain float32 because the current engine
does; reject values that cannot be represented finitely after conversion. Keep
quadrature weights and scalar reductions in float64 for the new path. State clearly
that this does not turn network computations into float64.

Supplied quadrature weights may be signed, since numerical quadrature rules can
have signed weights. Generated simplex measure weights are nonnegative. These are
separate from legacy matching/equation density weights, which remain nonnegative.

Direct quadrature may be empty, with points shaped `(0, d)` and weights `(0,)`.
Its integral and base gradient are zero; skip network contexts and zero-sized GPU
allocations. This is useful when a user-defined region disappears. Compositions such
as mass/area must still reject division by zero. Automatic triangulation with too
few or rank-deficient points fails rather than pretending to represent an empty
domain. Empty connectivity is accepted only with an explicit dimension and correctly
shaped arrays. Existing Data need not gain empty-input support for this shortcut.

### Connectivity and triangulation

Use an optional `geometry` dependency extra containing SciPy. Import it only when
automatic Delaunay is requested. Other construction modes require only NumPy.

Automatic modes are `(k,d)=(2,2)` and `(3,3)`, operating on the convex hull of the
supplied point set. Do not apply 3D Delaunay to surface point clouds and call the
tetrahedra a surface. Do not silently project, jitter points, or select an alpha
shape. Report degeneracy and omitted/coplanar-point diagnostics to the caller.
SciPy exposes simplex indices and coplanar-point information in its
[Delaunay API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html).

For explicit cells, validate integer type, bounds, distinct indices within each
cell, dimension, and nondegenerate measure using a scale-aware tolerance. Reject
duplicate cells. Do not claim to prove that arbitrary supplied meshes are globally
nonoverlapping or manifold; connectivity and domain correctness remain the user's
responsibility. Preserve cell order and orientation.

### Quadrature rules

Start with `centroid` and `vertices`. For a k-simplex with edge-column matrix B,
measure is `sqrt(det(B.T @ B)) / factorial(k)`; use a numerically stable QR/SVD-based
implementation rather than relying on an unstable raw determinant near degeneracy.

- Centroid: one point per cell, weight equal to cell measure.
- Vertices: k+1 points per cell, each with weight `measure/(k+1)`.
- Preserve element-local occurrences in the vertices rule. Shared coordinates may
  appear repeatedly, with distinct normals/tangents. Do not average normals to merge
  them. A future scalar-only optimization may merge compatible occurrences.
- Keep input vertices and cells as construction metadata. `q.points` always contains
  final evaluation points. Expose `q.vertices`, `q.cells`, and `q.cell_ids` for mesh
  modes; document their absence for direct quadrature.
- Parameters sampled on original mesh vertices are not implicitly interpolated.
  Per-point parameters must match final `q.points`. Users evaluate their fields there.
- Higher-order rules can be supplied as direct quadrature initially.

Both initial rules integrate affine densities exactly on straight simplices.
They approximate nonlinear network densities and curved geometries.

### Orientation and available directions

- Segments define `T=(p1-p0)/length`.
- In 2D, segment `N=(T.y,-T.x)` is the right-hand normal. Counterclockwise closed
  boundaries therefore produce outward normals. Reversing a segment reverses both.
- Embedded triangles in 3D define N from `(p1-p0) cross (p2-p0)`, normalized.
- A curve in 3D has T but no uniquely inferred N. A surface in 3D has N but no
  canonical single tangent T. Full-dimensional regions do not infer an interior N.
- Explicit N/T arrays refer to final evaluation points, including in mesh modes.
  Validate shape, finiteness, unit norm, and tangency/orthogonality where applicable.
  Do not silently normalize or flip supplied directions.
- Explicit smoothly varying directions may intentionally differ from flat element
  directions. Do not require an explicit surface N to equal the facet normal.
- Missing directions only cause an error when an expression requests them.

Geometry directions are fixed numeric fields during local expression differentiation.
If their dependence on the solution contributes to a physical derivative, that
contribution belongs in the user-supplied variation.

## Session and input ownership

Quadrature starts as a session-independent host object. Each session maintains its
own uploaded binding for a quadrature identity and revision. Multiple costs in one
session share that binding. The network determines the session; the constructor
does not need `session=`. Cross-session costs cannot be composed.

For the default coordinate path, input channel count must equal ambient dimension.
The binding creates coordinate values, first-coordinate derivatives equal to the
identity matrix, and zero higher input derivatives. Spatial network derivatives are
ambient derivatives, even for a lower-dimensional curve or surface.

Collect the derivative closure from every component read by the expression,
including frozen ones. A frozen derivative still needs to be evaluated. Keep
read dependencies separate from components that receive nonzero dE/dY. Reuse existing
schema/context mechanisms and preserve the order-6 limit.

Existing Data remains available for arbitrary input features and network chaining.
Do not imply that feeding one network's evaluation into another automatically
enables joint training through both networks. Coordinate cost construction does not
expose the internal Data adapter as a mandatory user step.

Costs retain their uploaded data, quadrature bindings, parameter arrays, modules,
and child graph nodes. Closing a referenced object fails clearly; explicit graph
release must not invalidate a child still used elsewhere. Session shutdown releases
all bindings/modules/graphs. Host quadrature can be reused in a new session.

## Expression language and differentiation

Use a restricted parser and typed scalar intermediate representation. Do not execute
Python expressions with eval or splice user identifiers directly into CUDA source.
Map validated symbols to generated slots.

| Syntax | Meaning |
| --- | --- |
| `u`, `v` | named network output values |
| `u_x`, `v_xy` | named ambient output derivatives |
| `coord.x` | evaluation-point coordinate |
| `N.x`, `T.x` | geometric direction components |
| numeric literals and parameter names | constants or numeric parameter fields |
| `+ - * / **` and parentheses | scalar arithmetic; constant exponent initially |
| `exp log sqrt sin cos tanh` | initial function whitelist |

Accept constant densities such as `1` and signed densities. Retain the old equation
parser for its existing fast path; general equation residuals can lower through the
new expression compiler, with the same squared-residual/rhs semantics. Expression
costs allow constants directly; old equation-form rules need not change silently.

Lower legacy cost descriptions into the same component model when a new feature
requires it, including freeze. Alternatively retain specialized kernels with an
equivalent component mask. Preserve complete target schemas and row mappings for
matching. A backend must reject an unsupported combination before execution rather
than silently ignoring freeze on an older cost form.

Parameters are finite scalars or arrays of length N. They are held fixed during
differentiation and uploaded separately from compiled code. Reserve geometry/function
names and reject collisions and unavailable derivative orders. Unsupported syntax,
functions, or implicit broadcasting fail before GPU execution. Defer branches,
comparisons, abs/min/max, arbitrary Python functions, and piecewise derivatives.

Reverse differentiation produces partial derivatives with respect to output matrix
components, not weights. Existing backpropagation computes weight gradients. Keep
accumulation for repeated symbols such as `u*u` and repeated derivative aliases.
Normalize mixed-partial names consistently with the current implementation.

`freeze` contains canonical network component names appearing in that expression.
Reject unknown/unreferenced entries. Freezing `u` does not freeze `u_x`. Set local
derivative seeds for frozen symbols to zero, but retain their forward values and
dependencies. Freeze is local to one leaf and does not affect other costs. Fixed
parameters and geometry are already non-differentiated; reject redundant freeze
entries for them. Named subexpressions/user-defined functions can be a later syntax
extension rather than an implicit first-release feature.

Numeric errors include invalid logarithm/square root/division/power domains, nonfinite
density values, and nonfinite local/weight gradients. Report cost, expression, and
sample location when available; do not silently clamp or insert epsilons. Handle
integer powers separately so negative bases do not acquire unnecessary log calls.
Handle p=0 and p=1 derivatives explicitly. No fast-math by default.

## Cost graph semantics

Represent costs as an immutable directed acyclic graph with scalar constants,
integral leaves, arithmetic/function nodes, and variation attachments. The public
name remains Cost; compatibility wrappers can retain CostSum without a separate
public Functional or Integral class.

For an ordinary integral leaf, V is its scalar value and G its gradient contribution.
An attachment has

    V(attach(A, B)) = V(A)
    G(attach(A, B)) = G(A) + G(B)

Arithmetic uses ordinary scalar values for chain-rule multipliers and complete G
contributions for derivatives. In particular,

    G(A*B) = V(B)*G(A) + V(A)*G(B)
    G(A/B) = G(A)/V(B) - V(A)*G(B)/V(B)^2

Reverse traversal sends the same incoming multiplier to both children of an
attachment. Forward traversal takes only the base child's value. The variation
subgraph's values may still be required to compute its gradient when it contains
nonlinear operations. Distinguish value dependencies from gradient dependencies.

Memoize shared node evaluation and add incoming derivative multipliers. Do not
deduplicate multiplicity: `C+C` has twice the value and gradient. Handle a correction
reused as both an attachment and an ordinary objective term with both roles intact.
Reject cycles, closed nodes, foreign networks/sessions, and nonfinite scalar constants.

Do not simplify a node to a gradient-free constant merely because its value is
constant: `expression="1", variation=B` has a constant base density and potentially
nonzero gradient. Preserve frozen-symbol identity during common-subexpression
elimination. Graph optimizations must preserve both V and G.

This is a custom first-derivative contract. Freeze or an arbitrary variation need
not produce the ordinary derivative of the reported numeric value under arbitrary
perturbations. Hessians and differentiation through the gradient are out of scope.

## Compilation and executable structure

### Proposed compiler boundary

Keep the externally built persistent server. Add optional expression compilation
inside it using NVRTC, with CUDA Driver API module loading. Python prepares the
restricted expression and generated device source; it does not need a CUDA compiler.
Build the server with an optional expression feature using NVRTC and the driver
library. A server without that feature still supports legacy costs and scalar graph
composition over supported leaves; it rejects unsupported compiled expressions
before training starts.

Only generated expression kernels are compiled for a new expression/freeze layout.
The server, forward/backward engine, optimizer kernels, scalar graph executor, and
CUB reduction wrapper are compiled once when the executable is built. Do not include
the network implementation in generated source or relink the executable for each
cost. Value and gradient kernels may share one generated module. This is the default
compilation boundary; profiling may justify fusion of particular local operations
without turning every problem into a complete executable rebuild.

NVRTC accepts CUDA C++ source and produces PTX/cubin that can be loaded through the
Driver API. This supports adding cost programs to an existing process without
rebuilding the network executable. See the
[NVRTC guide](https://docs.nvidia.com/cuda/nvrtc/index.html).

Use the existing runtime-created context and stream for compiled kernels. Do not
create a separate incompatible context for the modules. Initialize/check the current
context on the worker thread before driver calls. NVIDIA documents the context
sharing considerations in its
[Runtime and Driver API comparison](https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html).

### Generated programs and cache

Generate value-only and gradient kernels, sharing the expression IR and slot map.
One thread owns each sample and accumulates all local component derivatives without
cross-thread writes. Pass derivative-major output pointer tables matching the
existing engine, numeric field pointers, quadrature weights, output/derivative
counts, chunk width, valid count, sample offset, and the scalar graph multiplier.

The gradient kernel initializes every required dE/dY entry, writes zero for unused
or frozen components, and multiplies by quadrature weight and graph multiplier
exactly once. Padded lanes and zero-weight samples exit before evaluating expressions
or reading external arrays; clear their gradient/value slots first. This is necessary
for denominators that would be singular on padded zero inputs.

Compile on explicit `cost.compile()` or first use before training. Never compile in
an epoch loop. Cache keys include canonical IR, freeze set, component/field layout,
scalar/array parameter layout, compiler/ABI versions, compute architecture, and
compiler options. Numeric weights, coordinates, coefficient values, sample count,
and chunk size are runtime data and do not trigger recompilation. Serialize modules
only as a cache, not as portable network state. Validate cache manifests and write
atomically. In-memory caching is required; persistent cache is a later optimization.

No generated source contains raw arbitrary user C++ or requires recompiling the
network engine. Compile errors include the originating expression and compiler log.

### Signed reductions

The CUB update has replaced `pass()`'s `cublasSasum` with a precompiled
`DeviceReduce::Reduce` wrapper combining signed double sums and invalid-value
locations. Existing cost kernels fuse a block reduction, reuse temporary storage,
and keep results on the device stream without host scalar synchronization per chunk.
Reuse this infrastructure for general densities so cancellation and negative values
are preserved. Keep network and dE/dY buffers float32 initially.

For newly generated kernels, start with per-sample weighted densities followed by
CUB. Profile before fusing a
block reduction into generated kernels; fusion saves a temporary density array but
adds register/shared-memory/compilation considerations. A dot product with a vector
of ones is a valid signed fallback for the existing cuBLAS path, not the preferred
new reduction design.

CCCL 3.1 added explicit not-guaranteed, run-to-run, and GPU-to-GPU determinism options
for DeviceReduce. Default to run-to-run reproducibility. Cross-GPU reproducibility
can be a later explicit mode and does not by itself make cuBLAS/backpropagation
bitwise reproducible. Pin a released compatible CUDA/CCCL/compiler combination and
validate its language-standard requirements instead of coding against unstable
online documentation. See the
[CCCL 3.1 release](https://github.com/NVIDIA/cccl/releases/tag/v3.1.0) and
[NVIDIA determinism discussion](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/).

## Execution of a training update

1. Prepare or validate the current geometry revision and all numeric bindings.
2. Reserve contexts for the complete graph, including variation branches.
3. Evaluate integral values in chunks and reduce them. No weights change.
4. Evaluate the scalar graph and compute incoming multipliers for each integral
   leaf, including all variation edges. Use a small precompiled GPU graph executor
   over serialized node records and double scalar buffers. A CPU reference is useful
   for tests, but normal execution should not download every integral value just to
   calculate its multiplier. Copy report data to the host only when requested.
5. Evaluate each required leaf again in chunks, form scaled dE/dY, and backpropagate
   into the common weight-gradient buffers. Recompute outputs rather than retaining
   all intermediate network activations for the entire dataset.
6. Check the complete gradient for finiteness, then perform exactly one optimizer
   update. Any earlier evaluation error performs no optimizer update.
7. Report/checkpoint/stop at the established update boundary.

Start with the general two-pass implementation. Later preserve a one-pass fast path
for sums with known constant multipliers. Avoid implementing graph fusion before
the general path is verified.

Explicitly initialize common weight gradients even if every leaf has zero multiplier
or all output components are frozen. Do not reuse a previous gradient. Continue
normal optimizer semantics: zero current gradient does not necessarily imply zero
Adam movement when existing moments are nonzero.

Chunking limits working memory, not full dataset residency. Every term and chunk
uses one weight state and one geometry revision. Prevent any numeric update between
the value and gradient passes. Context growth must not invalidate shared module,
graph, or quadrature bindings.

## Geometry and numeric updates

### Explicit replacement

Provide `q.replace(...)` using the same construction arguments as Quadrature.
It preserves logical identity and increments a revision only after a fully validated
replacement is ready. Exposed arrays remain read-only. Different point counts are
allowed; variables, ambient/integration dimensions, and required field schemas stay
compatible with existing costs. A changed schema requires rebuilding affected costs.

Replace session bindings and coordinate adapters transactionally, invalidating
contexts and sample caches as needed. Reuse compiled programs when their slot layout
is unchanged. New sample counts require consistent replacements for per-point
parameters, targets, and explicit feature Data. Never silently resize or interpolate
those arrays. Add explicit `cost.set_parameters(...)` for numeric updates; values
must retain the compiled scalar/array layout unless the program is rebuilt.

Fixed-geometry training requires no callbacks. Frozen output components are still
re-evaluated on every forward pass, so freeze does not itself require a refresh hook.

### Separate extension for GPU geometry updates

Frequent updates should be device work scheduled by the server, not a mandatory
Python callback every epoch. Initial implementation supports static geometry and
explicit replacements between training calls. Reserve a device-updater interface;
implementing particular projection, clipping, or remeshing algorithms is separate.

An updater declares the network outputs/derivatives and geometric/numeric fields
it reads, the quadratures/fields it writes, and its refresh condition. The scheduler
arranges the declared forward evaluations and stream dependencies. Updaters do not
alter network weights, optimizers, or the cost graph.

Use four conceptual update classes:

| Update | Changes | Consequence |
| --- | --- | --- |
| Field update | Weights, N/T, external coefficients, active mask | No connectivity change; derived measures still must remain consistent |
| Point motion | Vertex/evaluation positions with fixed connectivity | Recompute dependent quadrature points, weights, and directions |
| Selection or compaction | Active samples/cells within a fixed candidate buffer | Keep count/mask on GPU; update all aligned fields together |
| Topology replacement | Connectivity and possibly capacity | Separate user-supplied stage with declared scratch/capacity requirements |

Each device binding contains buffer capacities, active counts or masks, schema,
revision, completion status, and pointers to points/weights/N/T/optional cells.
Schemas remain fixed across routine updates. Scratch and final buffers are allocated
in advance. Launch bounded kernels and mask unused lanes; no per-update host count
readback is required by the basic contract. Capacity overflow raises a status and
requests an explicit growth/rebuild point rather than corrupting or silently dropping
geometry. A host-side launch scheduler is compatible with device-resident data; the
contract does not require all launches to originate on the GPU.

Build into staging buffers and commit a coherent revision after validation. Every
array indexed by a sample must share the same active ordering/count. Refresh
per-point parameters and targets from their supplied update definitions; do not
carry old sample-indexed fields across point motion or compaction silently. For mesh
mode, moving vertices invalidates quadrature and direction fields. For direct
quadrature, the updater supplies the correct transformed weights/directions.

Run updates before evaluation at the required weight revision and hold geometry
fixed through both value and gradient passes. Increment weight revision on explicit
state loading as well as optimizer steps. Post-update physical reports require
geometry for the new weights. Less-frequent refresh is an explicit lagged-geometry
approximation; reports identify that state. A geometry-only readout never triggers
an optimizer update.

An iterative updater declares a maximum iteration count, convergence condition,
and failure status. If a projection moves evaluation points, forward evaluation at
those changed points must be scheduled for subsequent iterations. Weak gradients,
invalid measures, element inversion, failure to converge, and capacity overflow
cannot publish half-updated geometry. Related interior/boundary bindings commit
together. Abort a failed preparation before updating weights.

Geometry update programs are not differentiated automatically. The cost's variation
continues to supply the physical shape contribution. Updating a boundary point set
does not construct the matching interior domain, and projecting points onto a level
set does not supply correct integration weights by itself. Selection masks similarly
do not resolve a subcell boundary without a user-specified integration construction.

Treat device-generated geometry as authoritative. Downloading a host snapshot is an
explicit operation; initial constructor arrays must not masquerade as the current
device points after an update. Save committed revisions for reproducible restart.

A host preparation callback may be a later fallback for user algorithms that cannot
run on GPU, but it is not a baseline deliverable. It would require an explicit pause
and command-service state in the protocol; it cannot safely be bolted onto the
current training response loop.

## Protocol, reporting, and persistence

The current handshake supports legacy CPDE/1 and CPDE/2 with a gradient-diagnostics
capability bit. Extend capability negotiation and version changed command layouts
for the proposed quadrature/graph features; preserve supported legacy operations.
Do not append fields to old command layouts and hope older servers ignore
them. New commands cover quadrature binding/replacement, program registration,
graph registration, graph evaluation/training, and explicit replacement transactions.
The GPU-updater extension adds registration/status messages without requiring
per-epoch Python transactions.

Represent graphs in topological order with explicit node/child IDs and one root.
Represent numeric fields separately from compiled programs. Validate handles, sizes,
schema references, network ownership, bounds, and full message consumption on the
server as well as in Python. Send component mappings with a versioned kernel ABI.

Reports must contain the actual root value. The Python code currently derives total
by summing leaf values; this is wrong for products, quotients, and attachments.
For arbitrary graphs report `total`, named `leaf_values`, phase, completed update,
and geometry revision. Keep legacy additive `contributions` where they truly sum to
the total. Do not invent an additive attribution for a nonlinear objective. Variation
values are excluded from the total and appear only in explicitly requested diagnostics.

Retain optimizer/network snapshots and explicit NPZ disk I/O. Extend metadata with
the cost graph semantic hash, expression/freeze/variation definitions' hashes,
binding identities, relevant numeric revisions, and compiler ABI identifiers.
Checkpointing remains at update boundaries. No automatic downloads in ordinary
training completion.

Provide explicit Quadrature save/load for points, weights, directions, variables,
dimension, identity, and revision; mesh metadata is optional. Never pickle callbacks
or arbitrary code. Network state alone does not recreate a complete geometry or
cost graph. Users reconstruct the graph and load its required inputs when resuming.

Keep structural graph identity separate from moving numeric revisions so a geometry
refresh does not reset training progress or optimizer state. For fixed inputs,
resume validates input identities/versions as before. For moving inputs, checkpoint
the committed binding metadata; exact replay additionally requires saved geometry
and any external state used by the user's update algorithm. Repreparing after resume
is explicit and cannot promise bitwise replay for arbitrary stateful geometry code.

## Proposed source layout

| File or directory | Responsibility |
| --- | --- |
| `cpde/api.py` | retain Session/Data/Network public facade and compatibility dispatch |
| `cpde/quadrature.py` | host quadrature construction, validation, revisions, save/load |
| `cpde/_geometry.py` | simplex rules, directions, optional Delaunay adapter |
| `cpde/_expressions.py` | restricted parser, typed IR, symbols, local differentiation |
| `cpde/_codegen.py` | CUDA generation, ABI manifest, canonical compile keys |
| `cpde/cost.py` | cost leaves, graph arithmetic, freeze/variation API and ownership |
| `cpde/_graph.py` | graph validation, traversal, serialization, CPU reference evaluator |
| `cpde/_prepare.py` | retain legacy preparation and shared derivative closure helpers |
| `cpde/_wire.py` | versioned messages, capabilities, graph reports |
| `cuda/cpde_server.cu` | command dispatcher and training state transitions |
| `cuda/quadrature.hpp` | session-owned geometry/coordinate bindings and revisions |
| `cuda/expression_program.hpp` | NVRTC compilation, module lifetime, kernel ABI |
| `cuda/cost_graph.hpp` | scalar graph records and device execution for values/multipliers |
| `cuda/cost_execution.hpp` | value/gradient passes and common gradient accumulation |
| `cuda/reduction_kernels.cuh` | precompiled CUB reduction wrapper and finite checks |
| `cuda/cost_kernels.cuh` | retain legacy matching/polynomial kernels; add q and scaling support |
| `cuda/protocol.hpp` | framing and new protocol constants/records |
| `CMakeLists.txt` | optional expression compiler/driver dependencies |
| `pyproject.toml` | optional SciPy geometry dependency; NumPy remains core |
| `tests/` | geometry, expression, graph, protocol, host and real CUDA validation |
| `examples/` | concise user-facing examples using the implemented API |

Leave `cpde_engine.cuh` and `derivative_kernels_v4.cu` numerical algorithms unchanged
unless a concrete integration defect requires a targeted correction. Structural
refactoring should move responsibilities without rewriting optimizer/derivative math.

## Implementation sequence and exit criteria

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 1 | Lock API contracts, add signed reductions and non-mutating gradient evaluation | Existing ten tests pass; negative/cancelling sums and read-only gradient evaluation pass |
| 2 | Quadrature constructors, simplex rules, directions, session bindings, coordinate adapters | Measures, orientation, data layout, direct and supplied-cell paths verified; legacy costs accept quadrature |
| 3 | Scalar cost graph over existing leaves, arithmetic, functions, variation attachment, new reports | Shared-node and nested variation gradients correct; reports equal root values; graph training equals a reference |
| 4 | General expression IR, local derivatives, freeze, CPU reference, generated source | Analytic and finite-difference cases pass, with stop-gradient-aware checks for freeze |
| 5 | NVRTC integration, generated kernels, caching, common execution path | Compiled values and gradients match reference on real CUDA, including padding and signed reductions |
| 6 | Transactional numeric replacements between calls and reserved GPU update contract | Point-count changes, cache invalidation, atomic field replacement, and failure handling pass |
| 7 | Snapshot/resume integration, final examples/docs, full acceptance | Existing API regressions pass; new examples run end-to-end on CUDA and limitations are documented |

Phases 2 and 3 both depend on phase 1. Phase 4 can be developed without CUDA, but
phase 5 remains necessary for production acceptance. Deliver fixed-geometry use
before the separate GPU moving-geometry extension. Baseline completion does not
claim automatic moving-domain training. Do not label that extension complete while
requiring stale quadratures during a multi-update run.

## Validation matrix

### Geometry and bindings

- Known interval length, triangle area, tetrahedron volume, square area, cube volume.
- Exact affine integrals for centroid and vertex rules, plus nonlinear convergence.
- Orientation reversals preserve scalar measures and reverse signed fluxes.
- Closed polygon/polyhedron constant-vector flux is zero; affine vector flux agrees
  with the divergence theorem on the represented geometry.
- Distinct element normals survive repeated vertex positions and nonlinear N usage.
- Degenerate/out-of-range/repeated-index cells, incompatible dimensions, malformed
  directions, unavailable N/T, duplicate cells, and missing optional SciPy fail clearly.
- Explicit zero-weight and signed-weight quadratures behave as specified.
- Empty quadratures yield zero without allocating invalid contexts; composite
  division by their zero measure fails before any optimizer update.
- Coordinate input matrices through order 6 have correct identity/zero entries;
  surface derivatives are ambient, and arbitrary-feature Data remains supported.
- Sharing, replacement, release, and cross-session bindings preserve ownership.

### Local expressions and freeze

- Constant, negative, polynomial, rational, and each supported transcendental density.
- Multiple outputs, mixed derivatives, repeated components, scalar/array parameters.
- Density values and dE/dY checked against an independent CPU evaluator/analytic cases.
- Freeze leaves values unchanged, zeroes only selected local derivatives, and still
  requests their forward derivative closure. For frozen components, finite-difference
  checks hold their forward values fixed rather than recomputing them with perturbations.
- `-u*v/sqrt(v_x**2+v_y**2)` with u/v_x/v_y frozen has only the expected v derivative.
- Invalid operations in padded or zero-weight samples are never evaluated; invalid
  active samples abort before updating weights or optimizer state.

### Graphs and training

- Addition, scalar/reflected operations, product, quotient, powers, functions, and
  `sum(costs)`; shared A used in `A*A`, `A+A`, and several parent branches.
- Attachment leaves the reported parent value unchanged and adds exactly its
  gradient. Test constant-valued parents, nested attachments, composite variations,
  and shared corrections used both normally and as variations.
- Whole-objective weight gradients match finite differences for ordinary costs.
  Use an analytic custom-gradient reference for arbitrary freeze/variation graphs;
  naive finite differences of their values are not generally the intended derivative.
- Full data versus chunks, several chunk widths, final partial chunks, multiple
  quadratures, and zero incoming multipliers agree within numerical tolerances.
- Exactly one optimizer update occurs per root gradient; zero current gradients
  never reuse stale buffers; Adam/RPROP continuation and reset semantics persist.
- Nonlinear objective reports contain the root value, never a sum of leaf values.
- A read-only value/gradient evaluation leaves optimizer history and future updates
  identical to a control run without that diagnostic evaluation.

### Moving geometry and physical variations

- In 1D, moving endpoint tests validate value and boundary-gradient signs.
- For v=x^2+y^2-R^2, analytic area is pi*R^2 and its derivative is 2*pi*R;
  perimeter is 2*pi*R and its derivative is 2*pi. Use supplied current-domain
  quadrature and analytical first variations to validate both examples.
- Verify mass/area mean penalties, including the area derivative in the quotient.
- Repeat at finer geometric/quadrature resolution. Continuous shape derivatives
  approximate a continuum functional and need not equal the exact derivative of a
  finite polygonization algorithm. Use convergence tests rather than claiming an
  exact discrete gradient for arbitrary remeshing.
- In the separate GPU update extension, updates atomically replace all related
  fields, may change active sample count, and run at the right weight revision for
  before/after/final reports. Check overflow, invalid cells, and bounded iteration
  failures without copying complete geometry back to Python.
- A preparation error or stop request cannot publish partial geometry or perform a
  partial optimizer update. Resume preserves its documented binding/geometry contract.

### CUDA and regression gates

Keep all ten existing integration tests. Update the serial host probe for new source
layout and use an explicit test-only implementation of generated sample programs;
do not pretend the launch-substitution probe exercises NVRTC or driver loading.

Real CUDA tests must cover compilation/module loading in the existing context,
cuBLAS interoperability, generated-kernel padding, reductions, streams, memory
lifetime, cache reuse, replacement with growth, and the entire training graph.
Run compute-sanitizer on representative multi-chunk, multi-quadrature cases.
No real CUDA validation has been performed in this workspace so far; it is a required
release gate, not an optional follow-up after declaring the implementation complete.

## Decisions to confirm during implementation review

NVRTC kernel-only compilation and centroid as a simple default are confirmed.
Pin the supported CUDA/CCCL/compiler versions and benchmark reduction precision and
fusion on the target GPU during implementation. Develop the GPU geometry updater
as a separate extension using the proposed buffer/revision contract; its algorithm
set and public syntax are not yet settled. Keep default meshing/quadrature methods
minimal rather than expanding them into a general geometry package.

Do not broaden scope to point generation, constrained meshing, implicit-surface
reconstruction, multi-network differentiation, higher-order weight derivatives, or
automatic curvature/shape calculus. Users provide specialized geometry and variation
expressions through the same Quadrature and Cost interfaces.

## Retaining forward results in shifted memory regions

Saved 2 November 2026. This is a future implementation plan. The existing CUDA
engine has not been changed to implement retention. Conditional recomputation
remains the current execution strategy, with one exception: the most expensive
eligible forward chunk runs last and is consumed directly by backward after
scalar evaluation. Retention would extend this saving to multiple live blocks.

The layer banks store floats independently of their interpretation as derivative
rows and samples. A context with D derivative rows, W samples and H neurons needs
D*W*H floats for its forward values. Distinct regions of the same bank can retain
several forward results without merging their derivative sets or copying them.
Execution remains sequential, with one accumulated weight gradient and one update.

### Separate structure from memory binding

Cache the derivative schema and dependency tables once. Bind them to a storage
region with its own base offset and sample width. The current context lookup uses
schema, shape and mode; future lookup must distinguish identical shapes bound to
different regions. Context propagation must select the same binding across layers.

ADDR and STCK contain absolute pointers, so changing only a memory object's base
does not relocate all references. Prepare offset-specific pointer tables once,
including the derivative stack, then switch bindings without rebuilding dependency
tables or copying numerical arrays. Separate ownership of shared metadata and
non-owning bank views; releasing a binding must not free the bank or shared tables.

### Retain only forward results whose backward pass must wait

Use the existing scalar-graph analysis. A cost with a fixed multiplier runs forward
then backward and releases its working region. A cost with a value-dependent
multiplier runs forward into a retained region. After scalar evaluation, activate
that region, compute the local output gradients and run backward without another
network forward. Include attached variations in the multiplier analysis.

Retain the required values and derivatives at every layer, including input features.
Output matrices alone are insufficient. The temporary activation cache used by
D(...) can remain shared: backward already regenerates those activation derivatives
from retained layer values. Retention does not remove local derivative work inside
backward or the need to evaluate the cost's output-gradient kernel.

### Plan capacity before execution

For each layer, sum H*D*W over simultaneously retained blocks and add reusable
working storage. Reserve/grow the banks before live forwards exist, then prepare
the bindings. Growth changes base addresses and must invalidate/rebuild affected
pointer tables. Track a bank generation and input validity per destination region;
the current single loaded-input record is insufficient for multiple live regions.

Initially preserve each binding's adjacent forward/gradient layout, reserving
2*H*D*W floats. A later refinement can retain only forward arrays and share gradient
buffers across sequential backwards. Do not scale temporary scratch, ones, loss,
or reduction storage by the sum of retained blocks when a maximum working size is
sufficient. Use checked sizes and include pointer/dependency metadata in estimates.

### Retention is per chunk

Every delayed chunk needs its own live region. Keeping only one chunk-sized region
for a many-chunk cost still requires recomputation. Identify retained blocks by
cost, chunk, input version, weight version and storage binding. Release a region
only after its backward has consumed it; all forwards/backwards in one update use
the same weights. Reusing a leaf in the scalar graph must not duplicate its storage.

### User choice and implementation order

Proposed policies, with public syntax not yet decided:

- Recompute: keep the current bounded working storage and repeat affected forwards.
- Retain: reserve enough storage for all delayed blocks and avoid repeated forwards.
- Automatic with a memory budget: retain selected blocks and recompute the rest.

Report estimated storage, retained blocks and expected repeated forwards before
training. Cost grouping remains independent of this policy. Poisson's additive
interior and boundary costs do not benefit from retained forwards.

Implement shifted bindings and explicit retention first, then budgeted selection.
Compare retained and recomputed values, gradients and optimizer updates. Count
actual forwards; test identical-shaped bindings, different schemas, chunk tails,
frozen components, variations, growth/rebinding, data replacement and error recovery.
Verify retained regions survive intervening context execution and scratch use.
Real CUDA checks are required for pointer-table lifetimes and stream ordering.

## Third problem eigenvalues in a deformed solid torus

Use one real network u with H=-Delta inside the domain and u=0 on its boundary.
The training objective combines the squared Schrodinger residual, the global
normalization penalty (integral(u**2)-1)**2, and a boundary-value penalty. E is a
scalar parameter held fixed during each 100-epoch block, not another network.
Use volume and surface quadratures for these integrals, with coefficient one on
each objective term: equation + (mass - 1)**2 + walls. Do not rescale mass to a
domain average or add manual 100/1000 penalty multipliers.

The user explicitly permits a Python loop for this example: call net.train for
100 epochs, set E=integral(-u*Delta(u))/integral(u**2), update the residual parameter,
then continue training without resetting optimizer state. Use the same bulk
quadrature for both parts of this quotient and the residual, so this E minimizes
the discretized squared residual for fixed u. Reject a zero/nonfinite denominator.

Deform a standard solid torus by x=a(Y)*X, y=Y, z=b(Y)*Z, where
a(Y)=1+0.08*cos(Y/R) and b(Y)=1+0.10*sin(Y/R). Both factors are positive.
With X=x/a(y), Z=z/b(y), define
F=(X**2+y**2+Z**2+R**2-r**2)**2-4*R**2*(X**2+y**2).
F=0 is the wall and F<0 the solid torus. Use F only for geometry and normals;
the network directly represents u. Do not turn F into a trial-function factor.

Sample the reference torus and map its points, carrying the volume and surface
Jacobians into weights. This preserves the hole without unconstrained Delaunay
tetrahedra crossing it. Compute outward normals from grad(F)/norm(grad(F)).
The infinite wall imposes u=0. Add the tangential penalty |grad(u) cross N|**2
to the same boundary cost. For a unit normal it equals the sum of both squared
tangential derivatives, requiring no tangent frame. If individual tangent vectors
are needed later, choose the coordinate axis with smallest absolute component of N,
set T1=normalize(N cross axis), and T2=N cross T1. The summed penalty is independent
of this choice; individual frames may change orientation across axis-selection seams.
