# cPDE API and execution reference

Start with the [example walkthroughs](README.md#fourier-fitting) for the delivered
scripts, per-call explanations, equations and results. This reference covers the
remaining API options and execution details.

## Session and backend selection

`Session(executable=None, *, backend=None, device=0, cache_warning_threshold=64)`
starts a persistent server. `backend` is `cpu`, `cuda`, or `None`. With an explicit
path, `None` detects the backend from its handshake; an explicit selection must
match it. `session.backend` exposes the detected value. CPU uses host memory and
accepts only `device=0`; CUDA selects the indicated GPU.

With no path, `backend='cpu'` resolves `CPDE_CPU_SERVER`, then `cpde_cpu_server`
on PATH. `backend='cuda'` resolves `CPDE_SERVER`, then `cpde_server` on PATH.
`Session()` defaults to CUDA discovery. Missing executables or mismatched backends
raise an error; there is no automatic compilation or fallback to another backend.
The test adapter reports `host-probe` and cannot satisfy explicit CPU selection.

See [platform build instructions](BUILDING.md). The same public network, data,
quadrature, cost, training and checkpoint interfaces work on CPU and CUDA.
Float32 roundoff can differ between BLAS implementations and reduction orders.

## Data and residency

The public `table` function accepts the requested derivative list and includes
all lower dependencies automatically:

```python
from cpde import table

derivatives = ['xxx', 'xxy', 'xyy', 'yyy']
schema = table(derivatives, variables='xy')  # 10 rows, including values.
```

Each row is `(total order, variable counts...)`, sorted by total order then by
the count tuple. The input and target arrays must follow this returned order.
This prepares metadata; numerical derivative values are still supplied by the
user. For example, `table(['xx', 'yy'], variables='xy')` includes values, x, y,
xx and yy, but not xy. A maximum order integer is not accepted.

`Data` accepts float32-compatible arrays of shape `(derivative, channel, sample)`.
The integer derivative table has rows `(order, count_variable_1, ...)`; orders 0–6
are supported by the retained kernels. The table must include the value and the
lower dependencies. Variable names identify count columns, independently of the
number or interpretation of input channels. Input derivatives are always supplied.
Public derivative-row order is preserved. Preparation maps to a sorted internal
closure and evaluation maps results back.

Arrays upload once when constructed. `net.eval(data)` copies output values and
derivatives into an independent server allocation. A subsequent evaluation does not
change that result. It can be another network's input or matching targets.

`data.to_numpy()` downloads a detached host copy. Editing that copy does not modify
`Data`. `data.set_samples(array)` explicitly uploads a same-shape replacement.
`with data.edit_host() as array:` downloads once and commits/uploads once on normal
exit. An exception in the edit block does not commit. Saving downloads only when
needed by the explicit Python `data.save(path)` call. Host mirrors are not retained
by default. `Data.load(path, session=session)` reads this NPZ format, including its
derivative schema and variable names. The legacy binary file loaders are not used
by the Python interface.

Server objects remain resident until `.close()` or session shutdown. Costs retain
their input/target objects; close costs before explicitly closing those data or their
network. Release unneeded evaluation results in long-running sessions. Session
shutdown releases all remaining objects. Each CUDA session uses one GPU; a CPU session uses host memory in its server process.

## Costs

Matching uses `net.cost(inputs=data, targets=targets, weights=weights)`. Weights must
have exactly shape `(output, derivative)` or `(output, derivative, sample)`, cover
the complete public schema, and be finite and nonnegative. They multiply squared
errors directly. Targets may have a different output channel name; numerical
channels correspond positionally. Target variable names and derivative tuples
must match the input schema. Different derivative-row orders are mapped explicitly.

Equation costs use `equations={name: expression}`, optional `parameters`, `rhs`, and
`weights` mappings. Expressions support output components, derivative suffixes,
parentheses, unary signs, `+`, `-`, `*`, and `**2`, with degree at most two in network
components. `u_xy` and `u_yx` resolve to the same component. Parameters are fixed
scalars or sample vectors. Nonzero output-independent terms belong in `rhs`.
Right-hand sides are sample vectors; omitted equations default to zero. Equation
weights are scalars or sample vectors and default to one. Unknown names and mapping
keys are rejected. Cost construction parses once and sends sparse monomials and
numerical coefficient arrays to the executable. The same monomials determine both
residuals and `dE/dY` for every output matrix; no user gradient kernel or recompilation is involved.

Contractions use `vector_a='u, v_x, u_yy'`, `vector_b` of shape `(K,)` or `(N, K)`,
optional sample-vector `rhs`, and scalar/sample-vector `weights`. Their execution
uses the same linear monomials as equation costs. Repeated vector components add.

Cost arrays are captured at construction. Reconstruct a cost to change its fixed
weights, coefficients, or right-hand side. Updating input/target `Data` changes what
existing costs consume on the next command. Cost addition preserves multiplicity:
`cost + cost` contributes twice. It introduces no averaging or balancing.

## Integral costs and quadrature

```python
from cpde import Quadrature

# Users create the points and connectivity. Cells contain point indices.
q = Quadrature(points=points, cells=triangles, variables='xy', dimension=2)
energy = net.cost(
    quadrature=q,
    expression='sqrt(1 + u_x**2 + u_y**2) + 0.5*u**2',
)
volume = net.cost(
    quadrature=q,
    expression='u',
)
objective = energy + 100.0*(volume - 1.0)**2
result = objective.evaluate(gradient=True)
print(result['value'], result['gradient'])
net.train(cost=objective, epochs=100, optimizer='adam')
```

An `expression` is integrated directly, including its sign; it is not implicitly
squared. `evaluate()` returns a float. `evaluate(gradient=True)` returns a dictionary
with `value` and one `gradient` array per weight layer, in the same layout as saved
weights. Evaluation preserves weights, optimizer state, gradient history, and
progress, including on recoverable numerical errors.

Quadrature takes finite `points` of shape `(N,d)`, `variables` with d names, and
integration `dimension` in `0..d`, for ambient d in `1..3`. Select exactly one mode:

- Supply a scalar `weights` to use the same weight at every point, or an array of
  shape `(N,)`. Weights may be signed; a scalar is a per-point weight, not a total
  measure (use `area/N` for N equally weighted area samples). You may also supply
  unit-vector `N` and `T` arrays of shape `(N,d)`. No rule is selected in this mode.
- Supply integer `cells` with 2, 3, or 4 vertex indices per segment, triangle, or
  tetrahedron. The default `centroid` rule uses one sample per cell; `vertices`
  retains separate occurrences at shared vertices, each weighted by cell measure
  divided by its vertex count. `q.points` means final evaluation points;
  `q.vertices`, `q.cells`, and `q.cell_ids` retain construction information.
- Use `triangulation='delaunay'` for full-dimensional convex planar/3D regions.
  Install the optional dependency with `pip install '.[geometry]'`. This is basic
  convex-hull triangulation, not constrained meshing or surface reconstruction.

Segment order supplies T and, in 2D, the right-hand normal N. Counterclockwise
closed polygons therefore have outward normals. Embedded triangles in 3D supply
oriented N. Other directions must be supplied explicitly. No direction is silently
normalized or flipped. A spatial normal and a space-time surface normal are not
interchangeable; all measures and directions must describe the intended integral.

Geometry is stored as immutable float64 arrays; the network still uses float32.
Simplex weights use float64 measures. `q.save(path)` and `Quadrature.load(path)`
preserve evaluation points, weights, directions, identity, and revision; they do
not preserve optional mesh construction metadata. Empty direct quadrature produces
zero integral and zero gradient. `net.eval(q, derivatives=['', 'x', 'y'])` generates
coordinate inputs and derivative closure automatically (nonempty points required).

Expressions support named outputs/derivatives, `coord.x`, `N.x`, `T.x`, numeric
literals, scalar/per-point `parameters`, arithmetic, constant powers, and
`exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`. Mixed derivative aliases are canonicalized.
Expressions may be indented multiline strings. Parameter names start with a letter
and contain only letters and digits; underscores are reserved for derivatives.
No arbitrary Python or CUDA text is executed. Unsupported syntax and invalid real
domains raise errors. There is no automatic epsilon or clamping. Expression size
is limited to 512 scalar instructions; split larger densities into costs.

Matching, equation, and contraction forms also accept `quadrature=q`; their
densities remain squared residuals. New quadrature-backed equations may use the
general expression functions. Plain `inputs=data` legacy calls retain their old
polynomial fast path. `inputs=data, quadrature=q` supplies non-coordinate features;
sample counts and variable names must match. Geometric weights are separate from
nonnegative matching/residual density `weights`. Quadrature-backed matching captures
target values at construction; reconstruct that cost to change its targets.

Costs support reflected scalar arithmetic, constant powers, `sum(costs)`, and
`cpde.exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`. Shared leaves retain multiplicity.
Graphs cannot be implicitly converted to a float or boolean. The graph evaluates
in the selected server backend. Costs with fixed multipliers run forward then backward in sequence.
For value-dependent multipliers, the scheduler keeps the most expensive eligible
forward chunk live across scalar evaluation and backpropagates it immediately;
the other affected chunks repeat forward. See the scheduling section below.
Reports contain the actual root `total` and `leaf_values`,
which need not sum to the root. Legacy additive reports retain `contributions`.

### Freeze and attached variations

For a compact shape-variation demonstration, run `python -m examples.attached_variation`.
The [script](examples/attached_variation.py) integrates over the ellipse {u>0}
defined by the network, then compares the gradient with geometry held fixed to
the gradient including boundary motion. It evaluates one shape; it does not train
on an evolving mesh.

A separate illustration of the same API, using a surface-energy density:

```python
energy = net.cost(
    quadrature=q,
    expression='sqrt(1+u_x**2+u_y**2)-contactcos+0.5*u**2',
    parameters={'contactcos': 0.5},
)
boundary_change = net.cost(
    quadrature=boundary,
    expression='u*(sqrt(1+u_x**2+u_y**2)-contactcos)/sqrt(u_x**2+u_y**2)',
    parameters={'contactcos': 0.5},
    freeze=['u_x', 'u_y'],
)
moving_energy = energy.with_variation(boundary_change)
# Equivalent attachment at construction: net.cost(..., variation=boundary_change)
```

`freeze` names referenced output components: their values are re-evaluated normally
but their local partial derivatives are zero. Freezing `u` does not freeze `u_x`.
Coordinates, parameters, and geometric directions are already fixed numeric fields.
An attachment retains the base value and adds the correction's complete gradient.
Corrections can be sums or nonlinear cost graphs, and attachments can be nested.
This is a custom first-derivative contract, not automatic differentiation through
mesh generation. Higher derivatives of the custom gradient are not implemented.

The shown boundary formula is for a drop footprint `u>0` whose contact line is
`u=0`, with nonzero boundary slope. It does not move or reconstruct that contour.
The test suite checks this boundary contribution against an analytical result and
finite differences after explicitly rebuilding a moving paraboloid footprint.

### Compilation and geometry refresh

`cost.compile()` prepares expression leaves explicitly; otherwise preparation
occurs before evaluation/training. CUDA compiles kernels; CPU uses the validated
scalar instructions directly, without NVRTC. On CPU, `compile_log` identifies the
interpreter and displayed CUDA source is a reference, not executed code. `leaf.inspect()` returns dependencies, freeze
symbols, instruction metadata, generated source, and available compile logs without
compiling or training. Source is generated from validated instructions. On CUDA,
NVRTC and the driver load only expression kernels into the existing GPU context. The
server caches programs in memory by generated source for its device. Numeric
parameters, points, weights, and sample counts are kernel arguments and do not
trigger recompilation when the expression/component layout is unchanged.

`q.replace(points=..., weights=..., N=..., T=...)` validates a replacement, preserves
identity, and increments revision. Supply one complete geometry mode; omitted
variables/dimension retain their old values. Direction schemas must remain compatible.
Costs rebind on next use. Per-point parameters must match the new point count; update
them with `cost.set_parameters(name=array)` while preserving scalar/array layout.
Explicit feature Data and captured targets require corresponding user updates or
cost reconstruction. Invalid replacements cannot start a partial training step.
Geometry changes retain optimizer state and structural stage progress; exact resume
checks still require the recorded geometry and data revisions.

Geometry stays fixed throughout each `train` call. Python can replace data or
quadrature between calls, but input-set updates during the server's training loop
are not implemented. The drop example is deferred until that contract is supplied.
Automatic point following,
remeshing, GPU geometry updates, and differentiation through geometry algorithms
remain separate work. Refresh explicitly between calls. Do not use a multi-epoch
call with stale geometry when the integration domain is intended to follow the
network after every update. Close graph nodes before explicitly closing children
they retain; session shutdown releases all remaining device resources.

## Contexts, memory bank, and chunks

Each cost retains its inputs, external data, derivative closure, and chunk width.
Layer contexts are installed before execution after the shared bank has been sized.
Only the closure needed by a cost is computed. Zero-weight matching derivatives
are excluded unless another selected derivative needs them. A value-only cost
therefore prepares value-only computation even if the data stores higher orders.

The existing per-layer storage and activation cache are reused across contexts.
The bank grows only when a new execution requires greater capacity; this rebuilds
pointer metadata without changing weights or optimizer buffers. A training call
sizes for its largest context before registering all its contexts. Switching
contexts overwrites forward intermediates; contexts do not retain activation tapes.

### Choosing cost groups and forward-pass work

Each `net.cost(...)` is an independent integration and execution context. Adding
costs or applying scalar functions to them does **not** merge their points or
derivative sets. To share a pass, put terms on the same points into one cost,
using a multiline `expression` with `+` between terms. The `equations` mapping
also groups multiple squared residuals into one cost. A grouped cost propagates
the union of the derivatives referenced by its expressions and their dependencies.

The server determines each leaf cost's multiplier from the scalar graph:

- A multiplier known before evaluating costs permits one forward pass followed
  immediately by that cost's backward pass. The next cost then runs in sequence
  and adds to the accumulated weight gradient. This covers weighted sums and
  attached variations with fixed scalar multipliers.
- A multiplier depending on computed cost values requires a value-only forward
  pass first. After all scalar values are available, the graph computes the
  multipliers. The scheduler runs the most expensive eligible context last,
  leaving its final chunk's activations live. It immediately runs that chunk's
  backward before switching contexts. The other affected chunks repeat forward
  then backward, adding their contributions. Overwritten activations are never
  reused, and no extra activation storage is allocated.

The selection estimates the saved network-forward work from execution width,
layer sizes, derivative row count and derivative orders. It is a heuristic, not
measured GPU timing. With chunking, only the last chunk remains live; the estimate
is for a chunk's allocated width (including padding), not the entire cost's point
count. Labels and leaf reporting order remain unchanged by execution reordering.

For one gradient evaluation **without chunking**, the forward count is below.
Where both multipliers depend on values, assume A's forward is more expensive:

| Objective | A | B |
| --- | ---: | ---: |
| `2*A + 3*B` | 1 | 1 |
| `A + 100*(B - target)**2` | 1 | 1 |
| `A / B` | 1 | 2 |
| `A.with_variation(B)` | 1 | 1 |
| `A.with_variation(B**2)` | 1 | 1 |

With chunking, fixed-multiplier chunks each need one forward; value-dependent
chunks each need two, except the one kept live, which needs only one. Avoiding
recomputation for more delayed chunks requires the future retention mechanism.

`target` here is a numerical constant. Analysis follows graph dependencies; it
does not perform general algebraic cancellation. Shared leaves are evaluated
once per stage, even when used multiple times in the graph. Variations affect
the prescribed gradient while retaining the base objective value. A value-only
evaluation needs one forward per chunk; requested post-update reporting adds
its own evaluation, independently of this gradient schedule.

Grouping can save repeated work when terms share points and derivatives. Keeping
costs separate avoids propagating many derivatives over a large point set when
only a small point set needs them. You choose the grouping in `net.cost`; the
server automatically selects the required schedule. Nonlinear operations inside
an expression are local to each point and need no extra forward. Nonlinear
operations on integrated costs can require recomputation. Moving an operation
inside the integral can change the mathematics: `(integral u)**2` is generally
different from `integral (u**2)`.

Poisson groups boundary values and tangential derivatives in one boundary cost,
and keeps the interior PDE in another. Their sum uses one forward then backward
per cost/chunk, with accumulated gradients and one optimizer update.

`chunk_size` is optional on `cost()` and `eval()`. It limits the execution width,
not data residency: all inputs, targets, and external arrays remain in server memory
(GPU memory on CUDA, host memory on CPU).
Every chunk of a term uses one fixed-width context. The final chunk receives finite
input padding, and its invalid lanes get zero loss and zero `dE/dY` before
any external target or coefficient is read. Evaluation copies back only valid lanes.
Derivative projections and sample slices are copied device-to-device into the bank;
there is no repeated host upload. An unchanged input slice already in the bank is
reused without another device copy. All terms and chunks see the same network weights.
The first backward pass overwrites the weight gradient; every subsequent pass adds.
RPROP or Adam then performs exactly one update.

The retained mixed-memory classes and loaders are unused by this interface. Their
behavior has not been extended or repaired, including their existing target handling
and equal-chunk assumptions.

## Optimizer state and reporting

```python
net.train(cost=total, epochs=1000, optimizer='rprop', state='continue',
          reset_steps=None, reset_moments=False, learning_rate=None,
          restore_every=0, restore_step=None,
          verbose=2, log_every=100, eval_every=100,
          print_outputs=False, save_every=100, diagnostics_every=100)
```

Continuation is the default. Omitting `optimizer` continues the loaded/current
method, or chooses RPROP for a fresh network. A method mismatch initializes fresh
optimizer state while retaining weights. `state='reset'` resets the selected
optimizer's intermediate buffers and counter while retaining network weights and
hyperparameters. `reset_steps=<positive float>` assigns all RPROP steps to that
value and clears both gradient-history buffers, so the next update cannot roll
back a step using the replacement size. A full RPROP reset uses the backbone's initial step `0.002`; the removed
Fourier driver no longer overrides it. `reset_moments=True` zeros Adam moments and
its bias-correction counter. `learning_rate=<positive float>` changes Adam's rate;
omitting it retains the current rate (initially `0.001`).

RPROP retains rollback for accepted updates and the ±20 weight bound. When that
bound rejects an update, the weight stays unchanged, its step resets to `0.1`,
and its stored gradient sign is cleared. The next inward update therefore starts
with step `0.1`, without rolling back the rejected update. This per-weight reset
is independent of `reset_steps` and the full optimizer reset. Zero gradients do
not move weights. Adam retains its kernel and weight-magnitude check but receives
correctly zero-initialized moments. `reset_steps` uses the backbone's step-assignment
routine and remains distinct from periodic restoration.

For long RPROP runs, enable the existing `reset_resile(R)` threshold-floor routine:

```python
net.train(cost=total, epochs=25000, optimizer='rprop',
          restore_every=1000, restore_step=1e-5)
```

After every 1000 cumulative network updates, any step strictly below `1e-5` is
raised to `1e-5`. Larger/equal steps and all weights remain unchanged by the
restoration itself. Both gradient-history entries are cleared only for restored
steps, preventing rollback with a replacement step that was never applied.
The interval uses the network's saved update counter, so it spans short training
calls and checkpoint reloads. Pass these options on each call, including
`net.resume(...)`; they are per-call controls, not serialized optimizer settings.
`restore_every=0` (default) disables the feature. Both options must be supplied
together when enabled, and they are restricted to RPROP. Rebuild the executable
to enable its advertised restoration capability; older executables still work
when the option is disabled.

`log_every` emits the objective and contributions evaluated before an update;
`eval_every` explicitly evaluates them again after an update. Records label the
phase and completed update. Every training call also evaluates and reports its final
total, including stopped calls. All scalars are unaveraged objective values. The
returned result contains reports, completed epochs, stop status, and progress metadata.
`verbose=2` prints term contributions; `verbose=1` prints totals; `verbose=0` still
prints a final total. An interval of zero disables that periodic activity.
`print_outputs=True` requires positive `eval_every` and prints full-schema output
snapshots at evaluation intervals. That explicit diagnostic may compute derivatives
outside the training cost's closure.

`diagnostics_every` requests summaries of the accumulated weight gradient, including
biases, before optimizer updates. Zero disables periodic diagnostic reports; a
positive interval also reports the last requested epoch. For example:

```python
result = net.train(cost=total, epochs=1000, diagnostics_every=100)
gradient = result['gradient_reports'][-1]
print(gradient['l2_norm'], gradient['max_abs'], gradient['nonfinite_count'])
print(gradient['layers'])
```

Each layer entry contains `layer`, `l2_norm`, `max_abs`, `nonfinite_count`, and
`first_nonfinite_index` (`None` when finite). Layers and indices are zero-based;
indices follow the state dictionary's column-major weight matrix followed by biases.
The report's `epoch` identifies the attempted step within this call, and `update`
counts successful updates preceding that step. `net.last_gradient_report` retains
the most recent summary, including one emitted before a numerical-error exception.
Norms and maxima are NaN when their gradient contains non-finite entries.

Finite checks run on every iteration, even when periodic reporting is disabled.
A non-finite objective or `dE/dY` raises an error with its cost index/label and
global sample index; a non-finite weight gradient identifies its layer and first
flattened index and forces a diagnostic report. Such an iteration is rejected
before the optimizer step. Already completed updates are retained. A failure during
post-update objective evaluation is reported separately and does not roll back
completed updates. The session remains usable after these numerical errors.

Cost kernels use CUB block reductions instead of writing per-sample losses.
On CUDA, CUB device reductions combine signed sums and error locations using reusable GPU
scratch space. Scalars accumulate in double precision; local cost arithmetic and
gradients remain float32. Unreported iterations skip scalar accumulation but retain
finite checks. All chunks finish without scalar host reads; small cost and gradient
summaries transfer together at one validation boundary before each optimizer step.
Gradient checks scan every layer each iteration, so their overhead must be measured
on the target GPU. The existing bias-gradient cuBLAS path is unchanged. CPU uses serial reductions
and its BLAS adapter for matrix operations; it does not use CUB or cuBLAS.

`save_every` sends periodic consistent snapshots to RAM as `net.checkpoint`; the
latest replaces the previous snapshot. The executable performs no disk writes.
`net.state_dict()` exports current weights, named optimizer buffers, optional
separate gradient history, and progress. `net.load_state_dict(state)` restores them.
Weight arrays contain each column-major weight matrix followed by its bias vector.
Snapshots are coherent at completed update boundaries. With checkpointing disabled,
normal training completion returns only metadata and scalars; it does not download
weights or optimizer buffers.

`net.save(path, include_history=True)` and `net.load(path)` perform Python NPZ disk
I/O without pickle. Omitting gradient history preserves optimizer steps/moments.
`Network(..., state=path_or_state_dict)` loads explicitly supplied existing state.
The state path must exist; legacy network `.dat` files are not imported.

Python initializes new networks using NumPy's `default_rng(seed)` and uploads
the resulting float32 arrays. Pass `Network(..., seed=42)` for repeatable initial
weights. Each layer uses uniform weights in `[-2/sqrt(fan_in), 2/sqrt(fan_in)]`
and uniform biases in `[-0.1, 0.1]`; no CUDA RNG is used. Explicit state replaces
these initial values. The wrapper supports older servers through a state upload;
the updated server requires the updated wrapper's initialization payload
(capability bit 16).

The wrapper rejects overflowing engine dimension products, negative Adam
variances or RPROP steps in checkpoints, and positive optimizer settings that
round to zero or infinity in float32. Zero checkpointed RPROP steps remain valid.
These checks protect Python calls; they do not validate other protocol clients.

`Session(..., cache_warning_threshold=64)` warns once after that many distinct
compiled program keys, and once per network after that many requested
schema/chunk/mode contexts. These are conservative request-count warnings,
not memory measurements or eviction. Reuse parameterized costs and chunk
sizes; closing a network releases its contexts, while ending the session releases
compiled programs. Forward-only evaluation omits backward buffers and tables;
buffers already grown for training keep their capacity until the network closes.

`session.stop()` can be called from another Python thread. Ctrl-C while waiting for
training also requests a stop. The executable finishes the common update, returns
consistent state, and stays alive. A stopped state records remaining epochs;
`net.resume(cost=the_same_cost, ...)` executes them. The cost must be reconstructed
with the same descriptions and data identities/versions. Save/load of `Data`
preserves that identity. A normal new `train(epochs=...)` requests that many additional
epochs; it does not implicitly consume a previous call's remaining budget.

## External build and validation

See [BUILDING.md](BUILDING.md) for macOS, Linux and Windows scripts, direct CMake
commands, system BLAS selection and server discovery. Use `CPDE_BACKEND=CPU`,
`CUDA` (default), or `BOTH`. CPU builds require C++17 and CMake; CUDA builds also
require the CUDA Toolkit, cuBLAS and NVRTC. Apple Silicon runs the CPU backend
natively with Accelerate; this release has no Metal backend.

Python runtime use never invokes the build toolchain. The protocol uses binary
stdin/stdout, with diagnostics on stderr. A separate server input thread accepts
stop requests while the execution thread runs training. The server advertises
`CPDE/3` with diagnostics, scalar graphs, expression costs, periodic RPROP
restoration and backend identification. The wrapper still accepts CPDE/1 and
CPDE/2 for their supported operations. `CPDE_ENABLE_NVRTC=OFF` affects CUDA only:
it disables CUDA expression costs while retaining polynomial costs and scalar
graphs; the CPU interpreter remains available.

For native tests, enable `CPDE_BUILD_ENGINE_TESTS` and
`CPDE_BUILD_REDUCTION_TESTS`, then run CTest. Set `CPDE_TEST_EXECUTABLE` and
`CPDE_TEST_REDUCTION_EXECUTABLE` to the built server and reduction probe before
running `python -m unittest discover -s tests -v`. Also set
`CPDE_TEST_CPU_EXECUTABLE` to a production CPU server for CPU selection tests.

Without a supplied test executable, the suite builds a separate **test-only host
probe** using `g++`, temporary source conversion and independent serial shims.
That instrumentation remains useful for scheduling checks, but production CPU
execution compiles shared sources directly with the adapters under `cpu/`.
Neither CPU path tests GPU concurrency or stream behavior. Native CUDA tests
exercise the GPU implementation.

See [validation](VALIDATION.md) for actual platform coverage, test counts,
reproduction commands and limitations. No comparative performance claim is made.
