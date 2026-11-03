# Development history before the 0.2.0 release

Historical working notes. Settings, intermediate results and findings below may
be superseded. See the root changelog and known-issues document for release status.

# Changelog

## Unreleased

- Rewrite all vanilla examples as standalone minimal formulations: construct
  Fourier values directly and remove unused boundary derivative loops, tangents,
  normals, and curvature from the PDE scripts. Retain only the derivatives needed
  by each PDE residual and adapt the independent validation tools.

- Apply the approved division-adjoint correction in CUDA graphs, NVRTC expression
  generation, and graph scheduling. Remove unused derivative macros, make the
  expression reduction-header dependency explicit, and clean unused core includes.
  Keep F04 pending, F05 rejected, and the approved legacy repairs as a retained
  patch without restoring the removed interface.

- Apply the selected CUDA audit fixes: clear RPROP history on explicit step
  replacement, correct matrix copy/view ownership, share derivative metadata,
  omit backward storage and work for value-only evaluation, and preserve values
  when attached variations cancel. Remove the dormant legacy interface while
  retaining the separate high-order derivative functions and recomputation.
- Initialize weights in Python with optional `Network(seed=...)`, upload them to
  the device, remove unused cuRAND/loss/event state, and fill constants on device.
  Add wrapper dimension/state checks, cache warnings, and fresh kernel inspection
  after rebinding. Formula changes and legacy repairs are separate proposals.

- Deliver vanilla/extended pairs for Fourier fitting, disk Poisson, and circle
  Schrodinger, with explicit derivative requests and matching geometry/settings.
  Keep attached variation as a single example and older shapes as references.
  Extended Poisson now includes residual gradients and curvature-corrected
  first/second wall derivatives. Validate both flavors and include them in CI.

- Expose periodic RPROP restoration through `restore_every` and `restore_step`.
  Reuse `reset_resile(R)` to floor small steps, clear only restored entries'
  gradient history, and schedule across calls/checkpoints by total updates.

- Add a unit-circle ground-state experiment with three 128-neuron hidden layers,
  the same boundary-inclusive triangulated quadrature and wall derivatives, and
  exact normalized Bessel-function comparisons.

- Add a convex asymmetric rounded-triangle example with 144 interior and 64
  boundary residual samples, triangulated vertex quadrature, and curvature-aware
  first/second tangential wall penalties.

- Add the 2D Schrodinger ellipse example with direct RPROP training from E=5,
  a 145-point regular interior grid, cell-area/arc-length quadratures, and
  residual plus 0.5-weighted Rx/Ry penalties. Earlier trials remain in reports.

- Add squared x/y/z derivatives of the sphere Schrodinger residual, each weighted
  by 0.5 in the same interior cost, with explicit third-derivative requests.

- Add a convex ellipsoid Schrodinger variant with semiaxes (1.1, 1, 0.9),
  analytic normals/Jacobians, direct training without pretraining, and a 0.1
  tangential boundary penalty. The earlier strongly deformed trial is retained
  in the validation reports.
- Pretrain the torus network for 5000 epochs against a normalized positive
  centreline profile that vanishes on the boundary, using value-only array
  matching, before the 25000-epoch eigenproblem phase.
- Start the torus at E=10 and train for 25000 RPROP epochs, keeping the first
  quotient update at epoch 1000 and subsequent updates every 100 epochs.
- Hold the torus's initial energy fixed for 1000 epochs before the first quotient
  update; retain 100-epoch updates thereafter and 13000 total RPROP epochs.
- Start the torus example at `E=5` and run 13000 RPROP epochs, resetting its
  initial steps once and updating E every 100 epochs.
- Halve the torus hole radius while retaining tube thickness (`R=1.325`,
  `r=0.65`) and start its energy updates at `E=0.5`.
- Set the Schrodinger example's normalization penalty to `10 * (mass - 1)**2`.
- Reset a RPROP step to `0.1` and clear its gradient sign history when the ±20
  weight bound rejects its update; prevent rollback of an unapplied step.
- Accept finite scalar quadrature weights and broadcast them over the supplied
  points, including geometry replacement; retain vector-weight support.
- Add a moving-boundary attachment demo: integrate over the network-defined
  ellipse {u>0} and attach the boundary-motion contribution to the gradient.
- Run the most expensive value-dependent context last and backpropagate its
  still-live final chunk after scalar evaluation, saving one network forward
  without extra activation storage. Select by an explicit work estimate.
- Integrate the torus objective with volume/surface quadrature and coefficient
  one on each term; remove the additional normalization and wall multipliers.
- Save the proposed shifted-memory forward-retention plan without changing the
  CUDA engine. Retention remains unimplemented.
- Add a 3D deformed-torus Schrodinger example with normalization, wall-value and
  normal-only tangential penalties. A Python loop updates E from the agreed
  quotient every 100 epochs while preserving Adam state.
- Schedule cost gradients per leaf: fixed multipliers use forward then backward;
  value-dependent multipliers use scalar evaluation followed by a fresh forward
  and backward. Accumulate all contributions before one optimizer update.
- Prune compiled costs to their own derivative dependency closure, including
  referenced frozen components, even when explicit input Data has extra rows.
- Use a random triangulated interior and uniform circle boundary for Poisson;
  group the two boundary terms in one multiline cost, separate from the PDE cost.
- Deliver two fitting cases on a 13x13 grid: explicit value/first-derivative
  expressions, and bulk matching using third-order requests with automatic
  lower-order dependencies.
- Add unit-RHS Poisson in a disk, with boundary value and tangential-derivative
  costs using the supplied unit tangent. Fitting and Poisson each use one train call.
- Expose `table(derivatives, variables=...)` for explicit derivative lists and
  dependency closure; reject maximum-order integers.
- Make examples standalone Python/NumPy scripts, with no cost/data helpers or
  small-network mode; print direct `net.eval` arrays after training.
- Retain the rejected Python-controlled drop attempt under
  `experiments/deferred_drop/`; remove it from the example launcher and CI.
- Accept indented multiline expressions in both parsers; preserve underscores
  for derivative notation and correct the README parameter spelling.
- Check equivalence of explicit and bulk first-derivative fitting, bulk matching
  against NumPy, and tangential boundary cost values and gradients.
- Record unresolved implementation findings in `CODE_REVIEW.md`.

## 0.2.0 — 2026-11-02

- Explicit quadrature with direct weights, segment/triangle/tetrahedron cells,
  centroid/vertex rules, optional basic Delaunay, normals, and tangents.
- Restricted scalar expressions compiled with NVRTC; the engine is not recompiled.
- Signed integrals, scalar cost graphs, `freeze`, and nested attached variations.
- Read-only scalar/gradient evaluation and explicit geometry/parameter replacement.
- GPU scalar graph evaluation and a two-pass integral/backpropagation path.
- CPDE/3 capability negotiation with CPDE/1 and CPDE/2 client compatibility.
- Moving drop-footprint variation checked against analytical and numerical derivatives.

Automatic moving-domain training and point-following kernels are not included yet.

## 0.1.0 — 2026-11-02

- Persistent CUDA server with Python Session, Data, Network, Cost, and CostSum APIs.
- Matching costs, quadratic residual expressions, supplied-vector contractions,
  derivative orders up to six, and padded chunk execution.
- RPROP and Adam continuation, state export/import, checkpoints, stop and resume.
- Fused CUB block reductions and signed device accumulation with reusable scratch.
- Gradient summaries and non-finite checks before optimizer updates.
- CUB include isolation from the retained derivative engine's `E` macro.
- MIT license, runnable examples, host-probe CI, and portable source packaging.

CUDA Release build and all 17 tests validated with CUDA 13.0.88, MSVC 19.44, and an
RTX 3090. No performance comparison has been measured. Quadrature and composable
integral graphs remain design work, not part of this release.
