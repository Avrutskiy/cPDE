# Historical code-review record

This is the development record before the 3 November 2026 release. Descriptions
of current behavior below refer to their respective development stages. The
root CODE_REVIEW.md and KNOWN_ISSUES.md describe the release. Raw local validation
artifacts referenced here are not included in the repository upload.

# Focused code review — 2 November 2026

Scope: Python API/transport, expression preparation, scalar cost composition,
quadrature bindings, CUDA expression generation, and the server paths connecting
cost gradients to the existing network engine. This is not an exhaustive audit
of the retained derivative kernels, GPU races, or performance. Findings below
were reproduced on the serial host probe. Their current status is recorded below.

## Samples-only release — 2 November 2026

The accepted analytical-energy scripts are now the main `schrodinger_circle.py`
and `schrodinger_circle_vanilla.py`, byte-identical to the completed 2000-epoch
GPU experiments. The previous periodic-energy scripts are preserved under
`examples/reference/`. The focused geometry, paired-objective and projected
gradient tests passed on RTX 3090 after promotion (3 tests, including both
flavors); the log is `output/examples/samples-release-tests.log`.

`python scripts/bundle_samples.py` creates `dist/cpde-0.2.0-samples.zip` with
the three pairs, attached variation, supplementary explicit fitting, usage
guide, license and SHA256 manifest. It excludes the library, CUDA sources,
experimental scripts, checkpoints and generated plots. Archive integrity,
Python syntax and per-file manifest hashes were verified. The earlier full
source archive and library artifacts are unchanged. The known cancellation
limit is documented in the samples guide.

## Example experiments and earlier delivery — 2 November 2026

Analytical energy elimination was tested for 2000 epochs in the separate
`schrodinger_circle_projection.py` and `_vanilla.py` experiments. Both use one
training call and the existing compound-cost graph, with no CUDA changes.
The same initial weights, training/validation grids and RPROP settings were
verified against the fixed-energy comparison. Vanilla ends at E=5.782953330,
0.04915% relative solution L2 and 0.004521 independent PDE RMS; extended ends
at E=5.781320283, 0.03085% relative solution L2 and 0.0003199 PDE RMS.
The regression test checks analytical minimization over E and agreement of the
reduced gradient with the explicit residual gradient at frozen E=B/C. Both
variants passed on CUDA (relative gradient differences about 1.3e-7 early in
training). A separate final-checkpoint audit detects cancellation: expanded
residual values differ from direct evaluation by 1.12e-6 / 1.53e-6, and total
gradients differ by 4.10% / 0.508% relative L2 (vanilla / extended). Therefore
the formula improves convergence in this experiment but needs a numerically
stable residual evaluation before treating it as a high-accuracy replacement.
All plots, metrics and the audit are in
`output/examples/rtx3090-circle-projection-2000/`. No training exceeded 2000 epochs.

Fixed-energy circle diagnostic: the separate `schrodinger_circle_fixed_energy`
and `_vanilla` scripts prescribe the first J0 zero squared from initialization,
remove the quotient cost/update loop, and use one training call. Both completed
2000- and 10000-epoch RTX 3090 runs with bit-identical starting weights to their
adaptive counterparts. All saved numerical arrays are finite. The GPU regression
test executes both scripts and rejects any call to change cost parameters.
At 2000 epochs, independent-grid relative solution errors are 0.2760% (vanilla)
and 0.03565% (extended); at 10000 they are 0.07486% and 0.01616%. The earlier
adaptive extended run had 5.1323% error at 2000 epochs and 0.04047% at 10000.
This supports energy adaptation as a major convergence bottleneck in this setup;
the prescribed-energy test does not demonstrate recovery of an unknown eigenvalue.
No core CUDA or API code changed. Results, progress and residual plots are linked
from `output/examples/rtx3090-circle-fixed-exact-energy-comparison/index.html`.

The main delivery now has three standalone pairs: `fit_function_vanilla.py` /
`fit_function.py`, `poisson_disk_vanilla.py` / `poisson_disk.py`, and
`schrodinger_circle_vanilla.py` / `schrodinger_circle.py`. Attached variation
remains a single script. Explicit first-derivative fitting and the older shapes
remain supplementary/reference examples, as requested. No core CUDA or API
code changed for this delivery.

Vanilla fitting propagates values only. Vanilla PDE examples propagate xx/yy
and their dependencies in the interior and values only on the boundary.
Extended fitting matches all ten rows through third order. Extended PDE costs
add 0.5*(Rx²+Ry²), with wall densities u²+0.1*u_s²+0.1*u_ss²; curved-wall
second derivatives include the normal/curvature term. Both circle versions
retain normalization weight 10, 305 interior and 64 boundary points, 10000
RPROP updates, and E updates first at 1000 then every 100. Settings and geometry
within each pair match. Diagnostic derivatives in the validation harness are
evaluated after training and do not alter vanilla training contexts.

Thirteen targeted CUDA tests passed, including numerical objective checks,
derivative-table sizes, identical geometry, and explicit/bulk gradient agreement.
All six full training runs plus the attached-variation demonstration completed
on the RTX 3090. Saved initial weights were verified bitwise identical within
each pair. Independent-grid relative value L2 errors were:

| Example | Epochs | Vanilla | Extended |
| --- | ---: | ---: | ---: |
| Fourier fitting | 1500 | 2.08426% | 0.0296410% |
| Disk Poisson | 3000 | 0.115334% | 0.0114076% |
| Circle Schrodinger | 10000 | 0.111698% | 0.0404724% |

Poisson independent PDE RMS was 0.00103906 / 0.000138020 (vanilla/extended).
Circle independent PDE RMS was 0.0211266 / 0.000466656, while E was
5.78199520 / 5.78062321. Thus the vanilla energy was closer to exact but its
pointwise residual was substantially worse; an energy estimate alone does not
validate the eigenfunction. These are paired single-run measurements, not
statistical guarantees. The extended circle reproduces the accepted checkpoint
metrics exactly. Full reports, plots, arrays, checkpoints, source snapshots,
and CUDA test log: `output/examples/rtx3090-delivery-pairs/`.

## Circle restored to 305 interior points, 10000 epochs — 2 November 2026

The example returns to the 21x21 clipped lattice (305 interior + 64 boundary
points) and trains for 10000 epochs. Energy updates remain at epoch 1000 and
every 100 thereafter. CUDA geometry/cost/schedule checks passed. The saved run
verifies the original initial weights and exact same points and triangulation
as the earlier 305-point, 5000-epoch experiment, with exactly 10000 updates.

Final E is 5.78062321 (0.044314% below exact), dense relative solution error
0.0404724% (previously 0.83383% at 5000), independent PDE RMS 0.000466656,
independent wall-value RMS 0.000317267, and dense mass 1.00033068. Final objective
after the last energy update is 0.000173210. Independent second-tangent RMS
remains 0.00602195; finite boundary and derivative errors remain despite the
improved value approximation. Artifacts, source, and test log:
`output/examples/rtx3090-circle305-first1000-then100-10000/`.

## Circle interior grid doubled to 609 points — 2 November 2026

A 29x29 clipped lattice now supplies 609 interior points (previously 305), with
the same 64 boundary points, for 673 residual samples. The first energy update
remains at epoch 1000, followed by updates every 100 through epoch 5000. The
CUDA geometry/cost/schedule test passed. Saved artifacts verify identical
initial weights, boundary points, and update schedule to the 305-point run.

Final E is 5.71889091, dense relative solution error 0.83928%, independent PDE
RMS 0.00131131, independent wall-value RMS 0.00754805, and dense mass 1.00014536.
Final objective after the last energy update is 0.000748903. Compared with
305 interior points, the PDE RMS improves from 0.00145847 but solution error
is essentially unchanged (previously 0.83383%). Independent residual-gradient
RMS improves from [0.0153603,0.0169356] to [0.00900934,0.0101299]. Artifacts,
source, and test log: `output/examples/rtx3090-circle609-first1000-then100-5000/`.

## Corrected circle schedule: first 1000, then every 100, 5000 total — 2 November 2026

The user clarified that 1000 specifies the first energy-update time, not the
recurring interval. The example now trains 1000 epochs initially, followed by
forty 100-epoch blocks, updating E after each block through epoch 5000. The
CUDA schedule/geometry/cost test passed. Saved reports verify the exact schedule
and identical initial weights, points, and triangulation to preceding runs.

With the corrected schedule, final E is 5.71952410 (1.1008% below exact), dense
relative solution error 0.83383%, independent PDE RMS 0.00145847, independent
wall-value RMS 0.00748252, and dense mass 1.00036707. Final objective after the
last energy update is 0.000933728. This substantially improves on the mistaken
every-1000 schedule's 7.3696% solution error. Artifacts, source, and CUDA test
log: `output/examples/rtx3090-circle305-first1000-then100-5000/`.

## Circle energy updates every 1000 epochs, 5000 total — 2 November 2026

The circle now uses five 1000-epoch blocks, updating E at 1000,2000,...,5000.
The CUDA schedule/geometry/cost test passed. The new run uses the same saved
initial weights, 305 interior points, 64 boundary points, and triangulation as
the other recent schedule experiments. Saved records confirm exactly 5000
updates and five quotient updates. RPROP restoration remains every 1000 steps.

Energy progresses 5, 5.04639709, 5.09135834, 5.13368517, 5.17327407, 5.21041267,
ending 9.9041% below the exact value. Dense relative solution error is 7.3696%,
independent PDE RMS 0.00974528, wall-value RMS 0.06713207, and dense mass
0.99965008. Final objective after the last energy update is 0.03131700.
This remains unconverged despite the longer budget. Artifacts, source, and test
log: `output/examples/rtx3090-circle305-every1000-5000/`.

## Circle energy updates every 500 epochs — 2 November 2026

The circle example now trains four 500-epoch blocks and updates E at epochs
500,1000,1500,2000. Initial E=5 and all other settings remain unchanged. The
CUDA schedule/geometry/cost test passed. Saved artifacts verify identical initial
weights, points, and triangulation to the preceding 305-interior-point runs.
The quotient sequence is 5, 5.04120035, 5.08671750, 5.12879060, 5.16853311.
After exactly 2000 updates, dense relative solution error is 7.9017%, independent
PDE RMS 0.01037191, wall-value RMS 0.07202599, and dense mass 0.99947501.
Final objective after the last quotient update is 0.03682520. This is a modest
improvement over every-100 updates (8.6105% solution error), but worse than
first-1000/then-every-100 (5.1323%). Artifacts and test log are in
`output/examples/rtx3090-circle305-every500-2000/`.

## Circle energy updates starting at epoch 100 — 2 November 2026

The circle now updates the Hamiltonian quotient after epochs 100,200,...,2000.
Each block trains 100 epochs; optimizer state is reset only for the first block.
The CUDA geometry/cost/schedule test passed. A fresh 2000-update run reused the
preceding experiment's exact initial weights, points, and triangle connectivity;
the saved report confirms all twenty energy updates at the requested epochs.
Periodic step restoration and all other settings remain unchanged.

The first quotient update lowered E from 5 to 3.63578738. Subsequent updates
raised it monotonically to 5.11220367 at epoch 2000, versus 5.38719342 when the
first update was delayed to epoch 1000. Dense relative solution error increased
from 5.1323% to 8.6105%; independent PDE RMS is 0.01118236, wall-value RMS
0.07858778, and dense mass 0.99932061. Final objective after the last energy
update is 0.04357040. Thus the earlier schedule did not improve this paired run.
Artifacts, source, and test log: `output/examples/rtx3090-circle305-first100-2000/`.

## Denser examples and 2000-update circle run — 2 November 2026

Both Fourier examples now use 27x27 training grids following the successful
Table I benchmark. Their other settings remain unchanged. The circle example
now uses a 21x21 lattice clipped to the disk: 305 interior points (previously
145), plus the unchanged 64 boundary points. All 369 points participate in the
triangulated residual quadrature. Its total training budget is now 2000 updates;
E starts at 5, is held for 1000 updates, then updated every 100. Periodic RPROP
restoration remains every 1000 updates with floor 1e-5. Plot labels derive point
counts from the saved arrays. Twelve targeted CUDA example tests passed.

The RTX 3090 run replayed exactly the same initial weights as the previous
145-interior-point circle experiment and stopped at exactly 2000 updates.
Energy is 5.38719342 (6.8473% below the exact 5.78318596); dense relative value
L2 error is 5.1323%, independent PDE RMS 0.00715242, independent wall-value RMS
0.0465042, and dense mass 1.00012248. Maximum sampled boundary residual is
0.01827057 in magnitude. The old-grid run had E=5.39100645 at update 2000,
so the denser grid did not materially improve energy convergence at this budget.
The final wall cost remains the dominant objective component. Objective reports
at the end of training precede the last quotient update; the final evaluated
objective after that update is 0.01677199. Artifacts, source, and CUDA test log:
`output/examples/rtx3090-circle305-2000/`. No additional training was performed.

## Direct Fourier approximation against paper Table I — 2 November 2026

The paper's Section II/Table I references the Section III-A setup: 729 training
points (27x27), 24025 test points (155x155), architecture [2,128,128,128,128,1],
sigmoid hidden layers, linear output, and 1000 full-batch RPROP updates with
initial step 2e-4. All ten rows through derivative order three are matched with
inverse target-variance weights. No periodic restoration applies to this short
run. The previously requested 13x13 example grid is a separate, sparser setup.

`scripts/benchmark_fourier_order3.py` ran three native CUDA initializations,
each paired between 27x27 and 13x13, on the RTX 3090. The fixed Fourier target
uses seed 2026; the exact random target/initialization from the paper is not
available, so this checks reported performance rather than literal replication.
All initial/final checkpoints, target arrays, predictions, and progress reports
are saved in `output/examples/rtx3090-fourier-order3-1000/`.

For 27x27, median normalized RMS percentages by order 0/1/2/3 were:
train [0.03382, 0.08034, 0.46005, 1.16270],
test [0.03124, 0.09243, 0.35422, 1.67936]. These average per-derivative RMS
within each order and use each set's target population standard deviation.
Paper Table I reports train [0.07,0.18,0.77,1.9], test [0.06,0.17,0.61,2.8].
The aggregate sqrt(sum of ten normalized MSEs), using training standard
deviations on both sets, has median 2.4902% train / 3.4798% test, versus the
paper's approximately 4% / 5.6%. Across three runs, train spans 2.2401-4.0747%
and test 3.0708-4.6426%. All three are consistent with the reported accuracy.

The paired 13x13 runs have median 5.9435% train / 25.5222% test aggregate error,
with median test third-order error 12.3999%. This shows materially worse
generalization for the sparse grid in this experiment. Independent reductions
of predicted arrays reproduce the server cost within 1.34e-8 relatively;
no final weights are at the cap. No core, optimizer, or example code changed.
The 27x27 runs still finish with 18658-22258 exactly zero RPROP steps despite
reaching the reference accuracy: zero-step counts alone do not establish a
training defect. Direct array fitting passes this check; this does not validate
the separate composed PDE-cost/gradient path or establish its remaining issue.

## Persistent circle residual location — 2 November 2026

The periodic-restoration run still peaks near (-0.995, 0.098). This boundary
node is included in the equation cost, with area weight 0.0077462. Independent
float64 chain rules reproduce the boundary maximum 0.00761333; finite differences
of value-only evaluation converge to 0.00761329. The residual is therefore real,
not just a CUDA derivative-output or plotting artifact. Ring maxima increase
from 0.001432 at radius 0.8 to 0.007613 at the wall.

A controlled second 25000-update run rotated only the original first-layer
input weights by 90 degrees, retaining the other initial weights, grid,
triangulation, cost, and periodic-restoration settings. The largest boundary
residual moved from angle 174.375 degrees (0.00761336) to 270 degrees
(0.00448190). A substantial left-side lobe also remains: this is not a pure
rotation of the final error. The experiment demonstrates sensitivity to the
initial representation/training path, but does not isolate the remaining
mesh, optimization, or floating-point effects. Restoring RPROP steps fixes
zero-step stagnation; it does not guarantee convergence of the integrated
soft penalties or minimize the maximum pointwise error.

The rotated run has independent PDE RMS 0.00056299, but energy 5.78612135 and
dense relative value error 0.038024% (both worse than the original run).
It is a diagnostic experiment, not a replacement example or a validated fix.
Artifacts and comparison plot are in
`output/examples/rtx3090-circle128-rotated90-25000/`; the independent ring and
finite-difference audit is beside the periodic-restoration checkpoint.
No training or derivative-kernel code was changed for this investigation.

## Periodic RPROP restoration — 2 November 2026

The original 25000-update circle checkpoint contained 33370 exactly zero steps
out of 33537, all with nonzero freshly evaluated gradients. Multiplicative RPROP
growth cannot revive zero steps. The retained `reset_resile(R)` / `Reset_Resile_k`
path now backs optional `Network.train(restore_every=1000, restore_step=1e-5)`.
Its existing strict floor rule is preserved: if step<R, assign R. Only restored
entries have both gradient-history buffers cleared, preventing rollback using
a newly enlarged step that was never applied. Larger/equal steps and weights
are not modified by restoration. The schedule runs after cumulative network
update multiples and survives split calls and saved update counters. Options
are per-call and must also be passed when resuming. Default behavior is unchanged.

The server advertises capability bit 8; an optional training-payload trailer
preserves older request layouts. Clients reject enabled restoration on older
servers. Eight local host-probe RPROP tests passed. All 63 applicable CUDA
regressions and the native reduction CTest passed, including threshold equality,
selective history clearing, first movement after restoration, checkpoint/cost
changes at update 1000, legacy matching costs, and split-vs-batched execution.
One pre-existing randomized quotient assertion was adjusted from fixed nine
decimal places to a float32-appropriate relative tolerance after a 6.7e-10
absolute discrepancy; the new restoration tests themselves passed unchanged.

Replaying the identical circle initial checkpoint for 25000 updates with this
option gave objective 6.10175e-5 (previously 2.74907e-4), independent PDE RMS
0.00073539 (previously 0.00152430), and maximum training-boundary residual
0.00761336 (previously 0.01583264). Final energy is 5.78420364, 0.017597% from
the exact ground-state energy. Dense relative value L2 error is 0.018315%,
normalization 1.00002761, and independent second-tangent RMS 0.00295635.
After the final scheduled restoration all steps are at least float32(1e-5);
33314 are exactly at that floor and none are zero. Artifacts and source are in
`output/examples/rtx3090-circle128-periodic-restore-25000/`.

The independent float64 NumPy audit of the old checkpoint reproduced its
boundary residual peak: 0.01583263 versus CUDA 0.01583264. The total-cost relative
difference was 0.02946%; third-derivative absolute differences reached 0.000431.
Complex-step checks along each layer's CUDA-gradient direction differed by
0.032%–3.584% relatively. Some much smaller random directional gradients had
larger relative discrepancies, including one sign difference at roughly 2e-6
absolute scale. Thus this audit supports the residual-spike diagnosis but does
not establish high relative precision for every nearly cancelled gradient.
Audit outputs are saved beside the old circle checkpoint; no derivative kernels
were changed as part of this restoration feature.

## Unit circle, three 128-neuron hidden layers — 2 November 2026

Added `examples/schrodinger_circle.py` with network [2,128,128,128,1] (33537
parameters), 145 regular interior and 64 boundary residual points. It retains
the triangulated vertex area weights, separate arc-length wall weights, bulk
R²+0.5*Rx²+0.5*Ry², curvature-aware first/second tangential wall penalties,
normalization coefficient 10, E=5, no pretraining and 25000 RPROP epochs.
The changed architecture uses a fresh initialization, saved for replay.
The actual CUDA test passed the declared architecture, geometry, quadrature,
curvature, boundary derivative expression and residual cost checks.

Final E=5.78449434 versus the exact unit-disk ground-state eigenvalue
j01²=5.78318596, a relative error of 0.022624%. Against the exact normalized
field J0(j01*r)/(sqrt(pi)*abs(J1(j01))), dense relative L2 error is 0.033690%
and maximum absolute value error 0.00043520. Only the arbitrary global sign is
aligned for these primary errors; amplitude is not fitted. A separate normalized
shape comparison gives 0.033610% relative L2. Dense normalization=1.00004640,
training normalization=1.00001304, independent random estimate=1.00455116.

Independent bulk PDE RMS=0.00152430, with Rx/Ry RMS=(0.0318175,0.0132709).
Independent wall value RMS=0.00023258, first tangent RMS=0.00069931, second
tangent RMS=0.00689889 and PDE RMS=0.00395581. Finite derivative errors remain
despite the close value/eigenvalue agreement. The exact field's squared integral
under the saved training mesh is 0.99996251; radial Gauss quadrature independently
verified the analytic normalization to 1e-14. Saved mesh connectivity is used
for reproduction, since Delaunay tie choices on a regular lattice can differ
between platforms. Logs, exact/error plots, checkpoints, source and numerical
arrays are in `output/examples/rtx3090-circle128-tangent2-25000/`.

## Rounded triangle, boundary residuals and second tangential derivatives — 2 November 2026

Added `examples/schrodinger_domain.py` with support function
h=1+0.085*cos(3*theta)+0.045*sin(2*theta), x/y scaling 1.25/0.9 and strictly
positive curvature radius h+h''. The residual grid contains 144 regular
interior points and 64 approximately equal arc-length boundary points.
Triangulated vertex quadrature assigns one third of each triangle's area to
each vertex. Boundary nodes participate in the residual integral with area
weights, while the separate wall cost retains arc-length weights.

The boundary now explicitly requests xx,xy,yy and penalizes
u²+0.1*u_s²+0.1*u_ss², where u_ss=T^T H(u) T-curvature*(grad(u) dot N).
CUDA tests passed the mesh weights, curvature finite differences, polynomial
restriction differentiated along the curve, and assembled wall cost. The original
network initialization, E=5 and 25000-epoch RPROP schedule were retained.

Final E=5.69770038 versus a refined finite-element estimate 5.70346789.
Objective=0.00201593, training normalization=0.99988893 and dense-grid
normalization=1.00089233. Independent bulk PDE RMS=0.00240641; independent
Rx/Ry RMS=(0.0243350,0.0475300). At 1024 independent wall positions, value RMS
is 0.00142605, first tangent RMS 0.00546660, second tangent RMS 0.08685215 and
PDE RMS 0.00307457. The second tangent remains appreciably worse between
training points (its 64-point training RMS is 0.04476924).
Artifacts and source are saved under `output/examples/rtx3090-domain64-tangent2-25000/`.

## Regular 145-point ellipse with Rx/Ry penalties — 2 November 2026

Replaced the 4096 random interior points with a regular 15x15 reference lattice
clipped strictly inside the ellipse, leaving 145 points. Scalar weights equal
dx*dy (rectangle rule, without area/N renormalization); cut boundary cells are
approximated. Restored 0.5*Rx²+0.5*Ry² in the same interior cost, explicitly
requesting xxx, xyy, xxy and yyy. All other training settings, boundary samples,
E=5 and the original initial checkpoint were retained. CUDA regression checks
passed the lattice, scalar weights, integrand and derivative requests.

After 25000 epochs, E=5.95702899 and objective=8.49605e-5. Training normalization
is 0.99999818, independent random-sample normalization 1.00462313 and dense-grid
normalization 0.99984996. The normalized field differs from the finite-element
ground-state reference by 0.06727% in dense-sample relative L2.

The sparse sampling nevertheless leaves substantial local derivative error:
training PDE RMS is 0.00214825 versus 0.04803070 on independent points.
Training (Rx,Ry) RMS is (0.00451438,0.00384404), versus (0.16088656,1.47191368)
independently. The largest sampled absolute residual is 1.32031 at
(0.01917,-0.89851), near the lower boundary beyond the last interior lattice row.
Boundary value RMS is 0.00071767 and tangent RMS 0.00514896. Accurate field shape
and energy therefore coexist with under-resolved local PDE derivatives.
Saved source, arrays, checkpoints, plots and comparison data are in
`output/examples/rtx3090-ellipse-grid145-rgrad-25000/`.

## Two-dimensional ground-state restart at E=5 — 2 November 2026

Restarted the residual-only 4096-point ellipse experiment with initial E=5;
the initial checkpoint arrays are identical to the previous E=10 run. After
25000 RPROP epochs, E=5.95284744 and objective=3.9990e-5. Training normalization
is 0.99999693, independent random-sample normalization 1.02338659, and dense-grid
normalization 1.01851312. PDE RMS is 0.0034096 on training samples and 0.0041500
on independent samples. Boundary value RMS is 0.00020248 and tangent RMS 0.00238225.

An independent linear-triangle finite-element solve with consistent mass and
zero boundary nodal values gives lowest eigenvalues 5.95578816, 5.95407789,
5.95347896 on 2977, 8109 and 20505 vertices. The solver's unit-disk check gives
5.78576474 on the coarse mesh versus the exact Bessel-zero value 5.78318596.
The network energy differs from the finest ellipse reference by 0.01061%.
After normalizing both fields, their dense-sample relative L2 difference is
0.01185%. The network has a positive interior lobe; three tiny negative samples
lie on the ellipse boundary to floating-point precision. These checks support
ground-state identification, while normalization retains sampling error.
Results, reference source/convergence, and field comparison are saved under
`output/examples/rtx3090-ellipse-e5-25000/`.

## Two-dimensional ellipse, residual only — 2 November 2026

Added `examples/schrodinger_ellipse.py` with semiaxes (1.1,0.9), network
[2,64,64,64,64,64,1], 4096 area samples and 2048 weighted boundary samples.
At the user's correction, the preliminary derivative-penalty run was stopped
and Rx/Ry terms removed. The final interior schema requests only xx and yy.
The final objective contains integral R², 10*(integral u²-1)², and boundary
integral u²+0.1*(grad(u) dot T)². It has no pretraining and uses initial E=10,
25000 RPROP epochs, and quotient updates after 1000 epochs and every 100 thereafter.

The RTX 3090 regression passed the ellipse geometry, area/arc-length weights,
normals/tangents, independent integrand evaluation, quotient and training-stage
checks. Full training completed with objective 0.02871397 and E=13.710193.
Normalization was 0.998530 on the training quadrature and 0.999049 on independent
samples. PDE RMS was 0.052086 on training samples and 0.065146 on independent
samples; boundary value RMS was 0.051615 and tangent RMS 0.074263.
The plotted field has two lobes of opposite sign, resembling an excited mode;
this objective does not select the ground state. Wall error remains appreciable.
These results do not isolate dimension as a cause: geometry, initialization,
mode and residual-derivative penalties differ from the last 3D experiment.
Saved source, numerical arrays, checkpoints and full-domain figures are in
`output/examples/rtx3090-ellipse-direct-25000/`.

## Convex sphere with residual derivatives — 2 November 2026

Added 0.5 times each squared spatial derivative of R=-Delta(u)-E*u to the
same interior cost, with explicit third-derivative requests. The Hamiltonian
quotient update was retained; it does not minimize this augmented objective
with respect to E at fixed network weights. CUDA checks verified the combined
integrand against independently assembled output-array derivatives.

The 25000-epoch RTX 3090 run finished with E=3.692590, objective 1.646737,
normalization 0.826408 (independent estimate 0.833175), and boundary value RMS
0.322638. PDE RMS was 0.012295 on training samples and 0.014229 on independent
samples. Independent residual-derivative RMS values were (0.082566, 0.119406,
0.107858). Small PDE residuals therefore did not establish an infinite-wall
eigenfunction: the wall and normalization errors were substantial.
Artifacts and source are in `output/examples/rtx3090-sphere-residual-grad-25000/`.
The subsequent requested 2D experiment removes Rx and Ry penalties; the 3D
example retains the requested derivative-penalty variant.

## Convex sphere, direct training — 2 November 2026

Revised `examples/schrodinger_sphere.py` to the ellipsoid
(x/1.1)²+y²+(z/0.9)²<1. Its positive-definite implicit Hessian guarantees
strict convexity. Removed profile pretraining and reduced only the tangential
boundary coefficient from 1 to 0.1; the wall-value coefficient remains 1 and
normalization coefficient remains 10. The same original initial weights,
architecture, sample counts, E=10 and 25000-epoch RPROP schedule were retained.
The first quotient update occurs after 1000 epochs, then every 100.

The RTX 3090 regression passed geometry, quadrature Jacobian, normal, direct
training schedule, quotient and independent boundary-cost checks. The complete
run finished with objective 0.00329362, E=10.03862, training normalization
0.999811, training PDE RMS 0.024342 and independent PDE RMS 0.035652.
Boundary value RMS was 0.003815 and tangential derivative RMS 0.022819.
The independent 4096-point normalization estimate was 1.037162, so the near-unit
training integral does not establish equally accurate continuum normalization.
Several settings changed together; this comparison does not isolate the effect
of removing pretraining or lowering the tangential penalty.

Energy near 10 is consistent with the unit ball's lowest exact Dirichlet
eigenvalue pi² and the inverse-square dependence on domain scale. Initial E
does not constrain the final quotient. This finite-penalty numerical run is not
an independent proof of the exact ellipsoid eigenvalue or mode selection.
Artifacts and a copy of the example source are saved in
`output/examples/rtx3090-convex-sphere-direct-25000/`.

## Strongly deformed sphere — 2 November 2026

Added `examples/schrodinger_sphere.py`, retaining the torus example. The reference
unit ball is mapped by x=a(Y)X+s(Y), y=Y, z=b(Y)Z, with
a=1.25+0.55*sin(pi*Y), b=0.85+0.35*cos(pi*Y), s=0.45*(1-Y²).
The x scale spans 0.7–1.8, the z scale 0.5–1.2, and the centreline bends by up
to 0.45. Positive scales keep the map invertible. Volume and surface quadratures
include their respective Jacobians; normals come from the implicit boundary.

The RTX 3090 geometry/regression test passed: implicit boundary, finite-difference
normals and surface Jacobians, volume/divergence agreement, normalized pretraining
profile, optimizer stage transitions, and energy quotient. The full run used the
same original initial weights as the torus trials, with 5000 profile-matching
epochs followed by 25000 RPROP PDE epochs. Initial E=10 was held for the first
1000 PDE epochs, then updated every 100. The normalization penalty remains 10.

Final profile-matching cost was 0.00018550. The PDE objective fell from 591.3403
to 0.0232974, with E=10.60561 and training integral u²=0.998737. Boundary value
RMS was 0.004819 and tangential derivative RMS 0.020006. PDE RMS was 0.057092 on
training samples versus 0.229302 on 4096 independent samples; their normalization
estimate was 1.040269. This sampling gap and finite wall penalty mean the run
does not establish an accurate continuum eigenpair, despite the stable objective.
Plots, logs, checkpoints, and numerical arrays are preserved in
`output/examples/rtx3090-deformed-sphere-pretrained-30000/`.

## Full CUDA example validation — 2 November 2026

Built the current server with CUDA 13.0 / MSVC for SM 86 and ran it on the
remote Windows RTX 3090. All 51 applicable Python regression tests and the
native reduction CTest passed. The scheduling trace test requires the separately
instrumented host probe and was excluded from this native invocation.

Full example training exposed failures not covered by those short regressions:

| Example | Updates | Observed result |
| --- | ---: | --- |
| Explicit fitting | 362 / 1500 | Next iteration rejected a non-finite weight gradient. |
| Bulk derivative fitting | 986 / 1500 | Next iteration rejected a non-finite objective; some third derivatives are non-finite. |
| Poisson disk | 3000 | Reported loss reached 2.17e-5 around update 800, then destabilized; final PDE RMS = 1 and relative solution L2 error ≈ 1. |
| Schrodinger torus | 3000 | Integral u² = 2.5056e-7 instead of 1; output collapsed toward zero. E ≈ 1.80365 is not a validated eigenvalue. |
| Attached variation | No training | Maximum absolute error against the analytic shape gradient ≈ 8.01e-7. |

Both fitting failures reproduced from their saved initial checkpoints with
unchanged example training settings. Logs, checkpoints, numerical arrays,
progress, mesh/field figures, and an HTML report are saved under
`output/examples/rtx3090-20261102/`. The validation harness is
`scripts/validate_examples.py`. Its `completed` status means execution finished;
the residual and normalization metrics determine scientific success.

### Resolved P1: RPROP rolled back updates rejected by the weight bound

Location: `cuda/derivative_kernels_v4.cu`, `Resile_k`.

Same-sign gradients multiply the step by 1.2 without an upper limit. The normal
update is skipped when its proposed weight exceeds ±20, but step growth continues.
On a gradient sign reversal the rollback adds the current step directly to the
weight, without the bound check or tracking whether that previous step was
actually applied. This can move weights far outside the intended range.

The failed explicit fit has a maximum absolute first-layer weight of 1.35e21;
the bulk fit reaches 2.17e17 in a hidden layer. The completed Poisson run reaches
1.74e28. These are observed checkpoint values, not merely possible overflow
scenarios. The original CUDA kernels were left unchanged during the initial validation.
The agreed correction now resets a rejected step to `0.1` and clears that weight's
gradient history. Thus the rejected update cannot be rolled back, and an inward
update starts at `0.1`. It retains the existing accepted-step rollback and does not
add a new kernel parameter or per-weight buffer. Old failed checkpoints already
contain corrupted weights and must not be used to assess this fix.
`tests/test_rprop.py` covers both bounds, repeated rejections, checkpoint
restoration, accepted-step rollback, and zero gradients. The Adam torus normalization failure is a
separate issue; this RPROP finding does not explain it.

The correction passed all 55 applicable CUDA regression tests on the RTX 3090
and the native reduction test. The four new RPROP tests also pass on the host
probe; the rejection tests produce three failing assertions/subtests against
the preserved original CUDA executable. Replaying the affected examples from
their original initial checkpoints now completes without non-finite failures:

| Example | Updates | Final cost | Independent-grid value relative L2 error |
| --- | ---: | ---: | ---: |
| Explicit fitting | 1500 | 0.00760206 | 2.02% |
| Bulk derivative fitting | 1500 | 0.0963278 | 0.188% |
| Poisson disk | 3000 | 3.07260e-6 | 0.00693% |

Poisson's independent-grid PDE RMS is 9.857e-4. All final weights in these
replays remain within ±20. Fitting value accuracy does not imply equal accuracy
for every derivative: bulk third-derivative RMSE divided by target standard
deviation remains approximately 7.3–10.7% on the independent grid. Full metrics,
plots, progress and checkpoints are in `output/examples/rtx3090-rprop-reset-01/`.
The Adam torus was not rerun by this RPROP-only correction.

### Independent audit follow-up: the torus objective rewards amplitude collapse

The RPROP finding is not a complete explanation of the example failures.
`scripts/audit_example_gradients.py` compares all four initial training
checkpoints against a separate NumPy implementation of the spatial chain rules
through third order. Parameter-gradient directions are differentiated with
complex steps in that reference, independently of the CUDA engine and expression
compiler. The checks use the actual full architectures and point sets.

At the initial checkpoints, maximum absolute forward disagreement is below
2.85e-7. The worst relative disagreement among the tested parameter-gradient
directions is 1.993e-6. This tests one random direction and the gradient direction
in each layer, not every parameter separately. Combined-cost gradients also
agree with separately evaluated leaf gradients and their graph multipliers to
float32 precision.

`scripts/audit_optimizer_step.py` then compares evaluation with one training
iteration. The parameter gradients returned by evaluation and used by training
are identical in these four checks. The first Adam weight updates agree with
the mathematical update within 5.92e-8 absolute. These results do not validate
all later states, every derivative order, or long-run optimization.

There is a separate, directly reproduced amplitude problem in the torus cost.
At fixed E and fixed network shape, replacing u by a*u gives

```
J(a) = a²*A + (a²*M - 1)²
     = 1 + a²*(A - 2*M) + a⁴*M²,
A = equation + walls,  M = integral u².
```

Initially A/M ≈ 103.04, so A - 2*M is positive and this whole amplitude
direction is minimized at zero. Direct CUDA evaluations confirm:

| Output scale a | Total cost | Integral u² |
| ---: | ---: | ---: |
| 0 | 1.000000 | 0 |
| 0.25 | 1.392379 | 0.003883 |
| 0.5 | 2.569696 | 0.015533 |
| 1 | 7.281681 | 0.062132 |
| 2 | 26.173048 | 0.248526 |

At the final torus checkpoint, A/M ≈ 11.04, so the same amplitude preference
persists. This proves a defect in the suitability of this finite-penalty
formulation/initialization for the requested normalized eigenfunction; it does
not prove zero is the global minimum over all possible shapes. Quadrature
weights approximate the integrals but do not by themselves make the PDE,
boundary and normalization penalties balanced. Changing shape could lower the
residual, but the optimizer can instead reduce amplitude, and all these gradients
vanish at u = 0. Normalization enforcement and the residual's scaling require
review before interpreting a small residual as an eigenfunction.

The collapsed final checkpoint is also more sensitive to float32 cancellation:
the largest checked directional-gradient absolute discrepancy is 1.56e-7, with
about 0.68% relative disagreement in the output-layer gradient direction. One
much smaller projection differs by 6.05% (2.41e-9 absolute). Do not extend the
initial-checkpoint precision claim to this state or treat that discrepancy as a
separately localized implementation bug without further tests.

Raw records: `output/examples/rtx3090-20261102/gradient-audit/audit.json` and
`optimizer.json`. The original CUDA kernels and example objectives were not
changed by these diagnostics.

## Findings

### Torus normalization weight 10 — GPU replay

The user requested `equation + 10 * (mass - 1)**2 + walls`. Only that
normalization multiplier was changed; the proposed division by mass was not
adopted. The two torus regression tests passed on the RTX 3090. A full 3000-update
Adam replay from the original initial weights produced integral u² = 0.834257
(independent samples: 0.828850), E = 0.958335, held-out PDE RMS = 0.016819,
wall-value RMS = 0.203532, and tangential-gradient RMS = 0.043996.

The zero-amplitude collapse is avoided in this run, but unit normalization and
the hard-wall condition remain unsatisfied. E is therefore not a validated
hard-wall eigenvalue. Final cost after the last E update is 2.611738; the last
training report precedes that E update and is 2.614820. Updated plots, histories,
arrays and checkpoints are in `output/examples/rtx3090-torus-normalization-10/`.

### Smaller torus and initial E = 0.5 — GPU replay

Reduced the major radius from 2 to 1.325 with tube radius 0.65 unchanged:
the reference hole radius is now 0.675, half its former 1.35. Initial E is 0.5;
the normalization multiplier remains 10. Both torus checks passed. The same
initial network weights completed 3000 Adam updates, yielding E = 1.442066,
integral u² = 0.847127 (held-out: 0.847938), held-out PDE RMS = 0.016239,
wall-value RMS = 0.245034, and tangential-gradient RMS = 0.062246.
Final cost after the E update is 2.511307. Normalization and zero wall values
remain unsatisfied, so this is not yet a validated hard-wall eigenfunction.
Artifacts: `output/examples/rtx3090-smaller-torus-e05/`.

### Longer torus run with RPROP — 13000 total updates

Continued the smaller-torus checkpoint for 10000 additional RPROP updates after
its 3000 Adam updates. Preserved its weights and E = 1.442066, initialized RPROP
steps once to 0.0002, then retained RPROP history between 100-update E blocks.
The geometry, network, quadratures and cost weights were unchanged.

Final E = 11.207425; integral u² = 0.992709 (held-out: 1.016779); wall-value RMS
= 0.049723; tangential-gradient RMS = 0.017042. Total cost fell from 2.511307 to
0.131633. Held-out PDE RMS increased from 0.016239 to 0.057513, so improvement
in the total objective does not mean improvement in every equation metric.
The final checkpoint is also the lowest post-E-update cost observed in this
continuation. Cost is still decreasing and the wall/PDE errors remain nonzero;
no converged hard-wall eigenvalue is claimed.

The separate runner `scripts/continue_torus.py` preserves the original example
and records combined Adam/RPROP progress, final fields and best/latest states.
Artifacts: `output/examples/rtx3090-smaller-torus-rprop-13000/`.

### Fresh RPROP run with initial E = 5

The standalone example now starts at E = 5 and runs 13000 RPROP updates,
initializing steps once to 0.0002 and retaining them across 100-update quotient
blocks. Reused the original untrained weights for this trial. Unlike the previous
mixed Adam/RPROP run, all updates here use RPROP, so the runs do not isolate the
effect of changing only the initial energy. Both torus CUDA checks passed.

Final cost = 0.0590592, E = 11.995858, integral u² = 0.995101 (held-out:
1.018252), wall-value RMS = 0.0252274, and tangential-gradient RMS = 0.0149984.
Training PDE RMS = 0.049001 versus 0.089927 on independent samples. Residual
and wall errors remain; this is not a fully validated eigenpair. Saved the
complete plots, histories, arrays and checkpoints under
`output/examples/rtx3090-torus-e5-rprop-13000/`.

### First energy update after 1000 epochs

Kept E=5 fixed for the first 1000 RPROP epochs, followed by 120 blocks of 100
epochs, retaining 13000 total updates and the same initial network weights.
Both torus CUDA checks passed. Saved history confirms the first quotient update
at epoch 1000 (E=5.140853), followed by updates at 1100, 1200, etc.

Final E = 11.921190, integral u² = 0.994165 (held-out: 1.009138), total cost =
0.0863258, wall-value RMS = 0.027563, and held-out PDE RMS = 0.094087. This run
avoids the initial energy drop but does not improve final cost over the previous
100-epoch-first-update trial (0.0590592). The energy plot now displays piecewise
constant values, matching how E is held during each training block.
Artifacts: `output/examples/rtx3090-torus-e5-first1000/`.

### Initial E = 10, 25000 total RPROP epochs

Replayed from the same original initial weights, with initial E=10, its first
update after 1000 epochs, and 240 subsequent 100-epoch blocks. Both torus CUDA
checks passed; the saved history verifies all 25000 updates and energy timing.
Final cost = 0.102633, E = 13.868175, and integral u² = 0.994315 (held-out:
1.009566). Wall-value RMS = 0.0179332 and tangential-gradient RMS = 0.0310340.
Training PDE RMS = 0.069457 versus 0.290695 on independent samples. Cost and E
plateaued late in training; the field is concentrated on one side of the torus.
The training/independent residual gap remains substantial, so this is not a
validated eigenfunction despite near-unit normalization and a stable quotient.
Artifacts: `output/examples/rtx3090-torus-e10-25000/`.

### Value-only profile pretraining before the eigenproblem

Added one 5000-epoch RPROP array-matching call on the original bulk and boundary
points. The target is C*(1-rho²/r²) in the tube and zero at the wall, with C
chosen to give unit squared volume integral. It peaks on the reference tube
centreline and remains zero on the deformed implicit boundary. The derivative
list is explicitly `['']`; volume and surface quadrature weights are passed to
the array-matching kernel. This fits the network output without a trial-function
multiplier or a second network.

Fitted weights carry into the existing 25000-epoch PDE phase, but RPROP history
resets when changing objectives. E starts at 10 after pretraining and updates
after 1000 PDE epochs, then every 100. Both CUDA torus tests passed, including
target geometry/normalization and the phase reset. Saved history verifies
5000+25000=30000 updates, with the first energy update at overall update 6000.

Pretraining matching cost fell from 1.515519 to 0.000350222; unweighted value RMSE
over its bulk/boundary samples is 0.0040736. The PDE cost immediately after this
value-only fit is nevertheless 1474.93: a small value error does not certify
accurate spatial derivatives. The final eigenproblem cost is 0.458628 (worse
than 0.102633 without pretraining), E=11.228044, and integral u²=0.973445
(held-out: 0.999065). Training PDE RMS=0.151328, held-out PDE RMS=1.003028,
wall-value RMS=0.0520231, and tangential-gradient RMS=0.0493413.
This experiment does not demonstrate a converged eigenfunction or improved
PDE accuracy. Its pretraining and PDE histories are plotted separately, with
pretrained and final checkpoints retained for further investigation.
Artifacts: `output/examples/rtx3090-torus-pretrained-30000/`.

### Resolved P1: compiled costs propagated unused derivative rows

The compiled-expression registration path in `cuda/cpde_server.cu` now selects
only referenced derivatives and their lower dependencies. Frozen components
remain included for their values; constant/field-only expressions retain the
value row. Public component indices are remapped into this cost's own schema.

The regression supplies a `[1,2,1]` network with input derivatives through order
six, including a finite but huge unused sixth derivative (`3e38`). Previously,
`u**2` propagated it and failed with nonfinite weight gradients. Compiled and
legacy value-only costs now agree with finite gradients. A second regression
checks permuted public rows, pruned closures, frozen second derivatives, and
chunked execution against independently prepared coordinate inputs.

### Open P2: kernel inspection can return a stale component mapping

Locations: `cpde/cost.py:167-172` and `cpde/cost.py:194-200`.

`_sync()` replaces the server-side cost but does not invalidate the cached source
and compile log. `inspect()` neither synchronizes the binding nor refreshes an
already populated source cache.

Reproduced sequence: create an empty 1D quadrature and an expression `u_x`, call
`inspect()`, replace the quadrature with one point, then evaluate and inspect
again. Evaluation uses the new program, but inspection still shows derivative
matrix index 0. Calling `compile()` refreshes the cache and inspection then shows
the actual index 1. This can mislead debugging of generated cost gradients during
input/geometry changes. Numeric execution used the correct program in this check.

Recommended correction: synchronize before inspection and invalidate cached
source/log whenever the underlying binding is replaced, or key the cache by the
actual compiled program identity. Add empty-to-nonempty inspection coverage.

## Architecture assessment

The intended public keyword is `freeze`; the current API spelling is correct.
No compatibility alias or rename is required.

The core formula path already follows the intended division of responsibility:

1. `cpde/_expressions.py` parses a restricted expression tree into scalar IR and
   identifies network values and derivative matrices. It preserves the underscore
   namespace for derivatives. Frozen components contribute values but no local
   partial derivative.
2. `cpde/cost.py` builds integral leaves, scalar relations such as `mass/area`,
   and attached variations. Numeric fields are bound separately from expressions.
3. `cuda/expression_program.hpp` generates a specialized CUDA density/gradient
   kernel. It reverse-differentiates the scalar expression, sums repeated
   component adjoints, and writes into the engine's output-gradient matrices.
4. `cuda/graph_schedule.hpp` identifies leaf multipliers known before evaluation.
   Those costs execute forward then backward in sequence. Other costs first
   supply their scalar values; `cuda/cost_graph.cuh` evaluates the full graph and
   multipliers on the GPU. The server schedules the most expensive eligible
   forward chunk last and consumes it directly in backward, before another
   context can overwrite it. Remaining value-dependent chunks recompute forward.
   The ordering uses a work estimate, not GPU timings. Weight gradients accumulate
   across both stages before one update.
   Attached variations participate in gradient scheduling without changing their
   base objective values. Contexts share working storage, not saved activations.
5. `cuda/cpde_server.cu::train` owns the epoch loop, forward propagation,
   backpropagation, and RPROP/Adam updates. Python's receive loop only consumes
   reporting events; it is not an epoch-control loop. Expression programs are
   cached; the network engine is not regenerated for each formula or epoch.

The implementation is therefore a per-leaf compiled expression kernel plus a
precompiled GPU scalar-graph evaluator, not one monolithic generated kernel for
all integrals. This distinction does not require Python-side gradient arithmetic
or training orchestration.

## Input-set update gap

`Data.set_samples()` can upload same-shape replacements between commands.
`Quadrature.replace()` can replace geometry and rebuild bindings before the next
command. Neither provides updates during one ongoing `net.train(...)` call.
That is the deferred capability needed for the intended drop example. The
superseded Python contour/optimizer loop is archived outside the active examples;
this review does not prescribe a new physical model or extra network.

## Example delivery and validation

The current examples are standalone scripts without helper functions or a
small-network mode. Derivative requests are visible and `cpde.table(requests,
variables=...)` includes their dependencies. Array generation is ordinary
Python/NumPy in each script; cost and train calls are direct.

- Explicit Fourier fitting: 13x13 grid, four 128-neuron hidden layers, requests
  x and y, and writes the value/first-derivative matching expression directly.
- Bulk Fourier fitting: the same function and architecture, requests only xxx,
  xxy, xyy and yyy, and matches target arrays for the resulting ten rows.
- Poisson: direct output u on the unit disk. The interior explicitly requests xx
  and yy; the boundary requests x and y and supplies tangent T. The objective
  adds an interior PDE cost and one multiline cost combining boundary values
  and tangential derivatives. Random interior vertices and uniform circle
  vertices are triangulated; triangle areas weight centroid samples.

- Schrodinger: one real network in a 3D deformed solid torus, with a global
  normalization penalty and a boundary cost containing values and the squared
  cross product of the gradient with N. This enforces both tangential derivatives
  from the normal alone. The derivative requests are explicit in the script.

Fitting and Poisson use one server-side training call. The user explicitly permits
Python block orchestration for Schrodinger: train 100 epochs, update E from
integral(-u*Delta(u))/integral(u**2), then continue the same optimizer state. Each
script prints arrays from net.eval for inspection. Separate error calculations
and data helpers are confined to regression tests, not example usage.

The current suite checks automatic derivative dependencies and aliases, inline
data generation against independent analytical fixtures, explicit/bulk fitting
agreement, array matching gradients, and boundary tangent geometry and costs.
CI runs fitting/Poisson for one test epoch and Schrodinger for two one-epoch
blocks at their declared network sizes using a separate test harness. These checks use the serial host probe and do not establish
convergence, real NVRTC compilation, or GPU behavior. The host
suite is exercised with SciPy installed. A test-only instrumented engine verifies actual
forward counts for sums, ratios, mixed objectives and attached variations, as
well as accumulated gradients. The schedule tests cover chunked and unchunked
execution, actual context order and selection of the harder derivative context
regardless of expression operand order. The inspection-cache finding remains open.

The torus checks verify the implicit boundary, outward normals by finite
differences, surface weights against parametric tangents, and bulk volume against
the surface divergence identity. They verify that the quotient minimizes the
fixed-network discrete residual, the tangential penalty matches an independent
projection, and energy updates preserve Adam history. These short tests do not
establish a converged eigenpair or enforcement accuracy of the finite wall penalty.
The shifted-memory retention proposal is saved in IMPLEMENTATION_PLAN.md; it is
not implemented and required no changes to the original CUDA engine.

The attachment demonstration now integrates over the network-defined ellipse
u=a*x**2+b*y**2+c>0. It compares fixed-geometry gradients against an attached
boundary-motion gradient for integral(1+0.5*|grad(u)|**2). Finite differences of
the exact integral over the changing ellipse verify the additional contribution
and its sign, while the reported integral value stays unchanged. This is a
single-shape derivative demonstration, not a moving-mesh training implementation.
