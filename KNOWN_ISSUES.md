# Known issues in 0.2.0

These are existing, documented limits of the released implementation. The
release does not apply the deferred or rejected formula changes.

- **Adam second-moment overflow (F02).** A finite but sufficiently large float32
  gradient can overflow the squared-gradient accumulator. The update may be
  counted while the weight remains fixed, and the nonfinite state cannot be
  reloaded by the Python importer. Pre-update finite-gradient checks do not
  prevent this case. This remains unresolved.
- **Sigmoid tails (F04).** The retained derivative formulas can round a sigmoid
  factor to one and produce a zero derivative although the true derivative is
  representable. This affects higher derivatives and weight backpropagation too.
  The proposed formula replacement is not applied.
- **Tiny RPROP gradients (F05).** The retained product-based sign comparison can
  underflow for very small nonzero gradients and miss growth or reversal logic.
  The proposed sign-comparison replacement is not applied.
- **Reduced circle residual.** `A - B**2/C` suffers cancellation near convergence
  in float32 arithmetic. In the documented final runs, its total gradient differed
  from direct residual evaluation at `E=B/C` by 4.10% / 0.508% for vanilla/extended.
  The eigenproblem objective permits higher eigenmodes and does not guarantee
  ground-state selection. Division still requires nonzero `C`.
- **Geometry updates.** Geometry remains fixed during a `net.train` call. Moving
  domains require explicit geometry/data reconstruction between commands.
  Attached variations supply custom first derivatives; they do not differentiate
  mesh generation or implement a moving-mesh training loop.
- **CPU execution scope.** Matrix operations use Accelerate/system CBLAS when
  available, or portable serial routines. Derivative propagation, reductions and
  expression interpretation are serial. Float32 reduction order and BLAS choices
  can change final rounding. Apple GPU/Metal execution is not implemented.
- **Resource lifetime and protocol scope.** Compiled kernels and working storage
  are retained until their owning session/network closes; cache warnings are not
  eviction. Python dimension/state checks do not harden independent raw-protocol
  clients. The engine retains signed 32-bit indexing.

The removed legacy interface stays removed. Its approved repair patch is kept
only as a [historical patch](docs/patches/legacy-repairs-before-R08.APPROVED-NOT-APPLIED.patch)
against the pre-removal snapshot; it is not applied to this release.
