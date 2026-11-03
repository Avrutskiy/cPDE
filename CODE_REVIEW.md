# Release review — 3 November 2026

The 0.2.0 source includes the accepted wrapper and CUDA fixes: RPROP history
clearing, matrix ownership and view checks, shared immutable derivative metadata,
forward-only evaluation storage, skipped unnecessary backward work, preservation
of values under attached-gradient cancellation, device constant fills, wrapper
weight initialization, division-adjoint corrections, and removal of unused code.
The separate d4/d4p, d5/d5p and d6/d6p functions remain for clarity.

The wrapper validates dimension products and optimizer state, warns about cache
request growth, and refreshes kernel inspection after rebinding. The formerly
stale inspection mapping is fixed. Both server backends own the epoch loop. CPU compiles shared sources directly,
uses host-memory/BLAS adapters, and interprets scalar expressions; CUDA retains
its kernel launches and NVRTC implementation. Session selection verifies the
server backend. macOS/Linux/Windows build scripts are included.

The [known-issues document](KNOWN_ISSUES.md) records the retained Adam overflow,
sigmoid-tail and tiny-gradient RPROP limitations, circle cancellation, fixed
geometry, and resource/protocol limits. No deferred formula change or removed
legacy interface has been silently restored.

The [validation record](VALIDATION.md) distinguishes production CPU checks, real CUDA checks and
the separate serial host probe. The [example guide](README.md#fourier-fitting) describes
the delivered scripts and results. Historical trials and superseded findings
remain in the [development review](docs/history/CODE_REVIEW.md).
