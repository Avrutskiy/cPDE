# Changelog

## 0.2.0 — 2026-11-03

- Persistent CPU and CUDA execution with Python data, network, evaluation, training and
  checkpoint interfaces. Python generates new weights and uploads them; optional
  `Network(seed=...)` makes initialization repeatable.
- Explicit session backend selection, native Apple Silicon CPU execution with
  Accelerate, optional system CBLAS or portable serial matrix operations, and
  macOS/Linux/Windows compile scripts. Production CPU builds require no CUDA.
- Explicit derivative requests with automatic dependency closure through order
  six, array matching, residual expressions, signed quadrature integrals, and
  differentiable scalar cost graphs. CUDA expression kernels compile with NVRTC; CPU interprets the same instructions.
- Direct and simplex quadrature, optional Delaunay geometry, normal/tangent
  fields, frozen components, and attached first variations.
- RPROP and Adam state management; periodic RPROP step restoration, explicit
  gradient-history clearing when steps are replaced, reporting and stop/resume.
- Correct matrix ownership and bounds checks, shared derivative metadata,
  forward-only storage, skipped unnecessary backward work, device constant fills,
  and stable division adjoints. Preserve values when attached gradients cancel.
- Python dimension and state validation, cache-growth warnings, and synchronized
  generated-kernel inspection. Remove the dormant legacy interface and cuRAND
  initialization; retain the separate high-order derivative kernels.
- Standalone vanilla/extended Fourier, disk Poisson and circle Schrödinger
  examples, plus explicit first-derivative fitting and an attached variation.
  The circle eliminates its eigenvalue analytically and trains for 2000 updates.
- Example-based documentation with per-call links, schema explanations,
  eigenvalue-elimination equations, figures and numerical variation results.
- Source, Python wheel/sdist, and samples archives with checksums and manifests.
  Both backends are built locally; no prebuilt server executable is distributed.

See [known issues](KNOWN_ISSUES.md) for retained numerical limits and
[validation](VALIDATION.md) for the release checks. Earlier intermediate changes
and experiments are preserved in the [development history](docs/history/DEVELOPMENT_LOG.md).
