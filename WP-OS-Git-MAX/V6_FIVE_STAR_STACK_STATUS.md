# V6 OF3Q optional high-performance stack (experimental)

This PR **adds a separate testable integration entry point**, not a production replacement for OpenFE or CUDA-Q. The existing main workflow and selection results are unchanged.

| Area | Candidate packages | Current status |
|---|---|---|
| Automatic feature engineering | Featuretools, tsfresh, AutoGluon | Optional dependency discovery only; no training or feature parity validated |
| Tabular and numerical computation | Polars, NumPy, JAX, Numba | Optional dependency discovery only; no benchmark-based speed claim |
| Model scoring | LightGBM | Existing production model; optional discovery in this probe |
| Optimization | OR-Tools | Independent 8-choose-6 CP-SAT solver implemented; opt-in CI comparison with exhaustive Python |
| Commercial/other solver | Gurobi, PySCIPOpt (SCIP) | Optional dependency discovery only; Gurobi may require a license |

The exact reference checks all 28 combinations. OR-Tools uses rounded integer coefficients, so its objective is an approximation to arbitrary floating-point scores. Tests use synthetic input and verify equal objective values on that fixture only.

**Not yet complete:** live feature generation, library performance measurements, actual package installation in production, integration into `v6_exact_e2e.py`, production selection parity, independent CUDA-Q selection, and full E2E certification. Do not label this a 100% production migration or claim improved lottery odds.

Use the new `V6 optional five-star stack smoke` workflow to verify the isolated Python and OR-Tools paths before any production integration. Keep legacy workflows and QAOA exclusion unchanged.
