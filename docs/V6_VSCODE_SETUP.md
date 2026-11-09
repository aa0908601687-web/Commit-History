# V6 OF3Q VS Code development setup

These workspace files provide recommended extensions, debugger profiles, Python unittest discovery, and local test tasks. VS Code is a development editor; it does not score lottery numbers or change hit probabilities.

1. Install VS Code and Python 3.11+ locally. Open the repository root in VS Code and accept the recommended extensions.
2. Select the intended Python interpreter; install the project test and optional AI dependencies in your environment separately.
3. Run **Terminal > Run Task > V6: Exact-search unit tests** for dependency-free exact-combination tests.
4. Run **V6 OF3Q: Debug exact 8-to-6 selection** to step through the current exact-selection entrypoint. This reads a repository historical CSV and writes a local result JSON; do not assume it is a live official draw prediction.
5. GitHub CI on PR #57 remains the source for integration verification; local editor configuration does not mean GitHub tests have passed or the PR is merged.

**Current intended architecture:** OpenFE/LightGBM/PyTorch candidate scoring -> top-eight pool -> Python exhaustive 28-combination choice; Ising/QUBO and Rocq remain independent validation. CUDA-Q may be used for local diagnostics, not QAOA selection.
