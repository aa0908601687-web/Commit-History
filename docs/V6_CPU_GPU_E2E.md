# V6 OF3Q CPU + GPU device verification

CPU GitHub Actions runs deterministic PyTorch float64 scores and the existing Python exact 8-choose-6 search (28 combinations). CPU PASS is mandatory. It reports GPU as NOT_TESTED on ordinary CPU-only runners.

A physical CUDA GPU run is opt-in through workflow_dispatch(enable_gpu=true). It requires an **already configured** self-hosted runner with labels self-hosted/linux/x64/gpu, plus NVIDIA drivers and CUDA-enabled PyTorch. The workflow does not provision hardware or install GPU drivers.

GPU verification fails if no CUDA device is detected. If a GPU exists, per-number scores must agree with CPU within 1e-8, selection must match, and both count 28 combinations.

This is a device-consistency smoke test, not real-model GPU training, full OpenFE/LightGBM/PyTorch integration, improved lottery hit-rate evidence, or proof of physical GPU installation.
