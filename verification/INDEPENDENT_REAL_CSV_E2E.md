# Independent CSV E2E verification

This is a **research** extension to `verification/openfe_three_lottery_integration.py`. It is separate from the legacy V6 selection pipeline and does not add PyTorch, tsfresh, Featuretools, ABE, Nemotron, or QAOA.

Run a real historical CSV:

```bash
python verification/openfe_three_lottery_integration.py --game power --csv WP-OS-Git-MAX/historical_archive.csv
python verification/openfe_three_lottery_integration.py --game lotto --csv PATH_TO_LOTTO_CSV
python verification/openfe_three_lottery_integration.py --game 539 --csv PATH_TO_539_CSV
```

Power format: `draw_id,first,second`, with six first-zone numbers (1–38) and second-zone 1–8. Lotto: `draw_id,first` with six first-zone numbers (1–49). 539: `draw_id,first` with five numbers (1–39). Each row must be an already completed draw. All rows sorted in ascending draw order.

The two short historical views use the most recent 5 and 10 labeled training anchors, and the archive training view uses earlier labeled anchors as well; **all features are computed from preceding 5 or 10 completed draws only**. The weights 0.35/0.35/0.30 are *research scoring weights*. The Power second zone uses a separate empirical 5/10-draw frequency baseline; it is **not** OpenFE/LightGBM and is not hardcoded for real CSV.

The pipeline uses OpenFE and LightGBM for first-zone scores, compares exact 8-choose-6 search with dimod ExactSolver's QUBO, and runs CUDA-Q qpp-cpu separately. CUDA-Q does **not** select final numbers. CSV provenance is not independently certified merely by reading the file. A CI pass is an execution check, not evidence of better lottery odds. This workflow tests repository Power historical data; Lotto and 539 require their own properly sourced real CSVs for real-data E2E.
