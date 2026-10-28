# Optional ABE + LSTM + GRU + Chronos-2 + t0-gamma research extension

The standalone custom Transformer model has been removed. Chronos-2 and t0-gamma are retained: their internal architectures may themselves use Transformer mechanisms. Original ABE source, main branch, production workflows, and caches are unchanged.

- Each input uses the most recent **20 completed draws**: 38 first-zone indicators and 8 second-zone indicators.
- LSTM and GRU train on prior completed draws only; at least 60 completed draws are required (40 supervised examples). Training never sees a future target draw.
- Both Chronos-2 and t0-gamma are optional pretrained, zero-shot, one-step inference models. Their first use requires third-party dependencies and downloaded weights.
- Without optional pretrained models and with sufficient history: **ABE 80%, LSTM 10%, GRU 10%**.
- With Chronos-2 only: **ABE 70%, LSTM 10%, GRU 10%, Chronos-2 10%**.
- With all remaining models enabled and sufficient history: **ABE 60%, LSTM 10%, GRU 10%, Chronos-2 10%, t0-gamma 10%**.
- If the history is insufficient for neural training, LSTM and GRU are skipped and their weights return to ABE.
- Individual pretrained models are opt-in via `--chronos2` and `--t0-gamma`. A missing dependency is reported as an error, not as a successful prediction.
- These percentages are initial research scoring coefficients, **not** verified accuracy or optimal weights.

## Run

```sh
python -m pip install numpy tensorflow chronos-forecasting 'tfc-t0[torch]>=0.6.0'
python WP-OS-Git-MAX/abe_lstm_gru.py --csv WP-OS-Git-MAX/historical_archive.csv --chronos2 --t0-gamma
python WP-OS-Git-MAX/abe_lstm_gru.py --csv WP-OS-Git-MAX/historical_archive.csv --no-neural
python -m unittest discover -s WP-OS-Git-MAX -p 'test_abe_lstm_gru.py' -v
```

This branch is not evidence of a predictive advantage in a fair lottery. Actual dependency download/inference, CI, and leakage-free historical validation require separate checks. Fair jackpot probability remains 1 / 22,085,448 for a single ticket.
