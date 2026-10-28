# Optional ABE LSTM/GRU extension

This is **experimental**, not a proven lottery prediction improvement. The original `adaptive_bayesian_ensemble_max.py` and its workflow/caches remain unchanged.

- Uses the latest **20 completed draws** as each neural input sequence (38 first-zone indicators and 8 second-zone indicators).
- Needs **at least 60 completed draws**: 20 for the first input and at least 40 next-draw training examples. If fewer are available, return the existing ABE prediction unchanged and clearly mark the neural portion as skipped.
- Trains one small LSTM and one GRU independently for each invocation; each receives only records already published before the target draw. Outputs are normalized by lottery zone.
- Research-only score blend: existing ABE 80%, LSTM 10%, GRU 10%. These are **heuristic scoring weights**, not measured accuracy or optimal probabilities.
- Does not claim that 20 observations can train a neural network reliably. Training uses older observations to construct examples, while each input window uses only 20 draws.
- No changes to production workflows or existing ABE caches. Neural dependencies and runtime remain opt-in.

## Usage

```sh
python -m pip install numpy tensorflow
python WP-OS-Git-MAX/abe_lstm_gru.py --csv WP-OS-Git-MAX/historical_archive.csv
python WP-OS-Git-MAX/abe_lstm_gru.py --csv WP-OS-Git-MAX/historical_archive.csv --no-neural
python -m unittest discover -s WP-OS-Git-MAX -p 'test_abe_lstm_gru.py' -v
```

For credible evaluation, add a genuine forward-only walk-forward comparison against unmodified ABE and random baselines, with tuning confined to the training period. **No such superiority test is claimed or provided by this extension.** Headline fair-jackpot odds remain 1 / 22,085,448 per ticket.
