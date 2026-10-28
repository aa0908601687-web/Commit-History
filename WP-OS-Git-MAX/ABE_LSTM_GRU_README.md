# Optional ABE LSTM/GRU/Transformer/Chronos-2/t0-gamma extension

This is **experimental**, not a proven lottery prediction improvement. The original `adaptive_bayesian_ensemble_max.py` and its workflow/caches remain unchanged.

- Uses the latest **20 completed draws** as each neural input sequence (38 first-zone indicators and 8 second-zone indicators).
- Needs **at least 60 completed draws**: 20 for the first input and at least 40 next-draw training examples. If fewer are available, return the existing ABE prediction unchanged and clearly mark the neural portion as skipped.
- Trains one small LSTM, one GRU and one attention-based Transformer independently for each invocation; each receives only records already published before the target draw. Outputs are normalized by lottery zone.
- Research-only score blend: existing ABE 70%, LSTM 10%, GRU 10%, Transformer 10%. These are **heuristic scoring weights**, not measured accuracy or optimal probabilities.
- Does not claim that 20 observations can train a neural network reliably. Training uses older observations to construct examples, while each input window uses only 20 draws.
- Transformer uses a two-head self-attention block, residual normalization and pooling; this is a small research architecture, not a pretrained time-series foundation model.
- No changes to production workflows or existing ABE caches. Neural dependencies and runtime remain opt-in.

## Chronos-2 optional inference

- Confirmed upstream model: `amazon/chronos-2` from the `chronos-forecasting` package.
- When `--chronos2` is enabled, it takes exactly the last 20 completed draws, encoded as 46 binary time series (38 first-zone indicators + 8 second-zone indicators), transposed to `(46,20)`, and generates one-step-ahead zero-shot median forecasts. No model fine-tuning.
- With >=60 prior draws and trained LSTM/GRU/Transformer, **provisional** fusion is ABE 60%, LSTM 10%, GRU 10%, Transformer 10%, Chronos-2 10%.
- With only 20–59 completed draws, neural training is skipped; Chronos-2 is optional and ABE 90% / Chronos-2 10%. Without `--chronos2`, the previous ABE + 3-model configuration is unchanged.
- Chronos-2 requires package installation and network access to download its pretrained weights on first use; failing dependencies are **not** silently reported as successful inference.
- Chronos-2 and the other models are not proven to predict lottery outcomes; no real-data backtest or end-to-end model download has yet been verified.

## t0-gamma optional inference

- `theforecastingcompany/t0-gamma` is an open-weights pretrained transformer, used through `tfc-t0[torch]>=0.6.0`. Source: https://huggingface.co/theforecastingcompany/t0-gamma
- `--t0-gamma` enables zero-shot inference on 46 one-hot indicator series, using the most recent 20 completed draws; 1-step median forecast. No lottery-specific fine tuning.
- With all six models active: **ABE 50%, LSTM 10%, GRU 10%, Transformer 10%, Chronos-2 10%, t0-gamma 10%**.
- If any optional model is not selected or neural training lacks enough examples, ABE receives its unused weight automatically. No changes to original ABE or pre-existing caches.
- Initial model download and PyTorch CPU inference may be substantial; live download, inference, CI, and out-of-sample validation are still unverified.
- Ordinary continuous-value time-series benchmark performance **does not** validate binary lottery indicator forecasting.

## Usage

```sh
python -m pip install numpy tensorflow
python -m pip install chronos-forecasting
python -m pip install 'tfc-t0[torch]>=0.6.0'
python WP-OS-Git-MAX/abe_lstm_gru.py --csv WP-OS-Git-MAX/historical_archive.csv --chronos2 --t0-gamma
python WP-OS-Git-MAX/abe_lstm_gru.py --csv WP-OS-Git-MAX/historical_archive.csv --no-neural
python -m unittest discover -s WP-OS-Git-MAX -p 'test_abe_lstm_gru.py' -v
```

For credible evaluation, add a genuine forward-only walk-forward comparison against unmodified ABE and random baselines, with tuning confined to the training period. **No such superiority test is claimed or provided by this extension.** Headline fair-jackpot odds remain 1 / 22,085,448 per ticket.
