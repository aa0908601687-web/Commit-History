#!/usr/bin/env python3
"""Optional LSTM+GRU research extension for ABE. No claimed lottery edge.

Requires TensorFlow only when neural training is requested. The original ABE
model is unchanged. All neural training examples precede the forecast draw.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abe_original", HERE / "adaptive_bayesian_ensemble_max.py")
abe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abe)

LOOKBACK = 20
MIN_TRAIN_EXAMPLES = 40
FEATURES = 46
NEURAL_WEIGHT = 0.20  # Research-only fixed cap; unvalidated.


def vector(draw):
    """38 binary first-zone positions + 8 binary second-zone positions."""
    x = [0.0] * FEATURES
    for n in draw["first"]:
        if not 1 <= n <= 38:
            raise ValueError("Invalid first-zone number")
        x[n - 1] = 1.0
    if len(set(draw["first"])) != 6:
        raise ValueError("Six distinct first-zone numbers required")
    n = draw["second"]
    if not 1 <= n <= 8:
        raise ValueError("Invalid second-zone number")
    x[38 + n - 1] = 1.0
    return x


def examples(draws):
    """Only previously completed draw results, 20-step inputs, next-step labels."""
    if len(draws) < LOOKBACK + MIN_TRAIN_EXAMPLES:
        raise ValueError("At least 60 completed draws required for neural training")
    seq = [vector(d) for d in draws]
    inputs = [seq[i - LOOKBACK:i] for i in range(LOOKBACK, len(seq))]
    targets = [seq[i] for i in range(LOOKBACK, len(seq))]
    return inputs, targets


def abe_scores(draws):
    result = {}
    for key, upper, scalar in (("first", 38, False), ("second", 8, True)):
        seq = [d[key] for d in draws]
        w = abe.weight(seq, scalar, upper)
        a = abe.normalized(abe.rank_zone(seq, upper, 5, 10, scalar))
        b = abe.normalized(abe.rank_zone(seq, upper, 10, 20, scalar))
        result[key] = [w * a[n] + (1 - w) * b[n] for n in range(1, upper + 1)]
    return result


def normalize(values):
    clean = [max(0.0, float(v)) if math.isfinite(float(v)) else 0.0 for v in values]
    total = sum(clean)
    return [x / total for x in clean] if total else [1 / len(clean)] * len(clean)


def train_neural(draws, epochs=12, seed=20261028):
    """Train two independent research models using only past completed draws."""
    xs, ys = examples(draws)
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Neural extension requires tensorflow and numpy") from exc
    tf.keras.utils.set_random_seed(seed)
    X = np.asarray(xs, dtype="float32")
    Y = np.asarray(ys, dtype="float32")
    next_x = np.asarray([[vector(d) for d in draws[-LOOKBACK:]]], dtype="float32")
    predictions = {}
    for kind in ("LSTM", "GRU"):
        layer = tf.keras.layers.LSTM if kind == "LSTM" else tf.keras.layers.GRU
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(LOOKBACK, FEATURES)),
            layer(16),
            tf.keras.layers.Dense(FEATURES, activation="sigmoid"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="binary_crossentropy")
        model.fit(X, Y, batch_size=min(16, len(X)), epochs=epochs, verbose=0,
                  shuffle=False)
        predictions[kind] = model.predict(next_x, verbose=0)[0].tolist()
    return predictions, len(xs)


def predict(draws, use_neural=True, epochs=12, seed=20261028):
    if len(draws) < LOOKBACK:
        raise ValueError("At least 20 completed draws required")
    baseline = abe.predict(draws)
    output = dict(baseline)
    output["model"] = "ABE + LSTM + GRU research extension"
    output["neural_status"] = "not_requested"
    output["neural_training_examples"] = 0
    output["neural_weight"] = 0.0
    if use_neural:
        if len(draws) < LOOKBACK + MIN_TRAIN_EXAMPLES:
            output["neural_status"] = "skipped_insufficient_history"
        else:
            neural, count = train_neural(draws, epochs, seed)
            baseline_scores = abe_scores(draws)
            for zone, offset, upper, select in (("first", 0, 38, 6), ("second", 38, 8, 1)):
                l = normalize(neural["LSTM"][offset:offset + upper])
                g = normalize(neural["GRU"][offset:offset + upper])
                fused = [(1 - NEURAL_WEIGHT) * a + NEURAL_WEIGHT * (x + y) / 2
                         for a, x, y in zip(baseline_scores[zone], l, g)]
                selected = sorted(range(1, upper + 1), key=lambda n: (-fused[n - 1], n))[:select]
                output[zone] = sorted(selected) if zone == "first" else selected[0]
            output["neural_status"] = "trained"
            output["neural_training_examples"] = count
            output["neural_weight"] = NEURAL_WEIGHT
    output["config"] = {"lookback": LOOKBACK, "min_neural_training_examples": MIN_TRAIN_EXAMPLES,
                        "epochs": epochs, "seed": seed, "neural_weight": NEURAL_WEIGHT}
    output.pop("sha256", None)
    output["sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--no-neural", action="store_true", help="Run unmodified ABE output only")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261028)
    args = p.parse_args()
    if not 1 <= args.epochs <= 100:
        p.error("--epochs must be from 1 to 100")
    draws = abe.base.load(args.csv)
    print(json.dumps(predict(draws, not args.no_neural, args.epochs, args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
