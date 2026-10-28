#!/usr/bin/env python3
"""Optional LSTM+GRU+Transformer research extension for ABE. No claimed lottery edge.

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
NEURAL_WEIGHT = 0.30  # Research-only fixed cap; unvalidated.


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
    for kind in ("LSTM", "GRU", "Transformer"):
        if kind == "Transformer":
            inputs = tf.keras.Input(shape=(LOOKBACK, FEATURES))
            projected = tf.keras.layers.Dense(32)(inputs)
            attended = tf.keras.layers.MultiHeadAttention(
                num_heads=2, key_dim=16, dropout=0.1)(projected, projected)
            hidden = tf.keras.layers.LayerNormalization()(
                tf.keras.layers.Add()([projected, attended]))
            feedforward = tf.keras.layers.Dense(32, activation="relu")(hidden)
            hidden = tf.keras.layers.LayerNormalization()(
                tf.keras.layers.Add()([hidden, feedforward]))
            pooled = tf.keras.layers.GlobalAveragePooling1D()(hidden)
            outputs = tf.keras.layers.Dense(FEATURES, activation="sigmoid")(pooled)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
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



def chronos2_scores(draws):
    """Pretrained Chronos-2 zero-shot prediction of 46 indicator series."""
    try:
        import numpy as np
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise RuntimeError("Install chronos-forecasting, numpy and torch") from exc
    history = np.asarray([vector(d) for d in draws[-LOOKBACK:]], dtype="float32").T
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    _, means = pipeline.predict_quantiles([history], prediction_length=1, quantile_levels=[0.5])
    values = np.asarray(means[0], dtype="float32").reshape(-1)
    if values.shape != (FEATURES,) or not np.isfinite(values).all():
        raise ValueError("Chronos-2 returned invalid values")
    return values.tolist()



def t0_gamma_scores(draws):
    """Zero-shot median 1-step forecast from 20 completed draws, 46 indicator rows."""
    try:
        import numpy as np
        from t0 import T0Forecaster
    except ImportError as exc:
        raise RuntimeError('Install "tfc-t0[torch]>=0.6.0" and numpy for --t0-gamma') from exc
    context = np.asarray([vector(d) for d in draws[-LOOKBACK:]], dtype="float32").T
    model = T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma").eval()
    forecast = model.predict(context, horizon=1, quantile_levels=[0.5])
    values = np.asarray(forecast.median, dtype="float32").reshape(-1)
    if values.shape != (FEATURES,) or not np.isfinite(values).all():
        raise ValueError("t0-gamma returned invalid scores")
    return values.tolist()


def predict(draws, use_neural=True, epochs=12, seed=20261028, use_chronos2=False, use_t0_gamma=False):
    if len(draws) < LOOKBACK:
        raise ValueError("At least 20 completed draws required")
    baseline = abe.predict(draws)
    output = dict(baseline)
    output["model"] = "ABE + LSTM + GRU + Transformer research extension"
    output["neural_status"] = "not_requested"
    output["neural_training_examples"] = 0
    output["neural_weight"] = 0.0
    output["chronos2_status"] = "not_requested"
    if use_neural:
        if len(draws) < LOOKBACK + MIN_TRAIN_EXAMPLES:
            output["neural_status"] = "skipped_insufficient_history"
        else:
            neural, count = train_neural(draws, epochs, seed)
            baseline_scores = abe_scores(draws)
            for zone, offset, upper, select in (("first", 0, 38, 6), ("second", 38, 8, 1)):
                l = normalize(neural["LSTM"][offset:offset + upper])
                g = normalize(neural["GRU"][offset:offset + upper])
                t = normalize(neural["Transformer"][offset:offset + upper])
                fused = [(1 - NEURAL_WEIGHT) * a + NEURAL_WEIGHT * (x + y + z) / 3
                         for a, x, y, z in zip(baseline_scores[zone], l, g, t)]
                selected = sorted(range(1, upper + 1), key=lambda n: (-fused[n - 1], n))[:select]
                output[zone] = sorted(selected) if zone == "first" else selected[0]
            output["neural_status"] = "trained"
            output["neural_training_examples"] = count
            output["neural_weight"] = NEURAL_WEIGHT

    if use_chronos2:
        c_scores = chronos2_scores(draws)
        a_scores = abe_scores(draws)
        trained = output["neural_status"] == "trained"
        for zone, offset, upper, count in (("first", 0, 38, 6), ("second", 38, 8, 1)):
            c = normalize(c_scores[offset:offset + upper])
            if trained:
                l = normalize(neural["LSTM"][offset:offset + upper])
                g = normalize(neural["GRU"][offset:offset + upper])
                t = normalize(neural["Transformer"][offset:offset + upper])
                fused = [0.60*a + 0.10*(x+y+z+q)
                         for a,x,y,z,q in zip(a_scores[zone],l,g,t,c)]
            else:
                fused = [0.90*a + 0.10*q for a,q in zip(a_scores[zone],c)]
            picks = sorted(range(1,upper+1),key=lambda n:(-fused[n-1],n))[:count]
            output[zone] = sorted(picks) if zone == "first" else picks[0]
        output["chronos2_status"] = "inferred"
        output["chronos2_weight"] = 0.10
        output["model"] += " + Chronos-2"
    else:
        output["chronos2_weight"] = 0.0
    if use_t0_gamma:
        t0 = t0_gamma_scores(draws)
        # Rebuild fusion from the underlying score tables, not from rounded picks.
        basic = abe_scores(draws)
        trained = output["neural_status"] == "trained"
        chronos_available = output["chronos2_status"] == "inferred"
        for zone, offset, upper, count in (("first", 0, 38, 6), ("second", 38, 8, 1)):
            components = [basic[zone]]
            if trained:
                for key in ("LSTM", "GRU", "Transformer"):
                    components.append(normalize(neural[key][offset:offset + upper]))
            if chronos_available:
                components.append(normalize(c_scores[offset:offset + upper]))
            components.append(normalize(t0[offset:offset + upper]))
            n_aux = len(components) - 1
            # Each enabled auxiliary model scores 10%; ABE gets the remainder.
            fused = [(1 - .10 * n_aux) * components[0][i] +
                     .10 * sum(component[i] for component in components[1:])
                     for i in range(upper)]
            picks = sorted(range(1, upper + 1), key=lambda n: (-fused[n-1], n))[:count]
            output[zone] = sorted(picks) if zone == "first" else picks[0]
        output["t0_gamma_status"] = "inferred"
        output["t0_gamma_weight"] = 0.10
        output["model"] += " + t0-gamma"
    else:
        output["t0_gamma_status"] = "not_requested"
        output["t0_gamma_weight"] = 0.0
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
    p.add_argument("--chronos2", action="store_true")
    p.add_argument("--t0-gamma", action="store_true", help="Enable optional pretrained t0-gamma")
    args = p.parse_args()
    if not 1 <= args.epochs <= 100:
        p.error("--epochs must be from 1 to 100")
    draws = abe.base.load(args.csv)
    print(json.dumps(predict(draws, not args.no_neural, args.epochs, args.seed, args.chronos2, args.t0_gamma),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
