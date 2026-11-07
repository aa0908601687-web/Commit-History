#!/usr/bin/env python3
"""Optional CPU PyTorch scorer trained on past-only OpenFE base-row features.

Experimental research model. Scores are not calibrated winning probabilities.
No future draw is used for training or feature generation.
"""
from __future__ import annotations

from v6_openfe_bridge import SEED, VIEW_LIMITS, _base_row, _training_frame

def score_three_history_views_pytorch(rows, *, epochs=12, hidden=24):
    if len(rows) < 12:
        raise ValueError("PyTorch three-view scorer requires at least 12 completed draws")
    if not 1 <= epochs <= 100 or not 1 <= hidden <= 256:
        raise ValueError("invalid PyTorch training parameters")
    import numpy as np
    import pandas as pd
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    features, labels = _training_frame(rows, three_views=True)
    current = pd.DataFrame([
        _base_row(rows, number, limit)
        for limit in VIEW_LIMITS.values()
        for number in range(1, 39)
    ])
    if list(features.columns) != list(current.columns):
        raise ValueError("PyTorch feature schema mismatch")
    x_train = features.to_numpy(dtype=np.float32)
    x_current = current.to_numpy(dtype=np.float32)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_train = (x_train - mean) / scale
    x_current = (x_current - mean) / scale
    if not np.isfinite(x_train).all() or not np.isfinite(x_current).all():
        raise ValueError("non-finite PyTorch input features")

    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    inputs = torch.from_numpy(x_train)
    targets = torch.as_tensor(labels.to_numpy(dtype=np.float32)).reshape(-1, 1)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        if not torch.isfinite(loss):
            raise ValueError("non-finite PyTorch loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x_current))).reshape(-1).tolist()
    if len(probs) != 114 or not all(0 <= p <= 1 and np.isfinite(p) for p in probs):
        raise ValueError("invalid PyTorch score vector")
    views = {}
    for index, name in enumerate(VIEW_LIMITS):
        views[name] = {n: float(probs[index * 38 + n - 1]) for n in range(1, 39)}
    fused = {n: sum(views[k][n] for k in VIEW_LIMITS) / 3 for n in range(1, 39)}
    return fused, {
        "engine": "PyTorch-CPU-MLP",
        "enabled": True,
        "trained": True,
        "epochs": epochs,
        "hidden_units": hidden,
        "seed": SEED,
        "training_rows": len(features),
        "feature_names": list(features.columns),
        "selection_feature_windows": [5, 10],
        "history_views": list(VIEW_LIMITS),
        "per_view_scores": views,
        "leakage_safe_by_construction": True,
        "predictive_advantage_verified": False,
    }
