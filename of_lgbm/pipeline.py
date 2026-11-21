"""Optional OpenFE + LightGBM integration; no lottery accuracy claims."""
from __future__ import annotations
import hashlib
from pathlib import Path

def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def train_openfe_lightgbm(train_x, train_y, valid_x, valid_y, *, seed=42, n_jobs=1):
    """Fit OpenFE on training data ONLY; evaluate on held-out validation rows.

    Inputs must be chronologically split by the caller, with labels and
    all input features available as of the prediction timestamp.
    """
    if len(train_x) != len(train_y) or len(valid_x) != len(valid_y):
        raise ValueError("Feature and label row counts must match")
    if len(train_x) == 0 or len(valid_x) == 0:
        raise ValueError("Train and validation sets must not be empty")
    from openfe import OpenFE, transform
    import lightgbm as lgb
    generator = OpenFE()
    features = generator.fit(data=train_x, label=train_y, n_jobs=n_jobs)
    transformed_train, transformed_valid = transform(
        train_x, valid_x, features, n_jobs=n_jobs
    )
    model = lgb.LGBMClassifier(random_state=seed, verbosity=-1, n_jobs=n_jobs)
    model.fit(transformed_train, train_y)
    probabilities = model.predict_proba(transformed_valid)
    return {"model": model, "features": features, "probabilities": probabilities}
