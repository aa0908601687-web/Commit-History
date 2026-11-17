"""KumoTabular end-to-end smoke: download weights and predict held-out rows."""
import json
from datetime import datetime, timezone

import torch
import sdm
from sdm.evaluation import to_class_indices
from sklearn.datasets import load_breast_cancer

device = "cuda" if torch.cuda.is_available() else "cpu"
df = load_breast_cancer(as_frame=True).frame
table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}),
    device=device,
)
context, query = table[:300], table[300:]
model = sdm.models.KumoTabular(task="classification", size="small", device=device)
with torch.inference_mode(), torch.amp.autocast(
    device_type=device, dtype=torch.float16, enabled=device == "cuda"
):
    probs = model(
        x_context=context.drop_columns("target"),
        y_context=context["target"],
        x_query=query.drop_columns("target"),
        num_estimators=8,
        generator=torch.Generator(device=device).manual_seed(0),
    )
scores, labels = to_class_indices(probs, query["target"])
assert scores.shape[0] == len(df) - 300, "Prediction count mismatch"
accuracy = float((scores.argmax(dim=-1) == labels).float().mean())
assert 0 <= accuracy <= 1, "Invalid accuracy"
result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "device": str(device),
    "predictions": int(scores.shape[0]),
    "accuracy": accuracy,
    "success": True,
}
print(json.dumps(result, indent=2))
with open("prediction_smoke_report.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
