"""Real KumoTabular inference on a small public classification sample."""
import json
from datetime import datetime, timezone

import torch
import sdm
from sklearn.datasets import load_breast_cancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = load_breast_cancer(as_frame=True).frame
table = sdm.TableTensor.from_pandas(
    df=df, stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}), device=device
)
model = sdm.models.KumoTabular(task="classification", size="small", device=device)
with torch.inference_mode(), torch.amp.autocast(
    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
):
    probs = model(
        x_context=table[:300].drop_columns("target"),
        y_context=table[:300, "target"],
        x_query=table[300:].drop_columns("target"),
        num_estimators=8,
    )
assert len(probs) == len(df) - 300, f"Prediction count mismatch: {len(probs)}"
result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "device": str(device),
    "predictions": len(probs),
    "success": True,
}
print(json.dumps(result, indent=2))
with open("prediction_smoke_report.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
