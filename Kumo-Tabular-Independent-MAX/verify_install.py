import importlib
import json
import platform
import sys
from datetime import datetime, timezone

report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
          "python": sys.version, "platform": platform.platform(), "checks": {}}
for name in ("torch", "sdm", "sklearn", "pandas"):
    try:
        module = importlib.import_module(name)
        report["checks"][name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as e:
        report["checks"][name] = {"ok": False, "error": str(e)}
try:
    import sdm
    report["checks"]["KumoTabular"] = {"ok": hasattr(sdm.models, "KumoTabular")}
except Exception as e:
    report["checks"]["KumoTabular"] = {"ok": False, "error": str(e)}
print(json.dumps(report, ensure_ascii=False, indent=2))
with open("installation_check.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
if not all(x["ok"] for x in report["checks"].values()):
    raise SystemExit("Installation verification FAILED")
