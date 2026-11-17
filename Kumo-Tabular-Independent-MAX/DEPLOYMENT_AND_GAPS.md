# Kumo Independent MAX: local service and remaining gaps

Implemented:
- SQLite report registry (model name, status, UTC time, SHA-256, JSON report).
- Loopback-only GET /health and GET /runs API, no mutation endpoints.
- Optional HTTPS webhook alerts via KUMO_ALERT_WEBHOOK environment variable.
- Existing local/HTTPS CSV ingestion, drift heuristic, custom Kumo CSV inference, LightGBM walk-forward and CI.

Examples:
```sh
python service.py register --db runs.sqlite3 --report kumo_csv_report.json --model kumo
python service.py serve --db runs.sqlite3 --port 8765
KUMO_ALERT_WEBHOOK=https://YOUR-TRUSTED-ENDPOINT python service.py alert --message "Kumo pipeline failed"
```

Security: service binds 127.0.0.1 only. Do not expose it publicly or assume production authentication, TLS termination, authorization, audit trail, backups or deployment. Never commit webhook credentials. Webhook delivery is opt-in and not configured by default. No external database adapter or managed model registry exists. Model accuracy and lottery prediction advantage are unproven. GitHub workflow checks are not a production release certification.
