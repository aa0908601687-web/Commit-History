# Full Pipeline Upgrade — implementation status

This upgrade is isolated from OF-LGBM and the existing Kumo smoke test.

Implemented:
- CSV import and chronological data checks (unique ordered time, label validation, numeric features)
- Training-only imputation and scaling in walk-forward splits
- Expanding-window walk-forward with a dummy baseline, logistic regression and optional LightGBM
- Accuracy and log-loss evaluation; JSON output; input SHA-256 and environment details
- Unit tests for time ordering, invalid input and walk-forward
- Dedicated GitHub Actions job for baseline pipeline evaluation

Separate existing workflow: NVIDIA structured-data-models install and KumoTabular real inference smoke.
Not yet implemented: generalized Kumo-on-user-CSV adapter, database/API ingestion, production serving, drift alerts, authentication, secrets management, hosted NIM connection, model registry and operational monitoring. These require data schema, credentials and deployment decisions. The included sklearn dataset is a CI fixture, not lottery data. A green CI does not establish predictive advantage.
