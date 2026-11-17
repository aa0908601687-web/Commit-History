# Kumo Independent MAX — operations and security

## New capabilities
- Local CSV and HTTPS CSV ingestion (20 MB remote response limit, timeout, no plain HTTP).
- Data drift alert heuristic using standardized mean shift (requires numeric columns).
- SHA-256 manifest for data and reports, with timestamp, Python version and file sizes.
- GitHub Actions scheduled/manual operations smoke test, with artifacts.
- Existing combined E2E validates real Kumo inference and independent LightGBM walk-forward.

## Commands
```sh
python operations.py ingest --source ./data.csv --out imported.csv
python operations.py drift --reference train.csv --current latest.csv --columns feature1,feature2
python operations.py manifest --files imported.csv pipeline_report.json --out run_manifest.json
```

## Security and limitations
Use GitHub Actions secrets or managed identity for credentials; never commit tokens or private datasets. HTTPS download is intended for trusted allowlisted endpoints; do not run untrusted URLs in privileged runners. SHA-256 does not provide trusted pre-event timestamps. Production auth, database connectors, secret rotation, model registry, alert delivery and deployed API service require infrastructure decisions. No production deployment or lottery forecasting advantage is claimed.
