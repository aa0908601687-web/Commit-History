# Kumo-Tabular Independent CI validation

This file records a request to rerun the independent CI pipeline on the `kumo-tabular-setup` branch.

Required checks: `unit`, `install` (including `pip check` and SDK import), and `e2e` (real KumoTabular inference, 269 held-out predictions, report assertions, and artifact upload).

A passing workflow is required before claiming end-to-end success. This branch is isolated from OF-LGBM.
