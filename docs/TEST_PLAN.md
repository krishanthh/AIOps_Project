# Test Plan and Acceptance Criteria

I validate three cases separately: isolated application contracts, real dataset/model behavior, and deployed integration. 

| Area | Cases | Acceptance |
|---|---|---|
| Upload | Missing, empty, corrupt, oversized, disguised unsupported format, small dimensions | Clear 4xx response; no model call for rejected input |
| Serving boundary | Valid response, timeout, unavailable service, malformed response | Stable JSON contract; 504/503 failures; no invented prediction |
| Health | Liveness without model; readiness with/without serving | Liveness remains healthy; readiness accurately reflects dependency |
| Dataset | Repeated seeded splits, cross-partition exact duplicates, archive traversal | Reproducible manifests, unique retained decoded pixels, unsafe archive rejected |
| Model | Reloaded checkpoint, stable class order, explanation generation, hook cleanup | Repeat predictions agree; invalid class order rejected; PNG heatmap produced |
| Pipeline | Failed stage, simultaneous execution, disabled controls, absent artifacts/logs | Failure recorded, later stages pending, 409/403 enforced, useful empty states |
| Monitoring | Metrics during outage; sample-count gating; shifted brightness | Exposition available; insufficient samples not labeled drift; warning on shifted fixture |
| Real evaluation | Held-out accuracy, macro-F1, per-class recall, confusion matrix | Actual results recorded; validation promotion target ≥0.85 or model marked experimental |
| Integration | Browser/proxy → FastAPI → MLflow; invalid MLflow signature; container health | Successful real prediction, real explanation, invalid signature rejected |
| Performance | Ray throughput; actual prediction/explanation latency | Measured values saved; single-user prediction target ≤10 seconds |
