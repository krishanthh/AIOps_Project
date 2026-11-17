# Low-Level Design and API Contract

I separate HTTP application behavior from the model engine. FastAPI publishes an interactive OpenAPI contract at `/docs` and machine-readable specifications at `/openapi.json`.

| Method and endpoint | Input | Output / failure behavior |
|---|---|---|
| GET `/health` | None | 200 `{status: "ok"}`; process liveness only |
| GET `/ready` | None | 200 `{status: "ready"}` when MLflow `/ping` succeeds; otherwise 503 |
| GET `/api/v1/classes` | None | Fixed class array and upload limit |
| POST `/api/v1/predict?explain=true` | Multipart `image`: decoded JPEG/PNG, ≤10 MB, each dimension ≥32, ≤16 MP | Prediction schema below; 413 excessive bytes, 422 invalid image/missing file, 503 unavailable/malformed serving response, 504 timeout |
| GET `/api/v1/pipeline` | None | DAG order, stage states, last run metadata, quality/preparation reports, controls-enabled flag |
| POST `/api/v1/pipeline/run` | No body | 202 accepted PID; 403 controls disabled; 409 active run |
| GET `/api/v1/pipeline/log/{filename}` | Basename ending `.log` | Last 20 KB as text; 400 invalid name; 404 missing |
| GET `/api/v1/models` | None | Local experiment comparison, selected artifact manifest, held-out evaluation; missing artifacts represented by empty list/null |
| GET `/api/v1/drift` | None | Insufficient-data state or brightness departure and warning status |
| GET `/api/v1/links` | None | Local monitoring tool URLs |
| GET `/metrics` | None | Prometheus text exposition; available during model outage |

## Prediction response

```json
{
  "request_id": "generated unique identifier",
  "predicted_class": "glioma",
  "confidence": 0.7,
  "probabilities": {"glioma": 0.7, "meningioma": 0.1, "notumor": 0.1, "pituitary": 0.1},
  "gradcam_base64": "PNG base64, or empty string when explanation is disabled",
  "model_version": "selected MLflow run ID",
  "experimental": true,
  "inference_time_ms": 123.4
}
```

The numbers above illustrate the interface; they are not evaluation results. The probability order is fixed by the artifact manifest. Uploads are decoded and re-encoded to a bounded 224 × 224 PNG before transfer to serving, removing original image metadata. Preprocessing then applies the same resize/tensor/ImageNet normalization used during training.

## MLflow interface

`POST /invocations` receives `dataframe_records` containing `image_base64` (string) and `explain` (boolean). The saved MLflow signature validates these column types. The output dataframe includes class, confidence, serialized probability map, optional heatmap, version, and experimental flag. FastAPI converts the probability map into the public JSON shape. Serving uses the packaged model code and checkpoint, not a separate locally reconstructed API model.

## Pipeline and persistence

The ordered stages are acquire, split, prepare, train, evaluate, and package. The runner executes `python -m dvc repro <stage>`, records stage duration and exit code, stops after failure, and writes atomic current/history JSON reports. File-lock ownership prevents simultaneous runners. Logs contain pipeline output; prediction logs record route/status rather than image bytes or patient identifiers.

Data lineage includes the source archive SHA256, image hashes, split seed, split manifest fingerprint, Git commit, and MLflow run ID. The checkpoint includes class order, which is validated on reload. Prepared images are versioned through DVC. Operational reports are local files and are not transactional multi-user records.
