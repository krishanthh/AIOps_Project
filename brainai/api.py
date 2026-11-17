"""Application REST boundary; delegates inference exclusively to MLflow serving."""

import base64
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from brainai.common import CLASSES, MAX_BYTES, ROOT, decode_image, read_json
from brainai.pipeline import STAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("brainai")
app = FastAPI(title="Brain MRI Classification", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3100").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
SERVING_URL = os.getenv("MODEL_SERVING_URL", "http://127.0.0.1:5001")
TRACKING_URL = os.getenv("MLFLOW_SERVICE_URL", "http://127.0.0.1:5000")
TRACKING_AVAILABLE = Gauge("brainai_tracking_available", "MLflow tracking availability")
REQUESTS = Counter("brainai_requests_total", "HTTP requests", ["route", "status"])
LATENCY = Histogram(
    "brainai_request_seconds", "HTTP duration", ["route"], buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)
PREDICTIONS = Counter("brainai_predictions_total", "Completed predictions", ["class"])
ERRORS = Counter("brainai_prediction_errors_total", "Prediction failures", ["reason"])
AVAILABLE = Gauge("brainai_model_available", "Model-serving availability")
PIPELINE_SECONDS = Gauge("brainai_pipeline_stage_seconds", "Last stage duration", ["stage"])
THROUGHPUT = Gauge("brainai_preparation_images_per_second", "Ray image throughput")
DRIFT = Gauge("brainai_brightness_drift_zscore", "Recent mean brightness departure from baseline")
RECENT, HISTORY_LOCK = [], threading.Lock()
RUN_LOCK = threading.Lock()
pipeline_process = None


@app.middleware("http")
async def instrument(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    name = route.path if route else "unmatched"
    REQUESTS.labels(name, str(response.status_code)).inc()
    LATENCY.labels(name).observe(time.perf_counter() - started)
    logger.info("method=%s route=%s status=%s", request.method, name, response.status_code)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{SERVING_URL}/ping")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        AVAILABLE.set(0)
        raise HTTPException(503, "Model serving is unavailable") from exc
    AVAILABLE.set(1)
    return {"status": "ready"}


@app.get("/api/v1/classes")
def classes():
    return {"classes": CLASSES, "max_upload_mb": 10}


@app.post("/api/v1/predict")
async def predict(image: UploadFile = File(...), explain: bool = Query(True)):
    content = await image.read(MAX_BYTES + 1)
    try:
        decoded = decode_image(content)
    except ValueError as exc:
        ERRORS.labels("validation").inc()
        raise HTTPException(413 if len(content) > MAX_BYTES else 422, str(exc)) from exc
    # Remove original metadata and bound the serving payload.
    buffer = io.BytesIO()
    decoded.resize((224, 224)).save(buffer, format="PNG")
    payload = {
        "dataframe_records": [
            {"image_base64": base64.b64encode(buffer.getvalue()).decode(), "explain": explain}
        ]
    }
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{SERVING_URL}/invocations", json=payload)
        response.raise_for_status()
        result = response.json()["predictions"][0]
        probabilities = json.loads(result.pop("probabilities_json"))
        if set(probabilities) != set(CLASSES):
            raise ValueError("Serving class contract mismatch")
    except httpx.TimeoutException as exc:
        ERRORS.labels("timeout").inc()
        raise HTTPException(504, "Model serving timed out; please retry") from exc
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        AVAILABLE.set(0)
        ERRORS.labels("serving").inc()
        logger.error("Model serving failure: %s", type(exc).__name__)
        raise HTTPException(
            503, "Model serving is unavailable or returned an invalid response"
        ) from exc
    AVAILABLE.set(1)
    PREDICTIONS.labels(result["predicted_class"]).inc()
    with HISTORY_LOCK:
        RECENT.append(
            float(np.asarray(decoded.resize((224, 224)).convert("L"), dtype=float).mean() / 255)
        )
        del RECENT[:-100]
    return {
        **result,
        "probabilities": probabilities,
        "request_id": uuid.uuid4().hex,
        "inference_time_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.get("/api/v1/pipeline")
def pipeline():
    state = read_json(ROOT / "reports/pipeline.json", {"status": "never_run", "stages": {}})
    return {
        **state,
        "dag": STAGES,
        "preparation": read_json(ROOT / "reports/preparation.json"),
        "quality": read_json(ROOT / "reports/data-quality.json"),
        "controls_enabled": os.getenv("ENABLE_PIPELINE_CONTROL", "false") == "true",
    }


@app.post("/api/v1/pipeline/run", status_code=202)
def run_pipeline():
    global pipeline_process
    if os.getenv("ENABLE_PIPELINE_CONTROL", "false") != "true":
        raise HTTPException(403, "Pipeline control is disabled; use the local training command")
    with RUN_LOCK:
        if pipeline_process is not None and pipeline_process.poll() is None:
            raise HTTPException(409, "A pipeline is already running")
        import fcntl

        (ROOT / "reports").mkdir(exist_ok=True)
        with open(ROOT / "reports/pipeline.lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HTTPException(409, "A pipeline is already running") from exc
        pipeline_process = subprocess.Popen([sys.executable, "-m", "brainai.pipeline"], cwd=ROOT)
    return {"status": "accepted", "pid": pipeline_process.pid}


@app.get("/api/v1/pipeline/log/{filename}", response_class=PlainTextResponse)
def pipeline_log(filename: str):
    if Path(filename).name != filename or not filename.endswith(".log"):
        raise HTTPException(400, "Invalid log identifier")
    path = ROOT / "logs" / filename
    if not path.is_file():
        raise HTTPException(404, "Log unavailable")
    with open(path, "rb") as stream:
        stream.seek(max(0, path.stat().st_size - 20000))
        return stream.read().decode(errors="replace")


@app.get("/api/v1/models")
def models():
    return {
        "experiments": read_json(ROOT / "reports/experiments.json", []),
        "model": read_json(ROOT / "artifacts/manifest.json"),
        "evaluation": read_json(ROOT / "reports/evaluation.json"),
    }


@app.get("/api/v1/drift")
def drift():
    baseline = read_json(ROOT / "reports/preparation.json")
    with HISTORY_LOCK:
        recent = list(RECENT)
    if not baseline or len(recent) < 20:
        return {"status": "insufficient_data", "samples": len(recent), "minimum_samples": 20}
    zscore = abs(float(np.mean(recent)) - baseline["baseline_brightness_mean"]) / max(
        baseline["baseline_brightness_std"], 0.01
    )
    DRIFT.set(zscore)
    return {
        "status": "warning" if zscore > 2 else "within_baseline",
        "zscore": zscore,
        "samples": len(recent),
        "description": "Brightness distribution warning, not accuracy",
    }


@app.get("/metrics")
async def metrics():
    try:
        await ready()
    except HTTPException:
        pass
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            tracking = await client.get(f"{TRACKING_URL}/health")
        TRACKING_AVAILABLE.set(int(tracking.is_success))
    except httpx.HTTPError:
        TRACKING_AVAILABLE.set(0)
    state = pipeline()
    for stage, value in state.get("stages", {}).items():
        PIPELINE_SECONDS.labels(stage).set(value.get("seconds", 0))
    if state.get("preparation"):
        THROUGHPUT.set(state["preparation"]["images_per_second"])
    drift()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/links")
def links():
    return {
        "mlflow": "http://localhost:5000",
        "grafana": "http://localhost:3101",
        "prometheus": "http://localhost:9090",
    }
