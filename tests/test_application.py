"""Boundary, failure, resource, and reproducibility tests using isolated fixture artifacts."""

import base64
import io
import json
import subprocess
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from brainai import api, data, pipeline
from brainai.common import CLASSES, MAX_BYTES, decode_image, read_json, write_json
from brainai.model import MRIModel, build_model, load_checkpoint


def image_bytes(seed=0):
    image = Image.fromarray(
        np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setenv("ENABLE_PIPELINE_CONTROL", "false")
    with TestClient(api.app) as client:
        yield client


def serving(monkeypatch, mode="ok"):
    original = httpx.AsyncClient

    def handle(request):
        if mode == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if mode == "down":
            return httpx.Response(503)
        if request.url.path == "/ping":
            return httpx.Response(200, json={"status": "ok"})
        payload = json.loads(request.content)
        assert isinstance(payload["dataframe_records"][0]["explain"], bool)
        if mode == "bad":
            return httpx.Response(200, json={"predictions": []})
        return httpx.Response(
            200,
            json={
                "predictions": [
                    {
                        "predicted_class": "glioma",
                        "confidence": 0.7,
                        "probabilities_json": json.dumps(dict(zip(CLASSES, [0.7, 0.1, 0.1, 0.1]))),
                        "gradcam_base64": "",
                        "model_version": "fixture",
                        "experimental": True,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        api.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )


def test_health_does_not_require_model(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_file(client):
    assert client.post("/api/v1/predict").status_code == 422


@pytest.mark.parametrize("content", [b"", b"not an image", b"GIF89a"])
def test_invalid_image(client, content):
    response = client.post("/api/v1/predict", files={"image": ("scan.png", content)})
    assert response.status_code == 422


def test_oversized_image(client):
    response = client.post("/api/v1/predict", files={"image": ("scan.png", b"x" * (MAX_BYTES + 1))})
    assert response.status_code == 413


def test_unsupported_decoded_format(client):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="BMP")
    assert (
        client.post("/api/v1/predict", files={"image": ("scan.png", buffer.getvalue())}).status_code
        == 422
    )


def test_small_image_rejected():
    buffer = io.BytesIO()
    Image.new("RGB", (12, 12)).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="dimensions"):
        decode_image(buffer.getvalue())


@pytest.mark.parametrize(
    "mode,expected", [("ok", 200), ("down", 503), ("timeout", 504), ("bad", 503)]
)
def test_serving_failure_modes(client, monkeypatch, mode, expected):
    serving(monkeypatch, mode)
    response = client.post(
        "/api/v1/predict?explain=false", files={"image": ("scan.png", image_bytes())}
    )
    assert response.status_code == expected
    if expected == 200:
        result = response.json()
        assert set(result["probabilities"]) == set(CLASSES)
        assert sum(result["probabilities"].values()) == pytest.approx(1)
        assert result["inference_time_ms"] >= 0
        assert result["request_id"]


@pytest.mark.parametrize("mode,status", [("ok", 200), ("down", 503)])
def test_readiness_tracks_serving(client, monkeypatch, mode, status):
    serving(monkeypatch, mode)
    assert client.get("/ready").status_code == status


def test_pipeline_disabled(client):
    assert client.post("/api/v1/pipeline/run").status_code == 403


def test_pipeline_rejects_parallel_run(client, monkeypatch):
    monkeypatch.setenv("ENABLE_PIPELINE_CONTROL", "true")
    monkeypatch.setattr(api, "pipeline_process", SimpleNamespace(poll=lambda: None))
    assert client.post("/api/v1/pipeline/run").status_code == 409


def test_empty_operational_states(client, monkeypatch):
    monkeypatch.setattr(api, "RECENT", [])
    assert client.get("/api/v1/models").json()["model"] is None
    assert client.get("/api/v1/pipeline").json()["status"] == "never_run"
    assert client.get("/api/v1/drift").json()["status"] == "insufficient_data"


def test_pipeline_log_missing(client):
    assert client.get("/api/v1/pipeline/log/missing.log").status_code == 404
    assert client.get("/api/v1/pipeline/log/config.yaml").status_code == 400


def test_metrics_available_during_model_outage(client, monkeypatch):
    serving(monkeypatch, "down")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "brainai_model_available 0.0" in response.text


def test_brightness_warning_requires_samples(client, monkeypatch, tmp_path):
    write_json(
        tmp_path / "reports/preparation.json",
        {
            "baseline_brightness_mean": 0.1,
            "baseline_brightness_std": 0.1,
        },
    )
    monkeypatch.setattr(api, "RECENT", [0.9] * 20)
    assert client.get("/api/v1/drift").json()["status"] == "warning"


def test_atomic_artifact_roundtrip(tmp_path):
    path = tmp_path / "nested/result.json"
    write_json(path, {"status": "success"})
    assert read_json(path) == {"status": "success"}
    assert not list(path.parent.glob("*.tmp"))


def test_deterministic_split_and_duplicate_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    (tmp_path / "params.yaml").write_text("data:\n  seed: 42\n  validation_fraction: 0.2\n")
    for split in ["Training", "Testing"]:
        for c, label in enumerate(CLASSES):
            folder = tmp_path / "data/raw" / split / label
            folder.mkdir(parents=True)
            for index in range(10):
                (folder / f"{index}.png").write_bytes(
                    image_bytes(c * 100 + index + (1000 if split == "Testing" else 0))
                )
    duplicate = tmp_path / "data/raw/Training/glioma/duplicate.png"
    duplicate.write_bytes((tmp_path / "data/raw/Testing/glioma/0.png").read_bytes())
    data.split_dataset()
    first = (tmp_path / "data/splits.json").read_bytes()
    data.split_dataset()
    assert first == (tmp_path / "data/splits.json").read_bytes()
    quality = read_json(tmp_path / "reports/data-quality.json")
    assert len(quality["exact_duplicates_removed"]) == 1
    rows = read_json(tmp_path / "data/splits.json")
    assert len({r["pixel_hash"] for r in rows}) == len(rows)
    assert all(
        sum(r["split"] == s and r["label"] == c for r in rows) > 0
        for c in CLASSES
        for s in ["train", "val", "test"]
    )


def test_archive_traversal_rejected(tmp_path, monkeypatch):
    import zipfile

    monkeypatch.setattr(data, "ROOT", tmp_path)
    (tmp_path / "params.yaml").write_text("data: {}\n")
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("../../escape.jpg", "bad")
    with pytest.raises(ValueError, match="Unsafe archive"):
        data.acquire(archive)


def test_pipeline_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 1)
    )
    assert pipeline.run() == 1
    state = read_json(tmp_path / "reports/pipeline.json")
    assert state["status"] == "failed"
    assert state["stages"]["check"]["exit_code"] == 1
    assert state["stages"]["train"]["status"] == "pending"


def test_pyfunc_reload_explanation_and_class_contract(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = build_model()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, checkpoint)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"version": "synthetic-test-only", "experimental": True})
    wrapped = MRIModel()
    wrapped.load_context(
        SimpleNamespace(artifacts={"checkpoint": str(checkpoint), "manifest": str(manifest)})
    )
    row = pd.DataFrame(
        {"image_base64": [base64.b64encode(image_bytes()).decode()], "explain": [True]}
    )
    first = wrapped.predict(None, row).iloc[0]
    second = wrapped.predict(None, row).iloc[0]
    assert first.probabilities_json == second.probabilities_json
    assert decode_image(base64.b64decode(first.gradcam_base64)).size == (224, 224)
    assert len(wrapped.model.features[-1]._forward_hooks) == 0
    torch.save({"state_dict": model.state_dict(), "classes": list(reversed(CLASSES))}, checkpoint)
    with pytest.raises(ValueError, match="class order"):
        load_checkpoint(checkpoint)


def test_mlflow_packaging_rebuilds_runtime_lock(tmp_path):
    import mlflow.pyfunc
    from mlflow.models import infer_signature

    torch.set_num_threads(1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": build_model().state_dict(), "classes": CLASSES}, checkpoint)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"version": "synthetic-packaging-test", "experimental": True})
    row = pd.DataFrame(
        {"image_base64": [base64.b64encode(image_bytes()).decode()], "explain": [False]}
    )
    wrapped = MRIModel()
    artifacts = {"checkpoint": str(checkpoint), "manifest": str(manifest)}
    wrapped.load_context(SimpleNamespace(artifacts=artifacts))
    expected = wrapped.predict(None, row)
    output = tmp_path / "packaged"
    mlflow.pyfunc.save_model(
        str(output),
        python_model=MRIModel(),
        artifacts=artifacts,
        signature=infer_signature(row, expected),
        input_example=row,
        pip_requirements=[],
    )
    actual = mlflow.pyfunc.load_model(str(output)).predict(row)
    assert actual.iloc[0].probabilities_json == expected.iloc[0].probabilities_json


def test_preparation_failure_preserves_completed_output(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(data, "ROOT", tmp_path)
    (tmp_path / "params.yaml").write_text("data: {ray_workers: 2, image_size: 224}\n")
    write_json(tmp_path / "data/splits.json", [{"path": "fixture.png"}])
    previous = tmp_path / "data/processed/train/glioma/previous.jpg"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(image_bytes())
    closed = []

    def fail(_refs):
        raise RuntimeError("worker interrupted")

    fake_ray = SimpleNamespace(
        init=lambda **kw: None,
        shutdown=lambda: closed.append(True),
        remote=lambda fn: SimpleNamespace(remote=lambda *a: None),
        get=fail,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    with pytest.raises(RuntimeError, match="interrupted"):
        data.prepare()
    assert previous.is_file()
    assert closed == [True]


def test_training_rejects_incomplete_prepared_data(tmp_path, monkeypatch):
    from brainai import train

    monkeypatch.setattr(train, "ROOT", tmp_path)
    for split in ["train", "val", "test"]:
        for label in CLASSES:
            folder = tmp_path / "data/processed" / split / label
            folder.mkdir(parents=True)
            (folder / "fixture.png").write_bytes(image_bytes())
    write_json(
        tmp_path / "reports/data-quality.json",
        {
            "counts": {"train": {label: 2 for label in CLASSES}},
        },
    )
    with pytest.raises(ValueError, match="Incomplete prepared train"):
        train.datasets()


def test_acquisition_rejects_wrong_archive_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    (tmp_path / "params.yaml").write_text("data: {archive_sha256: expected-checksum}\n")
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"incorrect archive")
    with pytest.raises(ValueError, match="checksum"):
        data.acquire(archive)
