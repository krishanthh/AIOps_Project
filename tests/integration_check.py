"""Real Compose acceptance checks, separate from synthetic unit-test fixtures."""

import base64
import json
import io
import subprocess
import time

import httpx
from dotenv import dotenv_values
from PIL import Image

from brainai.common import ROOT, read_json, write_json


def container_python(code, payload, isolated=False):
    """Exercise internal serving without publishing another host port."""
    if isolated:
        command = [
            "docker",
            "compose",
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "-e",
            "SERVING_METRICS=false",
            "--entrypoint",
            "python",
            "serving",
            "-c",
            code,
        ]
    else:
        command = ["docker", "compose", "exec", "-T", "serving", "python", "-c", code]
    result = subprocess.run(
        command, input=json.dumps(payload), capture_output=True, text=True, check=True, timeout=120
    )
    return json.loads(result.stdout)


def main():
    sample = next((ROOT / "data/processed/test").rglob("*.jpg"))
    selected = dotenv_values(ROOT / ".env").get("MODEL_ARTIFACT_PATH", "artifacts/model")
    manifest = read_json(ROOT / "artifacts/manifest.json")
    results = []
    with httpx.Client(timeout=90) as client:
        client.get("http://127.0.0.1:8000/ready").raise_for_status()
        results.append({"case": "real-serving-readiness", "passed": True})
        elapsed = []
        for explain in [False, True]:
            started = time.perf_counter()
            response = client.post(
                f"http://127.0.0.1:3100/api/api/v1/predict?explain={str(explain).lower()}",
                files={"image": (sample.name, sample.read_bytes(), "image/jpeg")},
            )
            response.raise_for_status()
            prediction = response.json()
            assert abs(sum(prediction["probabilities"].values()) - 1) < 0.001
            assert bool(prediction["gradcam_base64"]) == explain
            assert prediction["model_version"] == manifest["version"]
            seconds = time.perf_counter() - started
            elapsed.append(seconds)
            results.append(
                {
                    "case": f"compose-prediction-explain-{explain}",
                    "passed": True,
                    "seconds": seconds,
                    "version": prediction["model_version"],
                }
            )
        invalid = container_python(
            "import sys,json,httpx; p=json.load(sys.stdin); "
            "r=httpx.post('http://127.0.0.1:5001/invocations',json=p); "
            "print(json.dumps({'status':r.status_code}))",
            {"dataframe_records": [{"wrong_column": "invalid"}]},
        )
        assert invalid["status"] == 400
        results.append({"case": "mlflow-signature-rejects-invalid-input", "passed": True})
        invalid_upload = client.post(
            "http://127.0.0.1:3100/api/api/v1/predict",
            files={"image": ("invalid.png", b"not an image")},
        )
        assert invalid_upload.status_code == 422
        results.append({"case": "compose-rejects-invalid-upload", "passed": True})
        # These generated boundary images test resource limits, not medical accuracy.
        for size, expected in [((4000, 4000), 200), ((4001, 4000), 422)]:
            buffer = io.BytesIO()
            Image.new("L", size, 80).save(buffer, format="PNG")
            boundary = client.post(
                "http://127.0.0.1:3100/api/api/v1/predict?explain=false",
                files={"image": ("pixel-boundary.png", buffer.getvalue(), "image/png")},
            )
            assert boundary.status_code == expected
            results.append(
                {
                    "case": f"pixel-limit-{size[0]}x{size[1]}",
                    "passed": True,
                    "status": boundary.status_code,
                }
            )
        metrics = client.get("http://127.0.0.1:8000/metrics")
        assert "brainai_predictions_total" in metrics.text
        results.append({"case": "real-prediction-instrumented", "passed": True})
        frontend = client.get("http://127.0.0.1:3100/")
        frontend.raise_for_status()
        assert 'id="root"' in frontend.text
        assert client.get("http://127.0.0.1:3100/api/health").json()["status"] == "ok"
        results.append({"case": "built-frontend-and-rest-proxy", "passed": True})
    # A separate capped container avoids adding another model to the live serving process.
    reload_result = container_python(
        "import sys,json,gc,mlflow.pyfunc,pandas as pd; p=json.load(sys.stdin); "
        "row=pd.DataFrame({'image_base64':[p['image']], 'explain':[False]}); "
        "a=mlflow.pyfunc.load_model(p['path']); first=a.predict(row).iloc[0].probabilities_json; "
        "del a; gc.collect(); b=mlflow.pyfunc.load_model(p['path']); "
        "second=b.predict(row).iloc[0].probabilities_json; "
        "print(json.dumps({'equal':first==second}))",
        {"path": selected, "image": base64.b64encode(sample.read_bytes()).decode()},
        isolated=True,
    )
    assert reload_result["equal"]
    results.append({"case": "real-artifact-reload-prediction-parity", "passed": True})
    results.append(
        {
            "case": "prediction-latency-under-ten-seconds",
            "passed": max(elapsed) <= 10,
            "maximum_seconds": max(elapsed),
        }
    )
    write_json(
        ROOT / "reports/integration.json",
        {
            "dataset": read_json(ROOT / "data/source.json"),
            "cases": results,
            "model_version": manifest["version"],
            "latency_target_seconds": 10,
            "note": "Executed through Nginx, FastAPI and MLflow in actual Compose containers",
        },
    )
    print(json.dumps(results, indent=2))
    if not all(case["passed"] for case in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
