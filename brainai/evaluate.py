"""Held-out evaluation, explanation examples, and signed MLflow packaging."""

import base64
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader

from brainai.common import ROOT, read_json, write_json
from brainai.model import MRIModel, load_checkpoint
from brainai.train import datasets, infer


def evaluate():
    torch.set_num_threads(2)
    manifest = read_json(ROOT / "artifacts/manifest.json")
    model = load_checkpoint(ROOT / "artifacts/checkpoint.pt")
    data = datasets()
    started = time.perf_counter()
    labels, probabilities = infer(model, DataLoader(data["test"], batch_size=8))
    predicted = probabilities.argmax(1)
    report = classification_report(
        labels, predicted, target_names=manifest["classes"], output_dict=True, zero_division=0
    )
    report["seconds"] = time.perf_counter() - started
    report["samples"] = len(labels)
    write_json(ROOT / "reports/evaluation.json", report)
    ConfusionMatrixDisplay.from_predictions(labels, predicted, display_labels=manifest["classes"])
    plt.tight_layout()
    plt.savefig(ROOT / "reports/confusion-matrix.png", dpi=150)
    plt.close()
    with mlflow.start_run(run_id=manifest["run_id"]):
        mlflow.log_metrics(
            {"test_accuracy": report["accuracy"], "test_macro_f1": report["macro avg"]["f1-score"]}
        )
        mlflow.log_artifact(str(ROOT / "reports/evaluation.json"))
        mlflow.log_artifact(str(ROOT / "reports/confusion-matrix.png"))


def package():
    manifest = read_json(ROOT / "artifacts/manifest.json")
    sample = next((ROOT / "data/processed/test").rglob("*.jpg"))
    example = pd.DataFrame(
        {"image_base64": [base64.b64encode(sample.read_bytes()).decode()], "explain": [True]}
    )
    from types import SimpleNamespace

    artifacts = {
        "checkpoint": str(ROOT / "artifacts/checkpoint.pt"),
        "manifest": str(ROOT / "artifacts/manifest.json"),
    }
    predictor = MRIModel()
    predictor.load_context(SimpleNamespace(artifacts=artifacts))
    predictions = predictor.predict(None, example)
    output = ROOT / "artifacts/model"
    shutil.rmtree(output, ignore_errors=True)
    requirements = [
        "mlflow==2.22.0",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "Pillow==11.2.1",
        "numpy==2.2.6",
        "pandas==2.2.3",
    ]
    mlflow.pyfunc.save_model(
        str(output),
        python_model=MRIModel(),
        artifacts=artifacts,
        code_paths=[str(ROOT / "brainai")],
        signature=infer_signature(example, predictions),
        input_example=example,
        pip_requirements=requirements,
    )
    with mlflow.start_run(run_id=manifest["run_id"]):
        mlflow.log_artifacts(str(output), artifact_path="model")
        for label in manifest["classes"]:
            path = next((ROOT / "data/processed/test" / label).rglob("*.jpg"))
            row = pd.DataFrame(
                {"image_base64": [base64.b64encode(path.read_bytes()).decode()], "explain": [True]}
            )
            result = predictor.predict(None, row).iloc[0]
            cam_path = ROOT / "reports" / f"gradcam-{label}.png"
            cam_path.write_bytes(base64.b64decode(result["gradcam_base64"]))
            mlflow.log_artifact(str(cam_path), artifact_path="explanations")
    write_json(
        ROOT / "reports/packaging.json",
        {
            "version": manifest["version"],
            "path": "artifacts/model",
            "signed": True,
            "experimental": manifest["experimental"],
        },
    )


if __name__ == "__main__":
    import sys

    {"evaluate": evaluate, "package": package}[sys.argv[1]]()
