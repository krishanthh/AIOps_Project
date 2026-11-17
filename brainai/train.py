"""CPU transfer learning with validation-only model selection and MLflow lineage."""

import copy
import random
import time

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder

from brainai.common import CLASSES, ROOT, git_commit, read_json, sha256, write_json
from brainai.model import build_model, transform


def datasets():
    result = {}
    expected = read_json(ROOT / "reports/data-quality.json", {}).get("counts", {})
    for split in ["train", "val", "test"]:
        dataset = ImageFolder(ROOT / "data/processed" / split, transform=transform())
        if dataset.classes != CLASSES:
            raise ValueError("Dataset label ordering does not match model contract")
        if split in expected:
            actual = {label: dataset.targets.count(i) for i, label in enumerate(CLASSES)}
            if actual != expected[split]:
                raise ValueError(f"Incomplete prepared {split} data; rerun DVC preparation")
        result[split] = dataset
    return result


def infer(model, loader):
    labels, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            probabilities.extend(model(images).softmax(1).tolist())
            labels.extend(targets.tolist())
    return np.array(labels), np.array(probabilities)


def train():
    cfg = yaml.safe_load((ROOT / "params.yaml").read_text())
    if cfg["train"]["fine_tune_epochs"] != 0:
        raise ValueError("Cached-backbone training requires fine_tune_epochs=0")
    seed = cfg["data"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    data = datasets()
    mlflow.set_experiment("brain-mri-classification")
    # A deterministic frozen backbone can be evaluated once instead of every epoch.
    # This preserves head-training gradients while making CPU experiments practical.
    extractor = build_model(pretrained=True).eval()
    cache_started = time.perf_counter()
    cached = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            embeddings, targets = [], []
            for images, labels in DataLoader(data[split], batch_size=8, shuffle=False):
                features = extractor.avgpool(extractor.features(images)).flatten(1)
                embeddings.append(features)
                targets.append(labels)
            cached[split] = DataLoader(
                TensorDataset(torch.cat(embeddings), torch.cat(targets)),
                batch_size=32,
                shuffle=split == "train",
            )
            print(f"Cached {split} embeddings", flush=True)
    cache_seconds = time.perf_counter() - cache_started
    del extractor
    best_score, selected = -1, None
    comparisons = []
    for lr in cfg["train"]["learning_rates"]:
        torch.manual_seed(seed)
        model = build_model(pretrained=True)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss()
        started = time.perf_counter()
        with mlflow.start_run(run_name=f"mobilenet-lr-{lr}") as run:
            mlflow.log_params(
                {
                    "architecture": "mobilenet_v3_small",
                    "lr": lr,
                    "epochs": cfg["train"]["epochs"],
                    "seed": seed,
                    "batch_size": 32,
                    "feature_extraction_batch_size": 8,
                    "feature_cache_seconds": cache_seconds,
                    "git_commit": git_commit(),
                    "dataset_fingerprint": sha256(ROOT / "data/splits.json"),
                    "image_size": 224,
                    "backbone": "frozen",
                    "device": "cpu",
                }
            )
            run_best, run_state = -1, None
            for epoch in range(cfg["train"]["epochs"]):
                model.train()
                model.features.eval()  # Freeze batch-normalization statistics as well as weights.
                losses = []
                for images, targets in cached["train"]:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model.classifier(images), targets)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                labels, probabilities = infer(model.classifier, cached["val"])
                score = f1_score(labels, probabilities.argmax(1), average="macro")
                mlflow.log_metrics(
                    {
                        "train_loss": float(np.mean(losses)),
                        "validation_macro_f1": score,
                        "validation_accuracy": accuracy_score(labels, probabilities.argmax(1)),
                    },
                    step=epoch,
                )
                print(f"lr={lr} epoch={epoch + 1} val_macro_f1={score:.4f}", flush=True)
                if score > run_best:
                    run_best, run_state = score, copy.deepcopy(model.state_dict())
            duration = time.perf_counter() - started
            mlflow.log_metrics({"best_validation_macro_f1": run_best, "training_seconds": duration})
            mlflow.log_artifact(str(ROOT / "reports/data-quality.json"))
            mlflow.log_artifact(str(ROOT / "params.yaml"))
            comparisons.append(
                {
                    "run_id": run.info.run_id,
                    "learning_rate": lr,
                    "validation_macro_f1": run_best,
                    "seconds": duration,
                }
            )
            if run_best > best_score:
                best_score = run_best
                selected = {
                    "state_dict": run_state,
                    "classes": CLASSES,
                    "run_id": run.info.run_id,
                    "validation_macro_f1": run_best,
                }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    torch.save(selected, ROOT / "artifacts/checkpoint.pt")
    write_json(ROOT / "reports/experiments.json", comparisons)
    write_json(
        ROOT / "artifacts/manifest.json",
        {
            "version": selected["run_id"],
            "run_id": selected["run_id"],
            "git_commit": git_commit(),
            "dataset_fingerprint": sha256(ROOT / "data/splits.json"),
            "classes": CLASSES,
            "image_size": 224,
            "validation_macro_f1": best_score,
            "experimental": best_score < cfg["train"]["promotion_macro_f1"],
            "source": read_json(ROOT / "data/source.json"),
        },
    )


if __name__ == "__main__":
    train()
