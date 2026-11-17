"""Download/import, leakage audit, deterministic splits, and Ray image preparation."""

import argparse
import hashlib
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from brainai.common import CLASSES, ROOT, decode_image, read_json, sha256, write_json


def acquire(archive=None):
    cfg = yaml.safe_load((ROOT / "params.yaml").read_text())["data"]
    target = ROOT / "data/raw"
    if (
        target.exists()
        and any(target.rglob("*.jpg"))
        and read_json(ROOT / "data/source.json", {}).get("archive_sha256")
        == cfg.get("archive_sha256")
    ):
        return
    archive = Path(archive) if archive else ROOT / "data/source.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset"
        url += "?datasetVersionNumber=" + str(cfg["dataset_version"])
        urllib.request.urlretrieve(url, archive)
    if cfg.get("archive_sha256") and sha256(archive) != cfg["archive_sha256"]:
        raise ValueError("Archive checksum differs from the pinned dataset version")
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive path")
        source.extractall(target)
    write_json(
        ROOT / "data/source.json",
        {
            "dataset": "masoudnickparvar/brain-tumor-mri-dataset",
            "source_url": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
            "archive_sha256": sha256(archive),
            "downloaded_at_unix": time.time(),
            "license": "CC BY 4.0 (publisher metadata snapshot in reports/dataset-metadata.json)",
            "version": "Identified by immutable downloaded archive SHA256",
        },
    )


def perceptual_hash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def split_dataset():
    cfg = yaml.safe_load((ROOT / "params.yaml").read_text())["data"]
    rows, rejected, duplicates, seen = [], [], [], {}
    # Test wins duplicate ownership: retained training data cannot contain test copies.
    for partition in ["Testing", "Training"]:
        directories = list((ROOT / "data/raw").rglob(partition))
        if len(directories) != 1:
            raise ValueError(f"Expected exactly one {partition} directory")
        for label in CLASSES:
            folder = directories[0] / label
            if not folder.is_dir():
                raise ValueError(f"Missing class directory: {folder}")
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue
                try:
                    image = decode_image(path.read_bytes())
                    digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
                    if digest in seen:
                        if seen[digest]["label"] != label:
                            raise ValueError("Conflicting labels for duplicate image")
                        duplicates.append(
                            {
                                "removed": str(path.relative_to(ROOT)),
                                "retained": seen[digest]["path"],
                            }
                        )
                        continue
                    row = {
                        "path": str(path.relative_to(ROOT)),
                        "label": label,
                        "sha256": sha256(path),
                        "pixel_hash": digest,
                        "phash": perceptual_hash(image),
                        "split": "test" if partition == "Testing" else "train",
                    }
                    seen[digest] = row
                    rows.append(row)
                except (ValueError, OSError) as exc:
                    rejected.append({"path": str(path), "reason": str(exc)})
    train = [r for r in rows if r["split"] == "train"]
    _, validation = train_test_split(
        train,
        test_size=cfg["validation_fraction"],
        random_state=cfg["seed"],
        stratify=[r["label"] for r in train],
    )
    for row in validation:
        row["split"] = "val"
    # Exact matching perceptual hashes flag similarities, not confirmed patient identities.
    buckets, similarities = {}, []
    for row in rows:
        key = str(row["phash"])
        for other in buckets.get(key, []):
            if other["split"] != row["split"]:
                similarities.append([other["path"], row["path"]])
        buckets.setdefault(key, []).append(row)
    counts = {
        split: {c: sum(r["split"] == split and r["label"] == c for r in rows) for c in CLASSES}
        for split in ["train", "val", "test"]
    }
    if any(n == 0 for classes in counts.values() for n in classes.values()):
        raise ValueError("Every split must contain every class")
    write_json(ROOT / "data/splits.json", rows)
    write_json(
        ROOT / "reports/data-quality.json",
        {
            "counts": counts,
            "rejected": rejected,
            "exact_duplicates_removed": duplicates,
            "cross_split_perceptual_matches": similarities,
            "patient_identifiers_available": False,
            "limitation": "Image-level split; perceptual matching cannot prove patient independence",
            "fingerprint": sha256(ROOT / "data/splits.json"),
            "seed": cfg["seed"],
        },
    )


def prepare_one(row, root, size, output_dir="data/processed"):
    root = Path(root)
    image = decode_image((root / row["path"]).read_bytes()).resize((size, size))
    output = root / output_dir / row["split"] / row["label"] / (row["sha256"] + ".jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)
    gray = np.asarray(image.convert("L"), dtype=float) / 255
    return {"mean": float(gray.mean()), "std": float(gray.std())}


def prepare():
    import ray

    cfg = yaml.safe_load((ROOT / "params.yaml").read_text())["data"]
    rows = read_json(ROOT / "data/splits.json")
    staging = ROOT / "data/prepared-next"
    shutil.rmtree(staging, ignore_errors=True)
    started = time.perf_counter()
    ray.init(
        num_cpus=cfg["ray_workers"], include_dashboard=False, object_store_memory=100 * 1024 * 1024
    )
    try:
        worker = ray.remote(prepare_one)
        stats = []
        # Bound pending work and object-store pressure on small machines.
        for offset in range(0, len(rows), 64):
            refs = [
                worker.remote(row, str(ROOT), cfg["image_size"], "data/prepared-next")
                for row in rows[offset : offset + 64]
            ]
            stats.extend(ray.get(refs))
    finally:
        ray.shutdown()
    # Publish only a complete preparation result. A killed worker leaves staging recoverable.
    if len(list(staging.rglob("*.jpg"))) != len(rows):
        raise RuntimeError("Prepared-image count does not match the split manifest")
    shutil.rmtree(ROOT / "data/processed", ignore_errors=True)
    staging.rename(ROOT / "data/processed")
    duration = time.perf_counter() - started
    train_stats = [s for r, s in zip(rows, stats) if r["split"] == "train"]
    write_json(
        ROOT / "reports/preparation.json",
        {
            "engine": "Ray",
            "workers": cfg["ray_workers"],
            "processed": len(rows),
            "seconds": duration,
            "images_per_second": len(rows) / duration,
            "baseline_brightness_mean": float(np.mean([s["mean"] for s in train_stats])),
            "baseline_brightness_std": float(np.std([s["mean"] for s in train_stats])),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["acquire", "split", "prepare"])
    parser.add_argument("--archive")
    args = parser.parse_args()
    {"acquire": lambda: acquire(args.archive), "split": split_dataset, "prepare": prepare}[
        args.stage
    ]()
