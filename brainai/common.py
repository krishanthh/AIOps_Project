"""Shared paths, atomic artifacts, bounded image validation and model label contract."""

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(os.environ.get("BRAINAI_ROOT", Path(__file__).resolve().parents[1]))
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def decode_image(content):
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Upload must contain an image no larger than 10 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only decoded JPEG and PNG images are supported")
            if image.width * image.height > MAX_PIXELS or min(image.size) < 32:
                raise ValueError("Image dimensions must be at least 32 pixels; maximum 16 MP")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (Image.DecompressionBombError, OSError, SyntaxError) as exc:
        raise ValueError("Invalid or unsafe image") from exc
