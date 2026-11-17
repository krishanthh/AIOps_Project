"""One preprocessing and inference implementation for training, evaluation, and serving."""

import base64
import io
import json
import os
import threading

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from brainai.common import CLASSES, decode_image, read_json


def transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_model(pretrained=False):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    return model


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("Model class order differs from application contract")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def explain(model, tensor, image, class_index):
    """Grad-CAM from last convolutional activations, with hook cleanup per request."""
    activations = []
    hook = model.features[-1].register_forward_hook(lambda _m, _i, out: activations.append(out))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor.detach().requires_grad_(True))
        gradient = torch.autograd.grad(logits[0, class_index], activations[0])[0]
        cam = (gradient.mean((2, 3), keepdim=True) * activations[0]).sum(1).relu()
        cam = (
            torch.nn.functional.interpolate(
                cam.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
            )[0, 0]
            .detach()
            .numpy()
        )
        cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-8)
        rgb = np.asarray(image.resize((224, 224)), dtype=float) / 255
        heat = np.stack([cam, 0.25 * (1 - cam), 1 - cam], axis=-1)
        overlay = Image.fromarray(np.uint8(np.clip(0.6 * rgb + 0.4 * heat, 0, 1) * 255))
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


class MRIModel(mlflow.pyfunc.PythonModel):
    """String/boolean dataframe input preserves raw-image preprocessing inside the artifact."""

    def __getstate__(self):
        # MLflow validates examples before serialization; runtime locks must be rebuilt.
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"lock", "model", "manifest"}
        }

    def load_context(self, context):
        torch.set_num_threads(2)
        self.model = load_checkpoint(context.artifacts["checkpoint"])
        self.manifest = read_json(context.artifacts["manifest"])
        self.lock = threading.Lock()
        if os.getenv("SERVING_METRICS", "false") == "true":
            from prometheus_client import start_http_server

            start_http_server(8002)

    def predict(self, context, model_input, params=None):
        output = []
        with self.lock:
            for row in model_input.to_dict(orient="records"):
                image = decode_image(base64.b64decode(row["image_base64"], validate=True))
                tensor = transform()(image).unsqueeze(0)
                with torch.no_grad():
                    probabilities = self.model(tensor).softmax(1)[0].tolist()
                index = int(np.argmax(probabilities))
                cam = explain(self.model, tensor, image, index) if row["explain"] else ""
                output.append(
                    {
                        "predicted_class": CLASSES[index],
                        "confidence": probabilities[index],
                        "probabilities_json": json.dumps(dict(zip(CLASSES, probabilities))),
                        "gradcam_base64": cam,
                        "model_version": self.manifest["version"],
                        "experimental": self.manifest["experimental"],
                    }
                )
        return pd.DataFrame(output)
