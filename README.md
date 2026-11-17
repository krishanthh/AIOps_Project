# BrainScope — Brain Tumour Classification with AIOps

**Krishanthh Mohan · DA24B044 · DA3408**

BrainScope predicts glioma, meningioma, pituitary tumour, or no tumour from a JPEG/PNG brain MRI image and provides a Grad-CAM visual explanation.

**Video demonstration:** [Watch the application recording](https://drive.google.com/drive/folders/1Q4nircJrvh6BINYBafkLjs410FEqFl9O).

I use React, FastAPI, PyTorch MobileNetV3 Small, Ray, DVC, MLflow Projects/serving, Docker Compose, Prometheus, and Grafana. The implementation runs locally. The [test report](reports/TEST_REPORT.md) records the executed checks, measured results, and limitations.

## Start the application (Docker Compose)

Docker Compose is my primary startup and demonstration method. From this folder, with the packaged model present:

```bash
test -f .env || cp .env.example .env
docker compose build backend frontend
docker compose up -d --no-build --wait --wait-timeout 180
docker compose ps
curl --fail http://localhost:8000/ready
```

Open **http://localhost:3100** for the app and **http://localhost:8000/docs** for the API. Tracking and monitoring start with the same Compose command. If readiness fails, inspect `docker compose logs serving backend`. Stop the stack with `docker compose down`; this preserves the named monitoring volumes.

The host needs sufficient disk space for dependencies, data, DVC cache, and container images. The verified platform is Linux ARM64 with CPU inference. Services have explicit memory limits. Once images are built, use the `up --no-build` command to start the application without rebuilding. Run full training separately from the serving stack on a small machine.

## Reproduce training

Use the training container with the same application image as serving:

```bash
mkdir -p ../.brainai-dvc-remote ../.brainai-model-cache
docker compose stop frontend frontend-exporter backend serving prometheus grafana
docker compose build backend
docker compose up -d mlflow
docker compose --profile training run --rm trainer
```

The trainer is capped at 1.5 GiB and mounts this project so Git, DVC outputs, logs, and reports persist. Start the serving stack again after training with the documented `up --no-build` command. On a memory-limited machine, stop the other project services before training. `MLproject` also exposes the pipeline for coursework reproducibility; it is not a separate application startup method.

DVC is initialized in this project. The local remote is configured under `../.brainai-dvc-remote`. On a copied fresh project, initialize Git/DVC if their metadata is absent and configure a local remote before `dvc push`:

```bash
git init
dvc init
dvc remote add -d local ../.brainai-dvc-remote
```

The submission bundle includes the prepared images and packaged model. If raw data is absent when reproducing training, ingestion downloads and verifies the pinned archive.

The pipeline downloads the public [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar automatically. Publisher metadata declares CC BY 4.0; the source metadata is saved in `reports/dataset-metadata.json`. If the publisher requires manual access, download its ZIP and run `python -m brainai.data acquire --archive /absolute/path/to/dataset.zip` before reproducing the remaining stages. Never put a dataset token into Git.

`params.yaml` controls splits and training. The baseline uses a frozen ImageNet backbone and cached deterministic features, then compares two classifier learning rates over five epochs. Validation selects the best model; the test partition supplies the held-out report. The saved artifact includes an MLflow signature, preprocessing code, checkpoint, class order, and model version.

## Select a model version or roll back

I select the packaged model directory using `MODEL_ARTIFACT_PATH` in `.env`. The artifact includes its weights, preprocessing, class order, and MLflow signature.

Before an update, check `artifacts/manifest.json`, `reports/experiments.json`, and `reports/evaluation.json`. The configured validation target remains unchanged; below-target models are explicitly experimental.

To preserve a packaged model, use its run ID as the version-directory name:

```bash
MODEL_VERSION=$(docker compose run --rm --no-deps backend python -c 'import json; print(json.load(open("artifacts/manifest.json"))["version"])')
mkdir -p artifacts/versions
# Copy only if this version has not already been saved.
if [ ! -d "artifacts/versions/$MODEL_VERSION" ]; then
  cp -R artifacts/model "artifacts/versions/$MODEL_VERSION"
fi
```

Set the following entry in `.env`, replacing `RUN_ID` with that version's actual run ID:

```text
MODEL_ARTIFACT_PATH=artifacts/versions/RUN_ID
```

Apply the change and check readiness:

```bash
docker compose up -d --no-deps --force-recreate --wait --wait-timeout 120 serving
curl --fail http://localhost:8000/ready
```

Use **recreate**, because a plain restart does not apply changes to the Compose model path. Confirm the version in a real prediction response. For rollback, put the previous saved directory back in `.env` and repeat these two commands. Keep saved version directories unchanged; a rollback demonstration requires two saved versions.

## Service addresses

| Service | Local address |
|---|---|
| Frontend | http://localhost:3100 |
| FastAPI / OpenAPI | http://localhost:8000/docs |
| MLflow tracking | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3101 |

Grafana's local demo user is `admin`; its password comes from `.env`. Dashboards and the Prometheus datasource are provisioned automatically. Published ports bind to localhost. The inference model runs behind a separate MLflow serving service, reached by FastAPI over the Compose network.

Containers share a pinned application image. `MLproject` and `python_env.yaml` describe the MLflow Project environment; `requirements-lock.txt` records the exact resolved Python versions.

## Python environment for code checks

Application startup and training use Compose. A local Python environment is useful for the Git hook and code checks; it lives outside the submission folder:

```bash
uv python install 3.11
uv venv --python 3.11 ../.venv
uv pip install --python ../.venv/bin/python -r requirements-lock.txt
. ../.venv/bin/activate
```

## Tests and evidence

```bash
git config core.hooksPath .githooks
sh scripts/ci.sh --checks-only
docker compose config --quiet
python -m tests.integration_check
```

Run these checks in the activated Python environment. The pre-commit hook runs the code/test gate automatically. The integration check requires the complete Compose stack and prepared test images. Generated reports distinguish fixture-based tests from held-out model performance.

The full DVC pipeline includes the same test gate before training. For a complete CI reproduction, stop serving as described in the training section, keep tracking running, and execute `docker compose --profile training run --rm trainer sh scripts/ci.sh`. The normal `trainer` command also records each pipeline stage for the console.

- [Architecture and high-level design](docs/HLD.md)
- [Endpoint definitions and low-level design](docs/LLD.md)
- [Test plan and acceptance criteria](docs/TEST_PLAN.md)
- [User manual](docs/USER_MANUAL.md)
- [AI disclosure](AI_DISCLOSURE.md)

## Limits

I evaluate a public image dataset rather than a clinical patient cohort. Patient identifiers are unavailable, perceptual checks are incomplete safeguards against leakage, and model scores are not calibrated clinical confidence. Grad-CAM is not segmentation. Brightness drift is a limited distribution check, not accuracy monitoring. The app is a single-user coursework demonstration without patient records or internet-facing authentication. Actual measured outcomes and remaining work are stated in `reports/TEST_REPORT.md`.
