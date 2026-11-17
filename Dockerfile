FROM python:3.11.16-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* && git config --system --add safe.directory /workspace
RUN pip install --no-cache-dir uv==0.7.13
COPY requirements-lock.txt ./
RUN uv pip install --system --no-cache -r requirements-lock.txt && uv pip check --system
RUN useradd --uid 1000 --create-home --shell /bin/sh brainai
COPY brainai ./brainai
COPY scripts ./scripts
COPY tests ./tests
COPY requirements.txt params.yaml dvc.yaml MLproject python_env.yaml pyproject.toml ./
CMD ["uvicorn", "brainai.api:app", "--host", "0.0.0.0", "--port", "8000"]
