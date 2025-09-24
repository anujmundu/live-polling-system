# Multi-Stage Dockerfile for Automated Retraining Pipeline Job
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim as runner

WORKDIR /app

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash appuser

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY src/ /app/src/
COPY training/ /app/training/
COPY pipelines/ /app/pipelines/
COPY models/ /app/models/

RUN chown -R appuser:appgroup /app
USER appuser

ENTRYPOINT ["python", "pipelines/retraining_pipeline.py"]
