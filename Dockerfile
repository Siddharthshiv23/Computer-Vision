# CPU-only inference image for the app, compatible with Hugging Face Spaces (Docker SDK).
#   docker build -t hybrid-isp .
#   docker run --rm -p 7860:7860 hybrid-isp
FROM python:3.11-slim

# OpenMP runtime needed by onnxruntime and torch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False
WORKDIR /home/user/app

# Two installs on purpose: the CPU index applies to torch only.
COPY --chown=user requirements.txt requirements-torch-cpu.txt ./
RUN pip install --no-cache-dir --user -r requirements-torch-cpu.txt \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user isp ./isp
COPY --chown=user app ./app
COPY --chown=user weights ./weights

EXPOSE 7860
CMD ["python", "-m", "app.app"]
