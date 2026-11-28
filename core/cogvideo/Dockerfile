# CogVideoX-2b text-to-video (Hugging Face Diffusers, PyTorch). stable-diffusion.cpp has no CogVideoX support, so this is its own service.
# The base image is pinned by digest, the same one as agent/Dockerfile: change both together (the how-to is there).
FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
WORKDIR /app
COPY requirements.txt .
# requirements.txt is a lock file (every package pinned, with hashes): pip refuses anything that differs. Change it with README > Updating dependencies.
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
COPY server.py generate.py ./
# The model files come from the Settings panel (sdmodels/): never let the libraries reach out to Hugging Face on their own.
ENV HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
CMD ["python", "server.py"]
