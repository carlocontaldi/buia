# CogVideoX-2b text-to-video (Hugging Face Diffusers, PyTorch). stable-diffusion.cpp has no CogVideoX support, so this is its own service.
# The base image is the one part a rebuild can still change. To freeze it, replace the tag with name@digest (see agent/Dockerfile).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# requirements.txt is a lock file (every package pinned, with hashes): pip refuses anything that differs. Change it with README > Updating dependencies.
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
COPY server.py generate.py ./
# The model files come from the Models panel (sdmodels/): never let the libraries reach out to Hugging Face on their own.
ENV HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
CMD ["python", "server.py"]
