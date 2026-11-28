# The base image is pinned by digest (Python 3.12.16 on Debian trixie, built 2026-11-24), so a rebuild starts from the same one. To move to a newer build,
# replace the digest with the "Digest:" line of  docker buildx imagetools inspect python:3.12-slim  (and in cogvideo/Dockerfile too).
FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
WORKDIR /app
COPY requirements.txt .
# requirements.txt is a lock file (every package pinned, with hashes): pip refuses anything that differs. Change it with README > Updating dependencies.
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
COPY agent.py auth.py models.py sdmodels.py videogen.py VERSION ./
COPY static ./static
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
