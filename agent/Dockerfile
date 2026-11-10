# The base image is the one part a rebuild can still change. To freeze it, replace the tag with name@digest:
#   docker pull python:3.12-slim
#   docker image inspect python:3.12-slim --format "{{index .RepoDigests 0}}"
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# requirements.txt is a lock file (every package pinned, with hashes): pip refuses anything that differs. Change it with README > Updating dependencies.
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
COPY agent.py auth.py models.py sdmodels.py ./
COPY static ./static
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
