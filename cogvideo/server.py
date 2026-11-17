"""CogVideoX-2b behind the same small HTTP API the agent already uses for stable-diffusion.cpp's sd-server (only what the agent calls):
  GET  /sdcpp/v1/capabilities            -> {"supported_modes": ["vid_gen"], "model": {"stem": "CogVideoX-2b"}}
  POST /sdcpp/v1/vid_gen                 -> 202 {"poll_url": "/sdcpp/v1/jobs/<id>"}   (409 while another job runs)
  GET  /sdcpp/v1/jobs/<id>               -> {"status": "running" | "completed" | "failed", "result": {..., "result_url": ...} | "error": "..."}
                                            a running job also has "progress": {"phase": "text" | "loading" | "sampling" | "decoding" | "saving", "step": i, "steps": N}
  GET  /sdcpp/v1/jobs/<id>/result        -> the finished video (video/webm); the result_url of a completed job
Each job runs generate.py in a fresh process, so this server itself never touches the GPU and idles at no VRAM. No imports beyond the
standard library on purpose: torch and diffusers are only loaded by the job.
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT, FRAMES, FPS = 720, 480, 49, 8   # fixed by the model; same values as generate.py
MIN_STEPS, MAX_STEPS, DEFAULT_STEPS = 1, 50, 50
MIN_GUIDANCE, MAX_GUIDANCE, DEFAULT_GUIDANCE = 1.0, 20.0, 6.0   # the default is the model card's value, same as generate.py
MAX_NEGATIVE = 2000                            # characters
JOBS: dict[str, dict] = {}                     # id -> {"status", "result" | "error"}; only the newest few are kept
JOB_FILES: dict[str, Path] = {}                # id -> finished video file in the temp folder; kept exactly as long as the job is
LOCK = threading.Lock()


def run_job(jid: str, body: dict) -> None:
    out = Path(tempfile.gettempdir()) / f"{jid}.webm"
    job_file = out.with_suffix(".json")
    job_file.write_text(json.dumps(body | {"out": str(out)}), encoding="utf-8")
    try:
        rc = subprocess.run([sys.executable, str(HERE / "generate.py"), str(job_file)]).returncode
        err = Path(str(out) + ".error")
        if rc == 0 and out.exists():
            result = {"output_format": "webm", "frame_count": FRAMES, "fps": FPS, "result_url": f"/sdcpp/v1/jobs/{jid}/result"}
            with LOCK:
                JOB_FILES[jid] = out  # the file first: a poll that sees "completed" must find it
                JOBS[jid] = {"status": "completed", "result": result}
        else:
            msg = err.read_text(encoding="utf-8") if err.exists() else f"the generation process ended with code {rc}" + \
                  (" (killed: probably out of memory, see README)" if rc in (-9, 137) else "")
            JOBS[jid] = {"status": "failed", "error": msg}
    finally:
        if jid not in JOB_FILES:  # a finished video stays until its job is dropped from JOBS
            out.unlink(missing_ok=True)
        for f in (job_file, Path(str(out) + ".error"), Path(str(out) + ".progress"), Path(str(out) + ".progress.tmp")):
            f.unlink(missing_ok=True)


def read_progress(jid: str) -> dict | None:
    """What generate.py last reported for a running job, or None (not started yet, or unreadable)."""
    try:
        p = json.loads((Path(tempfile.gettempdir()) / f"{jid}.webm.progress").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return p if isinstance(p, dict) else None


def check(req: dict) -> tuple[dict | None, str]:
    """Returns (job for generate.py, "") or (None, reason)."""
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "empty prompt"
    for key, want in (("width", WIDTH), ("height", HEIGHT), ("video_frames", FRAMES)):
        if req.get(key, want) != want:
            return None, f"CogVideoX-2b makes {WIDTH}x{HEIGHT} videos of {FRAMES} frames only ({key} was {req[key]})"
    if req.get("output_format", "webm") != "webm":
        return None, "output_format must be webm"
    sample_params = req.get("sample_params")
    if sample_params is not None and not isinstance(sample_params, dict):
        return None, "sample_params must be an object"
    steps = (sample_params or {}).get("sample_steps", DEFAULT_STEPS)
    if not isinstance(steps, int) or not MIN_STEPS <= steps <= MAX_STEPS:
        return None, f"sample_steps must be an integer from {MIN_STEPS} to {MAX_STEPS}"
    negative = req.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative) > MAX_NEGATIVE:
        return None, f"negative_prompt must be text of at most {MAX_NEGATIVE} characters"
    guidance_params = (sample_params or {}).get("guidance")
    if guidance_params is not None and not isinstance(guidance_params, dict):
        return None, "sample_params.guidance must be an object"
    guidance = (guidance_params or {}).get("txt_cfg", DEFAULT_GUIDANCE)
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not MIN_GUIDANCE <= guidance <= MAX_GUIDANCE:
        return None, f"guidance.txt_cfg must be a number from {MIN_GUIDANCE} to {MAX_GUIDANCE}"
    seed = req.get("seed", -1)
    if not isinstance(seed, int):
        return None, "seed must be an integer"
    return {"prompt": prompt.strip(), "negative": negative.strip(), "guidance": float(guidance), "steps": steps,
            "seed": seed if seed >= 0 else random.randrange(2**32)}, ""


class Handler(BaseHTTPRequestHandler):
    def reply(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/sdcpp/v1/capabilities":
            return self.reply(200, {"supported_modes": ["vid_gen"], "model": {"stem": "CogVideoX-2b"}})
        if self.path.startswith("/sdcpp/v1/jobs/"):
            jid, _, tail = self.path[len("/sdcpp/v1/jobs/"):].partition("/")
            if tail == "result":
                return self.send_result(jid)
            if not tail and (job := JOBS.get(jid)):
                if job["status"] == "running" and (progress := read_progress(jid)):
                    job = job | {"progress": progress}
                return self.reply(200, job)
        self.reply(404, {"error": "not found"})

    def send_result(self, jid: str) -> None:
        with LOCK:  # opened under the lock so that dropping the job cannot delete the file in between; the download runs outside it,
            path = JOB_FILES.get(jid)  # so a client that stops reading cannot hold up the next job
            try:
                f = path.open("rb") if path else None
            except OSError:
                f = None
        if not f:
            return self.reply(404, {"error": "no video for this job"})
        with f:
            self.send_response(200)
            self.send_header("Content-Type", "video/webm")
            self.send_header("Content-Length", str(os.fstat(f.fileno()).st_size))
            self.end_headers()
            try:
                shutil.copyfileobj(f, self.wfile, 64 * 1024)
            except (BrokenPipeError, ConnectionResetError):  # the client left; nobody to tell
                pass

    def do_POST(self):
        if self.path != "/sdcpp/v1/vid_gen":
            return self.reply(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self.reply(400, {"error": "body is not JSON"})
        job, why = check(req) if isinstance(req, dict) else (None, "body must be a JSON object")
        if not job:
            return self.reply(400, {"error": why})
        with LOCK:
            if any(j["status"] == "running" for j in JOBS.values()):
                return self.reply(409, {"error": "a video is already being made; wait for it"})
            jid = uuid.uuid4().hex
            for old in list(JOBS)[:-3]:  # keep the newest few results
                del JOBS[old]
                if old_file := JOB_FILES.pop(old, None):
                    old_file.unlink(missing_ok=True)  # a download that already opened it keeps reading
            JOBS[jid] = {"status": "running"}
        threading.Thread(target=run_job, args=(jid, job), daemon=True).start()
        self.reply(202, {"id": jid, "poll_url": f"/sdcpp/v1/jobs/{jid}"})

    def log_message(self, fmt, *args):  # polls every 2 s would fill the log
        pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "1234"))
    print(f"cogvideo: listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
