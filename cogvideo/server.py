"""CogVideoX-2b behind the same small HTTP API the agent already uses for stable-diffusion.cpp's sd-server (only what the agent calls):
  GET  /sdcpp/v1/capabilities            -> {"supported_modes": ["vid_gen"], "model": {"stem": "CogVideoX-2b"}}
  POST /sdcpp/v1/vid_gen                 -> 202 {"poll_url": "/sdcpp/v1/jobs/<id>"}   (409 while another job runs)
  GET  /sdcpp/v1/jobs/<id>               -> {"status": "running" | "completed" | "failed", "result": {...} | "error": "..."}
Each job runs generate.py in a fresh process, so this server itself never touches the GPU and idles at no VRAM. No imports beyond the
standard library on purpose: torch and diffusers are only loaded by the job.
"""
import base64
import json
import os
import random
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
JOBS: dict[str, dict] = {}                     # id -> {"status", "result" | "error"}; only the newest few are kept
LOCK = threading.Lock()


def run_job(jid: str, body: dict) -> None:
    out = Path(tempfile.gettempdir()) / f"{jid}.webm"
    job_file = out.with_suffix(".json")
    job_file.write_text(json.dumps(body | {"out": str(out)}), encoding="utf-8")
    try:
        rc = subprocess.run([sys.executable, str(HERE / "generate.py"), str(job_file)]).returncode
        err = Path(str(out) + ".error")
        if rc == 0 and out.exists():
            result = {"output_format": "webm", "frame_count": FRAMES, "fps": FPS, "b64_json": base64.b64encode(out.read_bytes()).decode()}
            JOBS[jid] = {"status": "completed", "result": result}
        else:
            msg = err.read_text(encoding="utf-8") if err.exists() else f"the generation process ended with code {rc}" + \
                  (" (killed: probably out of memory, see README)" if rc in (-9, 137) else "")
            JOBS[jid] = {"status": "failed", "error": msg}
    finally:
        for f in (out, job_file, Path(str(out) + ".error")):
            f.unlink(missing_ok=True)


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
    steps = (req.get("sample_params") or {}).get("sample_steps", DEFAULT_STEPS)
    if not isinstance(steps, int) or not MIN_STEPS <= steps <= MAX_STEPS:
        return None, f"sample_steps must be an integer from {MIN_STEPS} to {MAX_STEPS}"
    seed = req.get("seed", -1)
    if not isinstance(seed, int):
        return None, "seed must be an integer"
    return {"prompt": prompt.strip(), "steps": steps, "seed": seed if seed >= 0 else random.randrange(2**32)}, ""


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
        if self.path.startswith("/sdcpp/v1/jobs/") and (job := JOBS.get(self.path.rsplit("/", 1)[-1])):
            return self.reply(200, job)
        self.reply(404, {"error": "not found"})

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
            for old in list(JOBS)[:-3]:  # keep the newest few results, they are megabytes of base64
                del JOBS[old]
            JOBS[jid] = {"status": "running"}
        threading.Thread(target=run_job, args=(jid, job), daemon=True).start()
        self.reply(202, {"id": jid, "poll_url": f"/sdcpp/v1/jobs/{jid}"})

    def log_message(self, fmt, *args):  # polls every 2 s would fill the log
        pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "1234"))
    print(f"cogvideo: listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
