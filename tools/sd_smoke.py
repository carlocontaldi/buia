r"""First small video job against sd-server (native sdcpp API). Run it inside the agent container (PowerShell):
    docker compose cp .\tools\sd_smoke.py agent:/tmp/sd_smoke.py
    docker compose exec -T agent python /tmp/sd_smoke.py 256 256 9 10      # width height frames steps
Saves /data/workspace/sd_smoke.<format>  (= sd_smoke.<format> in your buia folder on Windows).
Request fields follow examples/server/api.md at master-920-2f88688; omitted fields keep the server defaults."""
import base64, os, sys, time

import httpx

SD = os.environ.get("SD_URL", "http://sd:1234")
OUT = os.environ.get("OUT_DIR", "/data/workspace")
args = sys.argv[1:]
w, h, frames, steps = (int(a) for a in args) if len(args) == 4 else (256, 256, 9, 10)
body = {"prompt": "a red ball rolling across a wooden floor, soft light", "width": w, "height": h,
        "video_frames": frames, "fps": 16, "seed": 42, "sample_params": {"sample_steps": steps},
        "output_format": "webm"}

t0 = time.time()
r = httpx.post(f"{SD}/sdcpp/v1/vid_gen", json=body, timeout=30)
print(f"submit: HTTP {r.status_code} {r.text[:200]}")
if r.status_code != 202:
    sys.exit(1)
poll_url, last = r.json()["poll_url"], None
while time.time() - t0 < 1800:
    j = httpx.get(SD + poll_url, timeout=30).json()
    if j["status"] != last:
        last = j["status"]
        print(f"[{time.time() - t0:6.1f}s] {last}")
    if last == "completed":
        res = j["result"]
        path = f"{OUT}/sd_smoke.{res['output_format']}"
        with open(path, "wb") as f:
            f.write(base64.b64decode(res["b64_json"]))
        print(f"done in {time.time() - t0:.1f}s: {res['frame_count']} frames at {res['fps']} fps -> {path} ({os.path.getsize(path)} bytes)")
        sys.exit(0)
    if last in ("failed", "cancelled"):
        print("job ended:", j.get("error"))
        sys.exit(1)
    time.sleep(2)
print("gave up after 30 minutes")
sys.exit(1)