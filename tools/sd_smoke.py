r"""First small video job against sd-server (native sdcpp API). Run it inside the agent container (from the core folder, in PowerShell or a Linux shell):
    docker compose cp ../tools/sd_smoke.py agent:/tmp/sd_smoke.py
    docker compose exec -T agent python /tmp/sd_smoke.py 256 256 9 10      # width height frames steps
Saves /data/buia/generated/<YYYYMMDD-HHMMSS>-sd-smoke-<hex>.<format>  (= generated/ in your buia folder), named like the
assistant's own videos so that the Generated media panel lists it.
Request fields follow examples/server/api.md at master-920-2f88688; omitted fields keep the server defaults."""
import base64, os, secrets, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

SD = os.environ.get("SD_URL", "http://sd:1234")
OUT = os.environ.get("OUT_DIR", "/data/buia/generated")
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
        path = f"{OUT}/{datetime.now(ZoneInfo(os.getenv('TZ', 'UTC'))).strftime('%Y%m%d-%H%M%S')}-sd-smoke-{secrets.token_hex(8)}.{res['output_format']}"
        if "result_url" in res:  # cogvideo serves the video from its own URL on the same service (like the agent)
            url = res["result_url"]
            if not url.startswith("/"):
                sys.exit(f"unusable result address: {url!r}")
            data = httpx.get(SD + url, timeout=120).content
        else:  # sd-server embeds the video as base64
            data = base64.b64decode(res["b64_json"])
        os.makedirs(OUT, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        print(f"done in {time.time() - t0:.1f}s: {res['frame_count']} frames at {res['fps']} fps -> {path} ({os.path.getsize(path)} bytes)")
        sys.exit(0)
    if last in ("failed", "cancelled"):
        print("job ended:", j.get("error"))
        sys.exit(1)
    time.sleep(2)
print("gave up after 30 minutes")
sys.exit(1)
