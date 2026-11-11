"""Image and video models for the stable-diffusion.cpp servers: download a model's files from Hugging Face into sdmodels/, then switch the server to it.

sd-server loads one model when it starts, and every model family needs its own set of files (diffusion model, VAE, text encoder) and
flags. So this is a short list of presets, not "any repo": each one is a set of files plus the flags that go with them (the ones
.env.example documents, measured on the RTX 5050 laptop). "Use" writes the flags to sdmodels/active-<kind>.args; sd-run.sh inside the
sd containers sees the file change and restarts sd-server with them. Nothing here touches Docker or .env.
"""
import asyncio
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import httpx
import models
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

SD_DIR = Path(os.getenv("SD_MODELS_DIR", "/data/sdmodels"))  # ./sdmodels on the host; the sd containers see it as /models
SD_URLS = {"image": os.getenv("SD_URL", "http://sd:1234"), "video": os.getenv("SD_VIDEO_URL", "http://sd-video:1234")}
MB = models.MB

# files: (Hugging Face repo, path inside it); each is saved as sdmodels/<last part of the path>, which is the name the flags use.
# Keep args equal to the matching SD_ARGS / SD_VIDEO_ARGS example in .env.example.
PRESETS = [
    {"id": "z-image-turbo", "kind": "image", "name": "Z-Image Turbo (Q4_0)",
     "note": "Fast text-to-image, 8 steps. Text encoder Qwen3-4B (Q4_K_M) runs on the CPU.",
     "files": [("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_0.gguf"),
               ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
               ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors")],
     "args": "--diffusion-model /models/z_image_turbo-Q4_0.gguf --vae /models/ae.safetensors --llm /models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf "
             "--cfg-scale 1.0 --steps 8 --diffusion-fa --backend clip=cpu,diffusion=CUDA0,vae=CUDA0 "
             "--params-backend te=cpu,diffusion=disk,vae=disk --lora-model-dir /models/loras"},
    {"id": "wan21-t2v-1.3b", "kind": "video", "name": "Wan 2.1 T2V 1.3B (fp16)",
     "note": "Text-to-video up to 512x512 and 33 frames on an 8 GB GPU. Text encoder umt5-xxl (Q4_K_S) runs on the CPU.",
     "files": [("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"),
               ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors"),
               ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_S.gguf")],
     "args": "--diffusion-model /models/wan2.1_t2v_1.3B_fp16.safetensors --vae /models/wan_2.1_vae.safetensors "
             "--t5xxl /models/umt5-xxl-encoder-Q4_K_S.gguf --diffusion-fa --max-vram cuda0=5 --lora-model-dir /models/loras "
             "--backend clip=cpu,diffusion=CUDA0,vae=CUDA0 --params-backend te=cpu,diffusion=disk,vae=disk"},
    {"id": "wan22-ti2v-5b", "kind": "video", "name": "Wan 2.2 TI2V 5B (Q4_K_M)",
     "note": "Text-to-video, 5B model with its own 16x compressing VAE; trained for 720p at 24 fps. Shares the umt5-xxl text encoder (CPU) with Wan 2.1. "
             "Not measured on the 8 GB laptop yet.",
     "files": [("unsloth/Wan2.2-TI2V-5B-GGUF", "Wan2.2-TI2V-5B-Q4_K_M.gguf"),
               ("unsloth/Wan2.2-TI2V-5B-GGUF", "VAE/Wan2.2_VAE.safetensors"),
               ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_S.gguf")],
     "args": "--diffusion-model /models/Wan2.2-TI2V-5B-Q4_K_M.gguf --vae /models/Wan2.2_VAE.safetensors "
             "--t5xxl /models/umt5-xxl-encoder-Q4_K_S.gguf --cfg-scale 6.0 --sampling-method euler --flow-shift 3.0 --diffusion-fa "
             "--max-vram cuda0=5 --lora-model-dir /models/loras "
             "--backend clip=cpu,diffusion=CUDA0,vae=CUDA0 --params-backend te=cpu,diffusion=disk,vae=disk"},
]

RUN: dict[str, dict] = {}  # preset id -> {"state": "downloading" | "failed", "done", "total", "message", "task"}; absent when idle


def preset(pid: str) -> dict:
    p = next((p for p in PRESETS if p["id"] == pid), None)
    if not p:
        raise HTTPException(404, "unknown model")
    return p


def dest(f: tuple) -> Path:
    return SD_DIR / f[1].rsplit("/", 1)[-1]


def active_file(p: dict) -> Path:
    return SD_DIR / f"active-{p['kind']}.args"


def shared(p: dict, d: Path) -> bool:
    """True when another preset lists file d and has some other file on disk: deleting p must keep d (Wan 2.1 and 2.2 share the text encoder)."""
    return any(dest(g) == d and any(dest(x) != d and dest(x).exists() for x in q["files"])
               for q in PRESETS if q is not p for g in q["files"])


def is_active(p: dict) -> bool:
    try:
        return active_file(p).read_text(encoding="utf-8").strip() == p["args"]
    except OSError:
        return False


# ---------- Hugging Face ----------
async def file_info(c: httpx.AsyncClient, repo: str, path: str) -> dict:
    """Size and sha256 of one file as Hugging Face lists them (one request, no download)."""
    r = await c.post(f"{models.HF}/api/models/{repo}/paths-info/main", json={"paths": [path]})
    if r.status_code in (401, 403):
        raise HTTPException(403, f"{repo} is gated or private: set HF_TOKEN in .env")
    if r.status_code == 404 or (r.status_code == 200 and not r.json()):
        raise HTTPException(404, f"{path} not found in {repo} on Hugging Face")
    r.raise_for_status()
    it = r.json()[0]
    lfs = it.get("lfs") or {}
    return {"size": lfs.get("size") or it.get("size") or 0, "sha256": lfs.get("oid")}


async def infos(p: dict) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=models._hf_headers()) as c:
            return await asyncio.gather(*(file_info(c, repo, path) for repo, path in p["files"]))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Hugging Face unreachable: {e}")


def hash_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h


def write_chunk(f, h, chunk: bytes) -> None:
    f.write(chunk)
    h.update(chunk)


async def fetch_file(repo: str, path: str, d: Path, info: dict, progress) -> None:
    """Download repo/path to d. Written to d.part first and renamed only when size and sha256 match, so d exists only when complete.
    A .part left by a cancel or a dropped connection is continued with a Range request."""
    part = d.with_name(d.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > info["size"]:
        part.unlink()
        have = 0
    h = await asyncio.to_thread(hash_file, part) if have else hashlib.sha256()  # the checksum covers what is already on disk too
    progress(have)
    if have < info["size"]:
        headers = {**models._hf_headers(), **({"Range": f"bytes={have}-"} if have else {})}
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=60), follow_redirects=True) as c:
            async with c.stream("GET", f"{models.HF}/{repo}/resolve/main/{quote(path)}", headers=headers) as r:
                if r.status_code in (401, 403):
                    raise RuntimeError(f"HTTP {r.status_code} from Hugging Face: gated repo? set HF_TOKEN in .env")
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} from Hugging Face")
                if have and r.status_code == 200:  # the server ignored Range and sends the whole file
                    have, h = 0, hashlib.sha256()
                with part.open("ab" if have else "wb") as f:
                    async for chunk in r.aiter_bytes(1 << 20):
                        await asyncio.to_thread(write_chunk, f, h, chunk)  # disk writes (a Windows bind mount) must not stall the event loop
                        have += len(chunk)
                        progress(have)
    if have != info["size"]:
        raise RuntimeError(f"{d.name}: got {have} of {info['size']} bytes (the partial file is kept; Download continues it)")
    if info["sha256"] and h.hexdigest() != info["sha256"]:
        part.unlink()
        raise RuntimeError(f"{d.name}: checksum mismatch (partial file removed; try again)")
    os.replace(part, d)


async def fetch_all(p: dict, infs: list[dict], run: dict) -> None:
    try:
        base = 0  # bytes of the files finished before the current one
        for f, info in zip(p["files"], infs):
            d = dest(f)
            if d.exists():
                base += d.stat().st_size
                run.update(done=base)
                continue
            await fetch_file(f[0], f[1], d, info, lambda n, b=base: run.update(done=b + n))
            base += info["size"]
        RUN.pop(p["id"], None)  # finished: the files on disk are the state
    except asyncio.CancelledError:
        RUN.pop(p["id"], None)
        raise
    except Exception as e:  # shown in the panel; Download again continues from the .part files
        run.update(state="failed", message=f"{type(e).__name__}: {e}"[:300])


# ---------- API ----------
router = APIRouter(prefix="/api/sd")


class IdBody(BaseModel):
    id: str


async def server_state(kind: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            j = (await c.get(f"{SD_URLS[kind]}/sdcpp/v1/capabilities")).json()
        return {"up": True, "model": (j.get("model") or {}).get("stem")}
    except (httpx.HTTPError, ValueError):
        return {"up": False}


async def view(p: dict) -> dict:
    have = [dest(f).exists() for f in p["files"]]
    infs = None
    if not all(have):  # sizes of what is still missing come from Hugging Face (cached); a failure only leaves them unknown
        try:
            infs = await models.cached(("sd-info", p["id"]), 600, lambda: infos(p))
        except (HTTPException, httpx.HTTPError):
            pass
    sizes = [dest(f).stat().st_size if h else (infs[i]["size"] if infs else None) for i, (f, h) in enumerate(zip(p["files"], have))]
    total = None if None in sizes else sum(sizes)
    run = RUN.get(p["id"], {})
    left = None if total is None else sum(s for s, h in zip(sizes, have) if not h)
    free = shutil.disk_usage(SD_DIR).free // MB if SD_DIR.is_dir() else 0
    return {"id": p["id"], "kind": p["kind"], "name": p["name"], "note": p["note"],
            "files": [{"name": dest(f).name, "have": h} for f, h in zip(p["files"], have)],
            "size_mb": None if total is None else round(total / MB), "ready": all(have), "active": all(have) and is_active(p),
            "state": run.get("state"), "done": run.get("done", 0), "total": run.get("total"), "message": run.get("message"),
            "disk_ok": left is None or free >= left / MB + 1024}


@router.get("/models")
async def api_list():
    image, video, *views = await asyncio.gather(server_state("image"), server_state("video"), *(view(p) for p in PRESETS))
    return {"servers": {"image": image, "video": video}, "models": views,
            "disk_free_mb": shutil.disk_usage(SD_DIR).free // MB if SD_DIR.is_dir() else 0}


@router.post("/download")
async def api_download(body: IdBody):
    p = preset(body.id)
    if any(r["state"] == "downloading" for r in RUN.values()):
        raise HTTPException(409, "another download is running; wait for it or cancel it")
    infs = await infos(p)
    SD_DIR.mkdir(parents=True, exist_ok=True)
    missing = [(f, i) for f, i in zip(p["files"], infs) if not dest(f).exists()]
    if not missing:
        raise HTTPException(409, "already downloaded")
    parts = sum(dest(f).with_name(dest(f).name + ".part").stat().st_size for f, _ in missing if dest(f).with_name(dest(f).name + ".part").exists())
    if shutil.disk_usage(SD_DIR).free < sum(i["size"] for _, i in missing) - parts + 1024 * MB:
        raise HTTPException(409, "not enough free disk space for this download")
    run = {"state": "downloading", "done": 0, "total": sum(i["size"] for i in infs), "message": ""}
    RUN[p["id"]] = run
    run["task"] = asyncio.create_task(fetch_all(p, infs, run))
    return {"id": p["id"]}


@router.post("/cancel")
async def api_cancel(body: IdBody):
    """Stops a running download (the .part files stay, so Download continues) or clears a failed one."""
    run = RUN.pop(preset(body.id)["id"], None)
    if run and run.get("task"):
        run["task"].cancel()
    return {"ok": True}


@router.post("/use")
async def api_use(body: IdBody):
    p = preset(body.id)
    if not all(dest(f).exists() for f in p["files"]):
        raise HTTPException(409, "download the model first")
    if not is_active(p):  # same flags again would only restart the server for nothing
        fd, tmp = tempfile.mkstemp(dir=SD_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(p["args"] + "\n")
        os.replace(tmp, active_file(p))
    return {"ok": True}


@router.delete("/models")
async def api_delete(id: str):
    p = preset(id)
    if RUN.get(id, {}).get("state") == "downloading":
        raise HTTPException(409, "cancel the download first")
    RUN.pop(id, None)
    for f in p["files"]:
        if shared(p, dest(f)):
            continue
        dest(f).unlink(missing_ok=True)
        dest(f).with_name(dest(f).name + ".part").unlink(missing_ok=True)
    if is_active(p):  # the server then goes back to SD_ARGS / SD_VIDEO_ARGS from .env
        active_file(p).unlink(missing_ok=True)
    return {"ok": True}
