"""Image and video models for the stable-diffusion.cpp servers: download a model's files from Hugging Face into sdmodels/, then switch the server to it.

sd-server loads one model when it starts, and every model family needs its own set of files (diffusion model, VAE, text encoder) and
flags. So this is a short list of presets, not "any repo": each one is a set of files plus the flags that go with them (the ones
.env.example documents, measured on the RTX 5050 laptop). "Use" writes the flags to sdmodels/active-<kind>.args; sd-run.sh inside the
sd containers sees the file change and restarts sd-server with them. Nothing here touches Docker or .env.

One preset is not an sd.cpp model: CogVideoX-2b (stable-diffusion.cpp does not support it) runs in the cogvideo service (cogvideo/), which
loads a Diffusers model folder per job. Its preset has "engine": "cogvideo" and no flags. "Use" writes sdmodels/active-video.engine instead,
which makes the agent send video jobs to the cogvideo service; "Use" on any other video preset removes that file again.
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

import httpx
import models
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

SD_DIR = Path(os.getenv("SD_MODELS_DIR", "/data/sdmodels"))  # ./sdmodels on the host; the sd containers see it as /models
SD_URLS = {"image": os.getenv("SD_URL", "http://sd:1234"), "video": os.getenv("SD_VIDEO_URL", "http://sd-video:1234")}
COG_URL = os.getenv("COGVIDEO_URL", "http://cogvideo:1234")  # compose service `cogvideo` (profile cogvideo)
MB = models.MB

# files: (Hugging Face repo, path inside it[, path under sdmodels/]); without the third item a file is saved as sdmodels/<last part of the
# path>, which is the name the flags use. A Diffusers model keeps its folder layout (text_encoder/config.json, vae/config.json, ...), so it uses the third item.
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
    {"id": "cogvideox-2b", "kind": "video", "engine": "cogvideo", "name": "CogVideoX 2B (Diffusers, fp16)",
     "note": "Text-to-video, always 720x480 and 6 s (49 frames at 8 fps), English prompts. Runs in the separate cogvideo service "
             "(install with -CogVideo). The 9.5 GB text encoder runs on the CPU, so it needs about 12 GB of free RAM. Not measured on the 8 GB laptop yet.",
     "files": [("zai-org/CogVideoX-2b", path, "CogVideoX-2b/" + path) for path in (
         "model_index.json", "scheduler/scheduler_config.json",
         "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
         "text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors",
         "tokenizer/added_tokens.json", "tokenizer/special_tokens_map.json", "tokenizer/spiece.model", "tokenizer/tokenizer_config.json",
         "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
         "vae/config.json", "vae/diffusion_pytorch_model.safetensors")]},
]

RUN: dict[str, dict] = {}  # preset id -> {"state": "downloading" | "failed", "done", "total", "message", "task"}; absent when idle


def preset(pid: str) -> dict:
    p = next((p for p in PRESETS if p["id"] == pid), None)
    if not p:
        raise HTTPException(404, "unknown model")
    return p


def dest(f: tuple) -> Path:
    return SD_DIR / (f[2] if len(f) > 2 else f[1].rsplit("/", 1)[-1])


def active_file(p: dict) -> Path:
    return SD_DIR / f"active-{p['kind']}.args"


ENGINE_FILE = SD_DIR / "active-video.engine"  # id of the engine preset (CogVideoX) when it is the video model; absent = the sd video server


def cog_active() -> bool:
    """True when video jobs go to the cogvideo service: CogVideoX was chosen with Use and its files are still all there."""
    try:
        p = preset(ENGINE_FILE.read_text(encoding="utf-8").strip())
    except (OSError, HTTPException):
        return False
    return p.get("engine") == "cogvideo" and all(dest(f).exists() for f in p["files"])


def put(path: Path, text: str) -> None:
    """Replaces path with text in one step: sd-run.sh restarts the server on any change, so it must never see a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=SD_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    os.replace(tmp, path)


# ---------- custom models from any Hugging Face repo (same families as the presets) ----------
RESERVE_MB = 512
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
# resident_mb: compute-buffer estimate when weights stream from disk; for CogVideoX it is added to transformer+VAE.
TEMPLATES = {
    "z-image-turbo":  {"model": 0, "pattern": r"z[_-]?image.*turbo.*\.(gguf|safetensors)$", "resident_mb": 3072, "basis": "estimate", "ref": "1024x1024"},
    "wan21-t2v-1.3b": {"model": 0, "pattern": r"wan2\.1[_-]?t2v[_-]?1\.3b.*\.(gguf|safetensors)$", "resident_mb": 2560, "basis": "measured", "ref": "512x512, 33 frames"},
    "wan22-ti2v-5b":  {"model": 0, "pattern": r"wan2\.2[_-]?ti2v[_-]?5b.*\.(gguf|safetensors)$", "resident_mb": 4096, "basis": "estimate", "ref": "720p, 24 fps"},
    "cogvideox-2b":   {"model": None, "pattern": r"^model_index\.json$", "resident_mb": 1024, "basis": "estimate", "ref": "720x480, 49 frames"},
}
CUSTOM_FILE = SD_DIR / "custom.json"
_VRAM = None
_VRAM_LOCK = asyncio.Lock()

def arg_value(args, flag):
    m = re.search(rf"(?:^|\s){re.escape(flag)}\s+(\S+)", args or "")
    return m.group(1) if m else None

def backend_map(args, flag):
    out = {}
    for part in (arg_value(args, flag) or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip().lower()
    return out

def max_vram_mb(args):
    m = re.search(r"(?:^|,)cuda0=([0-9.]+)([gGmM]?)", arg_value(args, "--max-vram") or "")
    if not m:
        return None
    n = float(m.group(1))
    return round(n) if m.group(2).lower() == "m" else round(n * 1024)

async def vram_mb():
    """GPU_VRAM_MB, else nvidia-smi; read off the event loop and cached (same sources as the chat fit check)."""
    global _VRAM
    async with _VRAM_LOCK:
        if _VRAM is None:
            try:
                _VRAM = int(os.getenv("GPU_VRAM_MB") or 0)
            except ValueError:
                _VRAM = 0
            if not _VRAM:
                def read():
                    try:
                        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                             capture_output=True, text=True, timeout=5)
                        return int(out.stdout.strip().splitlines()[0])
                    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                        return 0
                _VRAM = await asyncio.to_thread(read)
    return _VRAM

def fit_result(need_mb, vram, basis, ref):
    if not need_mb or not vram:
        return None
    need = round(need_mb)
    return {"need_mb": need, "vram_mb": vram, "fits": need <= vram - RESERVE_MB, "basis": basis, "ref": ref}

def weights_mb(p, sizes, t):
    if p.get("engine") == "cogvideo":
        parts = [s for f, s in zip(p["files"], sizes)
                 if s and dest(f).suffix == ".safetensors" and ("/transformer/" in dest(f).as_posix() or "/vae/" in dest(f).as_posix())]
        return sum(parts) / MB if parts else None
    s = sizes[t["model"]] if t["model"] is not None else None
    return s / MB if s else None

async def fit_for(p, sizes):
    t = TEMPLATES.get(p.get("family") or p["id"])
    if not t:
        return None
    vram = await vram_mb()
    if p.get("engine") == "cogvideo":
        mb = weights_mb(p, sizes, t)
        return fit_result(mb + t["resident_mb"] if mb else None, vram, t["basis"], t["ref"])
    args = p.get("args", "")
    cap = max_vram_mb(args)
    streamed = backend_map(args, "--params-backend").get("diffusion") == "disk"
    if streamed:
        need = cap if cap is not None else t["resident_mb"]
        basis = "capped" if cap is not None else t["basis"]
    else:
        mb = weights_mb(p, sizes, t)
        if mb is None:
            return None
        need = mb + t["resident_mb"]
        if cap is not None:
            need = min(need, cap)
        basis = t["basis"]
    return fit_result(need, vram, basis, t["ref"])

def make_custom(repo, path, fam):
    base, t = preset(fam), TEMPLATES[fam]
    pid = re.sub(r"[^a-z0-9]+", "-", (repo + "-" + path).lower()).strip("-")[:60] + "-" + hashlib.sha1((repo + path).encode()).hexdigest()[:6]
    if fam == "cogvideox-2b":
        files = [(repo, f[1], f[2]) for f in base["files"]]  # keep the Diffusers folder layout the cogvideo service expects
        args = ""
    else:
        files = list(base["files"])
        old = files[t["model"]][1].rsplit("/", 1)[-1]
        new = path.rsplit("/", 1)[-1]
        rel = f"custom/{pid}/{new}"
        files[t["model"]] = (repo, path, rel)
        args = base["args"].replace(f"/models/{old}", f"/models/{rel}")
    return {"id": pid, "kind": base["kind"], "name": f"{repo.rsplit('/', 1)[-1]} ({path.rsplit('/', 1)[-1]})",
            "note": "Custom model from Hugging Face.", "files": files, "args": args, "family": fam, "custom": True,
            **({"engine": base["engine"]} if base.get("engine") else {})}

def inside_sd(d):
    try:
        return d.resolve().is_relative_to(SD_DIR.resolve())
    except (OSError, RuntimeError):
        return False

def custom_ok(p):
    if not isinstance(p, dict) or not p.get("custom") or p.get("family") not in TEMPLATES:
        return False
    if p.get("kind") not in ("image", "video") or not isinstance(p.get("args", ""), str):
        return False
    if not isinstance(p.get("id"), str) or not re.fullmatch(r"[a-z0-9-]+", p["id"]):
        return False
    files = p.get("files")
    if not isinstance(files, list) or not files:
        return False
    for f in files:
        if not isinstance(f, (list, tuple)) or len(f) not in (2, 3) or not all(isinstance(x, str) for x in f):
            return False
        if not REPO_RE.fullmatch(f[0]) or Path(f[1]).is_absolute() or ".." in Path(f[1]).parts:
            return False
        if len(f) > 2 and (Path(f[2]).is_absolute() or ".." in Path(f[2]).parts):
            return False
        if not inside_sd(dest(f)):
            return False
    return True

def save_custom():
    SD_DIR.mkdir(parents=True, exist_ok=True)
    put(CUSTOM_FILE, json.dumps([p for p in PRESETS if p.get("custom") and custom_ok(p)], ensure_ascii=False))

def load_custom():
    try:
        items = json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(items, list):
        return
    known = {p["id"] for p in PRESETS}
    for p in items:
        if custom_ok(p) and p["id"] not in known:
            PRESETS.append(p)
            known.add(p["id"])

load_custom()

def clean_repo(repo):
    repo = re.sub(r"^https?://huggingface\.co/", "", repo.strip())
    repo = repo.split("/tree/", 1)[0].split("/resolve/", 1)[0].strip("/")
    if not REPO_RE.fullmatch(repo):
        raise HTTPException(400, "use a Hugging Face repo as owner/repo")
    return repo

def clean_path(path):
    path = path.strip().strip("/")
    if not path or "\\" in path or ".." in Path(path).parts:
        raise HTTPException(400, "invalid file path")
    return path

async def fetch_tree(repo):
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=models._hf_headers()) as c:
            r = await c.get(f"{models.HF}/api/models/{repo}/tree/main", params={"recursive": "true"})
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Hugging Face unreachable: {e}")
    if r.status_code in (401, 403):
        raise HTTPException(403, f"{repo} is gated or private: set HF_TOKEN in .env")
    if r.status_code == 404:
        raise HTTPException(404, f"{repo} not found on Hugging Face")
    r.raise_for_status()
    return [it for it in r.json() if it.get("type") == "file"]

async def repo_tree(repo):
    return await models.cached(("sd-tree", repo), 600, lambda: fetch_tree(repo))

def shared(p: dict, d: Path) -> bool:
    """True when another preset lists file d and has some other file on disk: deleting p must keep d (Wan 2.1 and 2.2 share the text encoder)."""
    return any(dest(g) == d and any(dest(x) != d and dest(x).exists() for x in q["files"])
               for q in PRESETS if q is not p for g in q["files"])


def has_args(p: dict) -> bool:
    """True when the sd server of p's kind is set (active-<kind>.args) to p's flags."""
    try:
        return active_file(p).read_text(encoding="utf-8").strip() == p["args"]
    except OSError:
        return False


def is_active(p: dict) -> bool:
    try:
        if p.get("engine"):
            return ENGINE_FILE.read_text(encoding="utf-8").strip() == p["id"]
    except OSError:
        return False
    return has_args(p) and not (p["kind"] == "video" and ENGINE_FILE.exists())  # an sd video preset is not in use while CogVideoX has the video jobs


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
            return await asyncio.gather(*(file_info(c, repo, path) for repo, path, *_ in p["files"]))
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
    part.parent.mkdir(parents=True, exist_ok=True)  # Diffusers models keep their folders
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
    url = COG_URL if kind == "video" and cog_active() else SD_URLS[kind]
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            j = (await c.get(f"{url}/sdcpp/v1/capabilities")).json()
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
    fit = await fit_for(p, sizes)
    badge = ""
    if fit:
        word = "fits" if fit["fits"] else "may not fit"
        badge = f" [{word} {fit['vram_mb'] // 1024} GB VRAM, {fit['basis']}]"
    return {"id": p["id"], "kind": p["kind"], "name": p["name"] + badge, "note": p["note"], "fit": fit,
            "files": [{"name": dest(f).relative_to(SD_DIR).as_posix(), "have": h} for f, h in zip(p["files"], have)],
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
    if p.get("engine"):  # the sd video server keeps its last model (idle, its weights stay on disk) and gets no jobs
        put(ENGINE_FILE, p["id"])
    else:
        if p["kind"] == "video":
            ENGINE_FILE.unlink(missing_ok=True)
        if not has_args(p):  # same flags again would only restart the server for nothing
            put(active_file(p), p["args"])
    return {"ok": True}


class RepoBody(BaseModel):
    repo: str

class AddBody(BaseModel):
    repo: str
    path: str

@router.post("/check")
async def api_check(body: RepoBody):
    repo = clean_repo(body.repo)
    tree = await repo_tree(repo)
    paths = {it["path"] for it in tree}
    size_by_path = {it["path"]: (it.get("lfs") or {}).get("size") or it.get("size") or 0 for it in tree}
    out = []
    for fam, t in TEMPLATES.items():
        base = preset(fam)
        if fam == "cogvideox-2b":
            if not all(f[1] in paths for f in base["files"]):
                continue
            sizes = [size_by_path.get(f[1]) for f in base["files"]]
            p = make_custom(repo, "model_index.json", fam)
            variants = [{"path": "model_index.json", "size_mb": round(sum(s for s in sizes if s) / MB), "fit": await fit_for(p, sizes)}]
        else:
            rx = re.compile(t["pattern"], re.I)
            variants = []
            for it in tree:
                if not rx.search(it["path"]):
                    continue
                size = size_by_path.get(it["path"]) or 0
                sizes = [None] * len(base["files"])
                sizes[t["model"]] = size
                p = make_custom(repo, it["path"], fam)
                variants.append({"path": it["path"], "size_mb": round(size / MB), "fit": await fit_for(p, sizes)})
            if not variants:
                continue
        out.append({"family": fam, "kind": base["kind"], "name": base["name"], "variants": variants})
    if not out:
        raise HTTPException(400, "unsupported model family; supported: Z-Image Turbo, Wan 2.1 T2V 1.3B, Wan 2.2 TI2V 5B, CogVideoX-2b")
    return {"repo": repo, "families": out}

@router.post("/add")
async def api_add(body: AddBody):
    repo = clean_repo(body.repo)
    path = clean_path(body.path)
    tree = await repo_tree(repo)
    paths = {it["path"] for it in tree}
    if path not in paths:
        raise HTTPException(404, f"{path} not found in {repo}")
    fam = next((f for f, t in TEMPLATES.items()
                if (f == "cogvideox-2b" and all(x[1] in paths for x in preset(f)["files"]))
                or (f != "cogvideox-2b" and re.search(t["pattern"], path, re.I))), None)
    if not fam:
        raise HTTPException(400, "unsupported model family for that file")
    p = make_custom(repo, path, fam)
    if any(x["id"] == p["id"] for x in PRESETS):
        raise HTTPException(409, "already added")
    PRESETS.append(p)
    save_custom()
    return await view(p)

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
        for d in dest(f).parents:  # folders of a Diffusers model: remove the ones that are now empty, never sdmodels/ itself
            if d == SD_DIR:
                break
            try:
                d.rmdir()
            except OSError:
                break
    if p.get("engine"):
        if is_active(p):
            ENGINE_FILE.unlink(missing_ok=True)  # video jobs go back to the sd video server
    elif has_args(p):  # the server then goes back to SD_ARGS / SD_VIDEO_ARGS from .env
        active_file(p).unlink(missing_ok=True)
    if p.get("custom"):
        PRESETS.remove(p)
        save_custom()
    return {"ok": True}
