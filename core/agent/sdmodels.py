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
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
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

# Template-only families (no download of their own): the companion files and flags that every model of the family needs, copied by
# make_custom. Slot 0 of "files" is a placeholder for the model file. Flags follow stable-diffusion.cpp's docs/flux.md and docs/sd.md at the
# pinned tag; they were not measured on the RTX 5050 laptop (the presets above were), and have no twin in .env.example.
BASES = {
    "flux1-schnell": {"id": "flux1-schnell", "kind": "image", "name": "FLUX.1 schnell",
                      "note": "Text-to-image in 4 steps. The text encoders clip_l and T5-XXL (fp16, 9.8 GB) run on the CPU.",
                      "files": [("leejet/FLUX.1-schnell-gguf", "flux1-schnell-q4_0.gguf"),
                                ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
                                ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
                                ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors")],
                      "args": "--diffusion-model /models/flux1-schnell-q4_0.gguf --vae /models/ae.safetensors --clip_l /models/clip_l.safetensors "
                              "--t5xxl /models/t5xxl_fp16.safetensors --cfg-scale 1.0 --steps 4 --diffusion-fa "
                              "--backend clip=cpu,diffusion=CUDA0,vae=CUDA0 --params-backend te=cpu,diffusion=disk,vae=disk --lora-model-dir /models/loras"},
    "sdxl": {"id": "sdxl", "kind": "image", "name": "SDXL checkpoint",
             "note": "Single-file SDXL checkpoint (text encoders and VAE inside). Flags for a standard model: 25 steps, cfg 7.",
             "files": [("owner/repo", "checkpoint.safetensors")],
             "args": "-m /models/checkpoint.safetensors --cfg-scale 7.0 --steps 25 --diffusion-fa --vae-tiling --lora-model-dir /models/loras"},
    "sd15": {"id": "sd15", "kind": "image", "name": "SD 1.x checkpoint",
             "note": "Single-file Stable Diffusion 1.x checkpoint (text encoder and VAE inside). Flags for a standard model: 20 steps, cfg 7.",
             "files": [("owner/repo", "checkpoint.safetensors")],
             "args": "-m /models/checkpoint.safetensors --cfg-scale 7.0 --steps 20 --diffusion-fa --vae-tiling --lora-model-dir /models/loras"},
}

RUN: dict[str, dict] = {}  # preset id -> {"state": "downloading" | "failed", "done", "total", "message", "task"}; absent when idle
RUN_LOCK = threading.Lock()


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


def restart_server(kind: str) -> None:
    """Makes sd-run.sh restart the sd-server of this kind with the flags it has now. The only way to end a job that is generating:
    sd-server cannot interrupt one."""
    f = SD_DIR / f"active-{kind}.args"
    try:
        t = max(time.time(), int(f.stat().st_mtime) + 1)
        os.utime(f, (t, t))
    except FileNotFoundError:
        f.touch()  # no file = the server runs on the flags of .env; an empty file means the same to sd-run.sh, and its creation restarts it


# ---------- custom models from any Hugging Face repo (same families as the presets) ----------
RESERVE_MB = 512
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
# resident_mb: compute-buffer estimate when weights stream from disk; for CogVideoX and the checkpoint families it is added to the weights.
# detect: a single-file checkpoint is recognised from the tensor names in its safetensors header (checkpoint_info), not from its file name.
# The CogVideoX service reads one fixed folder, so another repo cannot sit next to the preset: it has a template (for its fit) but no customs.
TEMPLATES = {
    "z-image-turbo":  {"model": 0, "pattern": r"z[_-]?image.*turbo.*\.(gguf|safetensors)$", "resident_mb": 3072, "basis": "estimate", "ref": "1024x1024"},
    "wan21-t2v-1.3b": {"model": 0, "pattern": r"wan2\.1[_-]?t2v[_-]?1\.3b.*\.(gguf|safetensors)$", "resident_mb": 2560, "basis": "measured", "ref": "512x512, 33 frames"},
    "wan22-ti2v-5b":  {"model": 0, "pattern": r"wan2\.2[_-]?ti2v[_-]?5b.*\.(gguf|safetensors)$", "resident_mb": 4096, "basis": "estimate", "ref": "720p, 24 fps"},
    "cogvideox-2b":   {"model": None, "pattern": r"^model_index\.json$", "resident_mb": 1024, "basis": "estimate", "ref": "720x480, 49 frames"},
    "flux1-schnell":  {"model": 0, "pattern": r"flux[-_.]?1[-_.]?schnell.*\.gguf$", "resident_mb": 4096, "basis": "estimate", "ref": "768x768, 4 steps"},
    "sdxl":           {"model": 0, "detect": True, "resident_mb": 2048, "basis": "estimate", "ref": "768x768"},
    "sd15":           {"model": 0, "detect": True, "resident_mb": 1024, "basis": "estimate", "ref": "512x512"},
}
CUSTOM_FILE = SD_DIR / "custom.json"


def base(fam):
    """What a family's customs copy their companion files and flags from: the preset with that id, else the template-only base."""
    return next((p for p in PRESETS if p["id"] == fam and not p.get("custom")), None) or BASES[fam]

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

RAM_RESERVE_MB = 1536  # kept for the OS and the other containers, as in the chat fit check
VERDICT_WORD = {"fits": "fits", "tight": "tight", "offload": "needs RAM offload", "no": "may not fit"}


def read_ram_mb():
    """MemAvailable from /proc/meminfo in MB; None when it cannot be read."""
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def fit_result(need_mb, vram, basis, ref, ram_mb=None):
    """Same four verdicts as the chat fit check (models.assess). need_mb is what the GPU holds during a job; sd.cpp moves what does not fit
    to RAM or disk, so a verdict past 'tight' means slower, not impossible, until RAM runs out too."""
    if not need_mb or not vram:
        return None
    need, usable = round(need_mb), vram - RESERVE_MB
    over = need - usable
    ram_room = None if ram_mb is None else max(0, ram_mb - RAM_RESERVE_MB)
    if need <= 0.9 * usable:
        v, why = "fits", "Fits in VRAM with room to spare."
    elif need <= usable:
        v, why = "tight", "Fits, but with little headroom: other GPU apps or a bigger image could break it."
    elif ram_room is None:
        v, why = "offload", "Does not fit in VRAM; sd.cpp keeps part of it in RAM or on disk (free RAM could not be read)."
    elif over <= ram_room:
        v, why = "offload", f"Does not fit in VRAM: about {round(over / 1024, 1)} GB would move to RAM. It runs, but slower."
    else:
        v, why = "no", f"Not enough VRAM + RAM (short by about {round((over - ram_room) / 1024, 1)} GB)."
    return {"need_mb": need, "vram_mb": vram, "fits": need <= usable, "verdict": v, "reason": why, "basis": basis, "ref": ref}

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
    ram = await models.cached(("sd-ram",), 10, lambda: asyncio.to_thread(read_ram_mb))
    if p.get("engine") == "cogvideo":
        mb = weights_mb(p, sizes, t)
        return fit_result(mb + t["resident_mb"] if mb else None, vram, t["basis"], t["ref"], ram)
    args = p.get("args", "")
    cap = max_vram_mb(args)
    streamed = backend_map(args, "--params-backend").get("diffusion") == "disk"
    if streamed:
        need = cap if cap is not None else t["resident_mb"]
        basis = "capped" if cap is not None else t["basis"]
    else:
        info = p.get("info")  # a checkpoint's weights, counted tensor by tensor from its safetensors header
        mb = info["mb"] if info else weights_mb(p, sizes, t)
        if mb is None:
            return None
        need = mb + t["resident_mb"]
        if cap is not None:
            need = min(need, cap)
        basis = t["basis"]
    res = fit_result(need, vram, basis, t["ref"], ram)
    if res and not streamed:
        res["source"] = "header" if p.get("info") else "file size"
    return res

def make_custom(repo, path, fam, info=None):
    b, t = base(fam), TEMPLATES[fam]
    pid = re.sub(r"[^a-z0-9]+", "-", (repo + "-" + path).lower()).strip("-")[:60] + "-" + hashlib.sha1((repo + path).encode()).hexdigest()[:6]
    files = list(b["files"])
    old = files[t["model"]][1].rsplit("/", 1)[-1]
    new = path.rsplit("/", 1)[-1]
    rel = f"custom/{pid}/{new}"
    files[t["model"]] = (repo, path, rel)
    args = b["args"].replace(f"/models/{old}", f"/models/{rel}")
    return {"id": pid, "kind": b["kind"], "name": f"{repo.rsplit('/', 1)[-1]} ({new})",
            "note": "Custom model from Hugging Face.", "files": files, "args": args, "family": fam, "custom": True,
            **({"info": info} if info else {})}

def inside_sd(d):
    try:
        return d.resolve().is_relative_to(SD_DIR.resolve())
    except (OSError, RuntimeError):
        return False

def shell_unsafe(path: str) -> bool:
    """sd-run.sh hands the flags to sd-server through unquoted word splitting and globbing: a space would cut such a path in two,
    and * ? [ ] would be expanded."""
    return any(c.isspace() or c in "*?[]" for c in path)

def custom_ok(p):
    if not isinstance(p, dict) or not p.get("custom") or p.get("family") not in TEMPLATES or p.get("engine"):
        return False
    info = p.get("info")
    if info is not None and not (isinstance(info, dict) and isinstance(info.get("mb"), (int, float)) and not isinstance(info["mb"], bool)):
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
    if shell_unsafe(path):
        raise HTTPException(400, "file paths cannot contain whitespace or shell wildcard characters")
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


# ---------- single-file checkpoints (SD 1.x, SDXL): recognised and sized from the safetensors header ----------
# The rules follow stable-diffusion.cpp's own version detection (src/model_loader.cpp) for the A1111 tensor layout.
TOKEN_EMB = ("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight", "cond_stage_model.model.token_embedding.weight",
             "conditioner.embedders.0.model.token_embedding.weight", "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight")
CKPT_SKIP = re.compile(r"(^|/)(vae|text_encoder\d*|unet|transformer|safety_checker|controlnet|loras?|embeddings)/", re.I)
CKPT_MIN_MB, CKPT_MAX_FILES, ST_HEADER_MAX = 1024, 12, 16 << 20
DISTILLED = re.compile(r"turbo|lightning|lcm|hyper|dmd|distill", re.I)


def checkpoint_info(header):
    """Family and exact weight sizes of a text-to-image checkpoint, from the JSON header of a .safetensors file; None for anything else
    (Diffusers folders, inpaint/pix2pix models, diffusion-only files, tiny or SD 2.x models, other architectures)."""
    t = {k: v for k, v in header.items() if k != "__metadata__" and isinstance(v, dict) and isinstance(v.get("shape"), list)}
    conv, tok = t.get("model.diffusion_model.input_blocks.0.0.weight"), next((t[n] for n in TOKEN_EMB if n in t), None)
    if not conv or not tok or len(conv["shape"]) != 4 or conv["shape"][1] != 4 or len(tok["shape"]) != 2:
        return None
    if not any(n.startswith("model.diffusion_model.middle_block.1.") for n in t):
        return None
    if any(n.startswith("conditioner.embedders.1.") for n in t):
        fam = "sdxl"
    elif tok["shape"][1] == 768:
        fam = "sd15"
    else:
        return None
    parts, params, dtypes = {"diffusion": 0, "te": 0, "vae": 0, "other": 0}, 0, {}
    for n, v in t.items():
        try:
            elems = math.prod(int(x) for x in v["shape"])
            lo, hi = v["data_offsets"]
            size = int(hi) - int(lo)
        except (KeyError, TypeError, ValueError):
            return None
        part = ("diffusion" if n.startswith("model.diffusion_model.") else "te" if n.startswith(("cond_stage_model.", "conditioner."))
                else "vae" if n.startswith("first_stage_model.") else "other")
        parts[part] += size
        params += elems
        dtypes[v.get("dtype", "?")] = dtypes.get(v.get("dtype", "?"), 0) + elems
    return {"family": fam, "params": params, "dtype": max(dtypes, key=dtypes.get), "mb": round(sum(parts.values()) / MB),
            "parts_mb": {k: round(b / MB) for k, b in parts.items()}}


async def fetch_head(c, url, n):
    buf = bytearray()
    async with c.stream("GET", url, headers={"Range": f"bytes=0-{n - 1}"}) as r:
        r.raise_for_status()
        async for chunk in r.aiter_bytes():
            buf += chunk
            if len(buf) >= n:
                break  # a server that ignores Range must not make us read the whole file
    return bytes(buf[:n])


async def read_checkpoint(repo, path):
    url = f"{models.HF}/{repo}/resolve/main/{quote(path)}"
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=models._hf_headers()) as c:
        buf = await fetch_head(c, url, 1 << 20)
        if len(buf) < 8:
            raise ValueError("not a safetensors file")
        n = struct.unpack("<Q", buf[:8])[0]
        if not 2 <= n <= ST_HEADER_MAX:
            raise ValueError("unreasonable safetensors header")
        if len(buf) < 8 + n:
            buf = await fetch_head(c, url, 8 + n)
        if len(buf) < 8 + n:
            raise ValueError("truncated safetensors header")
    header = json.loads(buf[8:8 + n])
    return checkpoint_info(header) if isinstance(header, dict) else None


async def checkpoint_of(repo, path):
    """checkpoint_info of one file in a repo (a Range request for its header, cached); None when it is not a supported checkpoint or unreadable."""
    try:
        return await models.cached(("sd-st", repo, path), 3600, lambda: read_checkpoint(repo, path))
    except (httpx.HTTPError, ValueError):
        return None


def checkpoint_candidate(path, size):
    return path.lower().endswith(".safetensors") and size >= CKPT_MIN_MB * MB and not CKPT_SKIP.search(path)


async def find_checkpoints(repo, size_by_path):
    """{path: checkpoint_info} for the checkpoints in a repo, and how many candidate files were left unchecked (the largest 12 are read)."""
    cand = sorted((x for x, size in size_by_path.items() if checkpoint_candidate(x, size)), key=lambda x: -size_by_path[x])
    todo = cand[:CKPT_MAX_FILES]
    heads = await asyncio.gather(*(checkpoint_of(repo, x) for x in todo))
    return {x: h for x, h in zip(todo, heads) if h}, len(cand) - len(todo)

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
        with RUN_LOCK:
            if RUN.get(p["id"]) is run:
                RUN.pop(p["id"], None)  # finished: the files on disk are the state
    except asyncio.CancelledError:
        with RUN_LOCK:
            if RUN.get(p["id"]) is run:
                RUN.pop(p["id"], None)
        raise
    except Exception as e:  # shown in the panel; Download again continues from the .part files
        with RUN_LOCK:
            if RUN.get(p["id"]) is run:
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
        badge = f" [{VERDICT_WORD[fit['verdict']]}, {fit['vram_mb'] // 1024} GB VRAM, {fit['basis']}]"
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
    with RUN_LOCK:
        if any(r["state"] == "downloading" for r in RUN.values()):
            raise HTTPException(409, "another download is running; wait for it or cancel it")
        run = {"state": "downloading", "done": 0, "total": 0, "message": "checking model files"}
        RUN[p["id"]] = run
    try:
        infs = await infos(p)
        SD_DIR.mkdir(parents=True, exist_ok=True)
        missing = [(f, i) for f, i in zip(p["files"], infs) if not dest(f).exists()]
        if not missing:
            raise HTTPException(409, "already downloaded")
        parts = sum(dest(f).with_name(dest(f).name + ".part").stat().st_size for f, _ in missing if dest(f).with_name(dest(f).name + ".part").exists())
        if shutil.disk_usage(SD_DIR).free < sum(i["size"] for _, i in missing) - parts + 1024 * MB:
            raise HTTPException(409, "not enough free disk space for this download")
        with RUN_LOCK:
            if RUN.get(p["id"]) is not run:
                raise HTTPException(409, "download was cancelled")
            run.update(total=sum(i["size"] for i in infs), message="")
            run["task"] = asyncio.create_task(fetch_all(p, infs, run))
        return {"id": p["id"]}
    except BaseException:
        with RUN_LOCK:
            if RUN.get(p["id"]) is run:
                RUN.pop(p["id"], None)
        raise


@router.post("/cancel")
async def api_cancel(body: IdBody):
    """Stops a running download (the .part files stay, so Download continues) or clears a failed one."""
    pid = preset(body.id)["id"]
    with RUN_LOCK:
        run = RUN.pop(pid, None)
    if run and run.get("task"):
        run["task"].cancel()
    return {"ok": True}


@router.post("/use")
async def api_use(body: IdBody):
    p = preset(body.id)
    if any(shell_unsafe(part) for f in p["files"] for part in f[1:]):  # such a path (say, in a hand-edited custom.json) is kept, but never started
        raise HTTPException(409, "this model's file path contains a space or a wildcard character, which the server start script cannot pass on; remove it from the panel")
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

def size_map(tree):
    return {it["path"]: (it.get("lfs") or {}).get("size") or it.get("size") or 0 for it in tree}


@router.post("/check")
async def api_check(body: RepoBody):
    repo = clean_repo(body.repo)
    size_by_path = size_map(await repo_tree(repo))
    found, skipped = await find_checkpoints(repo, size_by_path)
    out = []
    for fam, t in TEMPLATES.items():
        b = base(fam)
        if b.get("engine"):
            continue
        if t.get("detect"):
            hits = [(path, info) for path, info in found.items() if info["family"] == fam]
        else:
            rx = re.compile(t["pattern"], re.I)
            hits = [(path, None) for path in size_by_path if rx.search(path)]
        variants = []
        for path, info in hits:
            size = size_by_path[path]
            sizes = [None] * len(b["files"])
            sizes[t["model"]] = size
            v = {"path": path, "size_mb": round(size / MB), "fit": await fit_for(make_custom(repo, path, fam, info), sizes)}
            if info:
                v["detail"] = f"{info['dtype']}, {info['params'] / 1e9:.1f}B parameters"
                if DISTILLED.search(path):
                    v["warning"] = "Looks like a distilled model: it needs fewer steps and a lower cfg than these flags use (25 steps, cfg 7)."
            variants.append(v)
        if variants:
            out.append({"family": fam, "kind": b["kind"], "name": b["name"], "variants": variants})
    if not out:
        names = ", ".join(re.sub(r" \(.*\)$", "", base(f)["name"]) for f, t in TEMPLATES.items() if not base(f).get("engine"))
        raise HTTPException(400, f"unsupported model family; supported: {names}")
    return {"repo": repo, "families": out, "skipped": skipped}

@router.post("/add")
async def api_add(body: AddBody):
    repo = clean_repo(body.repo)
    path = clean_path(body.path)
    size_by_path = size_map(await repo_tree(repo))
    if path not in size_by_path:
        raise HTTPException(404, f"{path} not found in {repo}")
    fam, info = next(((f, None) for f, t in TEMPLATES.items()
                      if not t.get("detect") and not base(f).get("engine") and re.search(t["pattern"], path, re.I)), (None, None))
    if not fam and checkpoint_candidate(path, size_by_path[path]):
        info = await checkpoint_of(repo, path)
        fam = info["family"] if info else None
    if not fam:
        raise HTTPException(400, "unsupported model family for that file")
    p = make_custom(repo, path, fam, info)
    if any(x["id"] == p["id"] for x in PRESETS):
        raise HTTPException(409, "already added")
    PRESETS.append(p)
    save_custom()
    return await view(p)

@router.delete("/models")
async def api_delete(id: str):
    p = preset(id)
    with RUN_LOCK:
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
