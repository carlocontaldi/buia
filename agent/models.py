"""Model management for the router-mode llama-server.

- installed models, select / load / delete / download / cancel (thin wrappers over the router API)
- Hugging Face quant listing + GGUF header parsing (Range request, no full download)
- "can this machine run it?" estimate: weights + KV cache + overhead vs. VRAM / RAM / disk
"""
import asyncio
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

LLAMA_URL = os.getenv("LLAMA_URL", "http://llama:8080")
HF = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN", "")
CTX_SIZE = int(os.getenv("CTX_SIZE", "16384"))  # same value llama-server runs with
KV_TYPE = os.getenv("KV_CACHE_TYPE", "q8_0")  # same value llama-server runs with
KV_BPE = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
          "q4_1": 20 / 32, "q4_0": 18 / 32}  # bytes per KV element
VRAM_RESERVE_MB = int(os.getenv("VRAM_RESERVE_MB", "2048"))  # what Windows & co. use while a model is loaded
FIT_TARGET_MB = 1024  # llama-server's default --fit-target margin
SETTINGS = Path(os.getenv("DATA_DIR", "/data")) / "workspace" / ".buia-settings.json"
MB = 1024 * 1024


# ---------- settings (selected model) ----------
def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=SETTINGS.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, SETTINGS)


# ---------- router client ----------
async def call_router(method: str, path: str, **kw):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.request(method, LLAMA_URL + path, **kw)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"llama-server unreachable: {e}")
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:300])
    try:
        return r.json()
    except ValueError:
        return {}


LIVE: dict[str, dict] = {}  # model id -> live state from the router's SSE stream (downloads, load progress)


def apply_event(ev: dict) -> None:
    model, kind = ev.get("model"), ev.get("event")
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    if kind == "download_progress":
        files = [f for f in data.values() if isinstance(f, dict)]
        LIVE[model] = {"state": "downloading", "done": sum(f.get("done", 0) for f in files),
                       "total": sum(f.get("total", 0) for f in files)}
    elif kind == "download_failed":
        LIVE[model] = {"state": "failed", "message": str(data.get("message") or data or "download failed")[:200]}
    elif kind == "model_status" and data.get("status") == "loading":
        LIVE[model] = {"state": "loading", "progress": (data.get("progress") or {}).get("value")}
    elif kind in ("download_finished", "model_status", "model_remove"):
        LIVE.pop(model, None)  # the model list shows the new state


BAD_EVENTS: set = set()  # (event, error type) pairs already reported


def parse_event(line: str):
    line = line.strip()
    if line.startswith("data:"):
        line = line[5:].strip()
    if not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except ValueError:
        return None


async def watch_router() -> None:
    """Best effort: follow /models/sse for download and load progress; reconnect forever."""
    while True:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10)) as c:
                async with c.stream("GET", f"{LLAMA_URL}/models/sse") as r:
                    async for line in r.aiter_lines():
                        ev = parse_event(line)
                        if ev:
                            try:
                                apply_event(ev)
                            except Exception as e:  # an event shape nobody expected must not end progress tracking for good
                                key = (ev.get("event"), type(e).__name__)
                                if key not in BAD_EVENTS:  # once per kind: progress events come many times a second
                                    BAD_EVENTS.add(key)
                                    print(f"watch_router: ignoring {ev.get('event')} events it cannot read ({type(e).__name__}: {e})", flush=True)
        except httpx.HTTPError:
            pass
        await asyncio.sleep(3)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(watch_router())
    yield
    task.cancel()


async def list_models(reload: bool = False) -> list[dict]:
    data = await call_router("GET", "/models" + ("?reload=1" if reload else ""))
    out = []
    for m in data.get("data", []):
        s = m.get("status") or {}
        item = {"id": m["id"], "status": s.get("value", "unloaded"), "failed": bool(s.get("failed"))}
        files = [f for f in (s.get("progress") or {}).values() if isinstance(f, dict)] if item["status"] == "downloading" else []
        if files:
            item["done"], item["total"] = sum(f.get("done", 0) for f in files), sum(f.get("total", 0) for f in files)
        live = LIVE.get(m["id"])
        if live and live["state"] == "loading":
            item["progress"] = live.get("progress")
        out.append(item)
    known = {m["id"] for m in out}
    for mid, live in LIVE.items():  # downloads the router doesn't list (yet)
        if mid not in known and live["state"] in ("downloading", "failed"):
            out.append({"id": mid, "status": live["state"], "failed": live["state"] == "failed", "message": live.get("message"),
                        "done": live.get("done"), "total": live.get("total")})
    return out


def selected_model():
    return load_settings().get("model")


async def resolve_model() -> str:
    """Model name for chat requests (required in router mode)."""
    sel = selected_model()
    if sel:
        return sel
    ready = [m["id"] for m in await list_models() if m["status"] in ("loaded", "loading", "unloaded", "sleeping") and not m["failed"]]
    if len(ready) == 1:  # only one installed: just use it
        save_settings({**load_settings(), "model": ready[0]})
        return ready[0]
    raise RuntimeError("No model selected. Open Models, download one and select it.")


async def unload_all(timeout: float = 60) -> list[str]:
    """Unload every model that holds VRAM (loaded or loading) and wait until the router reports none left.
    Call before another engine (image/video generation) uses the GPU. The router reloads a model on demand."""
    ids = [m["id"] for m in await list_models() if m["status"] in ("loaded", "loading")]
    for mid in ids:
        await call_router("POST", "/models/unload", json={"model": mid})
    deadline = time.time() + timeout
    while any(m["status"] in ("loaded", "loading") for m in await list_models()):
        if time.time() > deadline:
            raise RuntimeError("timed out waiting for the model to unload")
        await asyncio.sleep(0.5)
    return ids


# ---------- GGUF header ----------
_FIXED = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class NeedMore(Exception):
    """The buffer ends before the metadata we need."""


class _Buf:
    def __init__(self, b: bytes):
        self.b, self.i = b, 0

    def skip(self, n: int) -> None:
        if self.i + n > len(self.b):
            raise NeedMore
        self.i += n

    def take(self, n: int) -> bytes:
        j = self.i + n
        if j > len(self.b):
            raise NeedMore
        s, self.i = self.b[self.i:j], j
        return s

    def num(self, code: str):
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def string(self, keep: bool = True):
        n = self.num("Q")
        if not keep or n > 512:
            self.skip(n)
            return None
        return self.take(n).decode("utf-8", "replace")

    def skip_strings(self, n: int) -> None:
        b, i = self.b, self.i
        for _ in range(n):
            if i + 8 > len(b):
                raise NeedMore
            i += 8 + struct.unpack_from("<Q", b, i)[0]
        if i > len(b):
            raise NeedMore
        self.i = i

    def value(self, t: int, keep: bool = True):
        if t in _FIXED:
            v = self.num(_FIXED[t])
            return v if keep else None
        if t == 8:
            return self.string(keep)
        if t == 9:
            et, n = self.num("I"), self.num("Q")
            big = not keep or n > 4096
            if et in _FIXED and big:
                self.skip(n * struct.calcsize("<" + _FIXED[et]))
                return None
            if et == 8 and big:
                self.skip_strings(n)
                return None
            return [self.value(et, keep) for _ in range(n)]
        raise ValueError(f"unsupported GGUF value type {t}")


def parse_gguf(buf: bytes) -> dict:
    """Key/value metadata of a GGUF header. Vocabulary arrays are skipped; stops once the architecture keys are read."""
    b = _Buf(buf)
    if b.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    b.num("I")  # version
    b.num("Q")  # tensor count
    meta, arch = {}, None
    for _ in range(b.num("Q")):
        key, t = b.string(), b.num("I")
        if arch and key.startswith("tokenizer.") and f"{arch}.attention.head_count" in meta:
            break  # all architecture keys come before the tokenizer ones
        meta[key] = b.value(t, keep=not key.startswith("tokenizer."))
        if key == "general.architecture":
            arch = meta[key]
    return meta


def arch_info(meta: dict) -> dict:
    a = meta.get("general.architecture")
    if not a:
        raise ValueError("GGUF has no architecture")
    g = lambda k: meta.get(f"{a}.{k}")  # noqa: E731
    layers, emb, heads, kv = g("block_count"), g("embedding_length"), g("attention.head_count"), g("attention.head_count_kv")
    kv = heads if kv is None else kv
    klen, vlen = g("attention.key_length"), g("attention.value_length")
    if isinstance(heads, int) and heads and emb:
        klen, vlen = klen or emb // heads, vlen or emb // heads
    elems = None
    if layers and klen and vlen and kv is not None:
        per_layer = kv if isinstance(kv, list) else [kv] * layers
        elems = sum(per_layer) * (klen + vlen)
    return {"arch": a, "name": meta.get("general.name"), "layers": layers, "ctx_train": g("context_length"),
            "kv_elems": elems, "swa": g("attention.sliding_window") is not None, "experts": g("expert_count")}


# ---------- Hugging Face ----------
def _hf_headers() -> dict:
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def valid_repo(repo: str) -> str:
    repo = repo.strip().removeprefix(HF + "/").strip("/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise HTTPException(400, "use the form owner/repo")
    return repo


async def hf_files(repo: str) -> list[dict]:
    files, url, params = [], f"{HF}/api/models/{repo}/tree/main", {"recursive": "true"}
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=_hf_headers()) as c:
            while url and len(files) < 5000:
                r = await c.get(url, params=params)
                if r.status_code in (401, 403):
                    raise HTTPException(403, "this repo is gated or private: set HF_TOKEN in .env")
                if r.status_code == 404:
                    raise HTTPException(404, "repo not found on Hugging Face")
                r.raise_for_status()
                for it in r.json():
                    if it.get("type") == "file" and it["path"].lower().endswith(".gguf"):
                        files.append({"path": it["path"], "size": (it.get("lfs") or {}).get("size") or it.get("size") or 0})
                url, params = (r.links.get("next") or {}).get("url"), None
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Hugging Face unreachable: {e}")
    return files


_SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)
_QUANT = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?(?:IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))(?![A-Za-z0-9])", re.I)


def group_quants(files: list[dict]) -> list[dict]:
    """One entry per quantization tag (split files summed); mmproj files are ignored."""
    groups: dict[str, dict] = {}
    for f in files:
        name = f["path"].rsplit("/", 1)[-1]
        m = None if name.lower().startswith("mmproj") else _QUANT.search(_SHARD.sub(".gguf", name))
        if m:
            g = groups.setdefault(m.group(1), {"tag": m.group(1), "files": [], "size": 0})
            g["files"].append(f["path"])
            g["size"] += f["size"]
    for g in groups.values():
        g["files"].sort()
    return sorted(groups.values(), key=lambda g: g["size"])


async def gguf_header(repo: str, path: str) -> dict:
    """Architecture info from the first bytes of a GGUF file (Range request; never downloads the whole file)."""
    url = f"{HF}/{repo}/resolve/main/{quote(path)}"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=_hf_headers()) as c:
        for size in (1 << 20, 4 << 20, 16 << 20, 64 << 20):
            buf = bytearray()
            async with c.stream("GET", url, headers={"Range": f"bytes=0-{size - 1}"}) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) >= size:
                        break  # a server that ignores Range must not make us read the whole file
            try:
                return arch_info(parse_gguf(bytes(buf)))
            except NeedMore:
                if len(buf) < size:
                    raise ValueError("truncated GGUF header")
    raise ValueError("GGUF header larger than 64 MB")


_CACHE: dict = {}


async def cached(key, ttl: float, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = await fn()
    _CACHE[key] = (time.time(), val)
    return val


# ---------- machine + estimate ----------
def free_disk_mb() -> int:
    """Free space a download can use. / is the container's own disk: on Docker Desktop (WSL2) that is a virtual disk whose free space
    says nothing about the Windows drive it grows into. The workspace is a bind mount of a Windows folder and reports that drive,
    so the smaller of the two counts."""
    free = []
    for path in ("/", SETTINGS.parent):
        try:
            free.append(shutil.disk_usage(path).free)
        except OSError:
            pass
    return min(free) // MB if free else 0


def _hardware_sync() -> dict:
    gpu = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        rows = [[c.strip() for c in ln.split(",")] for ln in out.stdout.splitlines() if ln.strip()] if out.returncode == 0 else []
        if rows:
            gpu = {"name": " + ".join(r[0] for r in rows), "total_mb": sum(int(r[1]) for r in rows), "used_mb": sum(int(r[2]) for r in rows)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    manual = os.getenv("GPU_VRAM_MB", "").strip()
    if gpu is None and manual.isdigit():
        gpu = {"name": "manual (GPU_VRAM_MB)", "total_mb": int(manual), "used_mb": VRAM_RESERVE_MB}
    ram = {}
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = ln.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                ram[k] = int(v.split()[0]) // 1024
    except (OSError, ValueError):
        pass
    return {"gpu": gpu, "ram_total_mb": ram.get("MemTotal", 0), "ram_avail_mb": ram.get("MemAvailable", 0),
            "disk_free_mb": free_disk_mb()}


async def machine() -> dict:
    hw = await asyncio.to_thread(_hardware_sync)
    try:
        hw["model_loaded"] = any(m["status"] in ("loaded", "loading") for m in await list_models())
    except HTTPException:
        hw["model_loaded"] = False
    if hw["gpu"]:  # VRAM used by other apps: measured when idle, assumed while a model is loaded
        hw["others_mb"] = VRAM_RESERVE_MB if hw["model_loaded"] else hw["gpu"]["used_mb"]
        hw["usable_mb"] = max(0, hw["gpu"]["total_mb"] - hw["others_mb"] - FIT_TARGET_MB)
    return hw


def kv_mb_per_token(arch):
    if not arch or not arch.get("kv_elems"):
        return None
    return arch["kv_elems"] * KV_BPE.get(KV_TYPE, 2.0) / MB


def assess(size_bytes: int, arch, hw: dict, ctx: int = CTX_SIZE, extra_mb: float = 0.0, cpu_mb: float = 0.0) -> dict:
    # extra_mb: GPU memory beyond weights + KV (e.g. a diffusion run's compute buffer); cpu_mb: RAM held by parts kept on the CPU (e.g. a text encoder)
    w = size_bytes / MB
    per_tok = kv_mb_per_token(arch)
    kv = per_tok * ctx if per_tok else 0.0
    over = 300 + max(256.0, 0.06 * w)  # CUDA context + compute buffers (rough)
    need = w + kv + over + extra_mb
    res = {"weights_mb": round(w), "kv_mb": round(kv), "overhead_mb": round(over), "need_mb": round(need),
           "kv_known": per_tok is not None, "kv_upper_bound": bool(arch and arch.get("swa"))}
    if not hw.get("gpu"):
        return {**res, "verdict": "unknown", "reason": "GPU memory could not be read (nvidia-smi is not available to the agent). Set GPU_VRAM_MB in .env."}
    usable = hw["usable_mb"]
    if per_tok:
        cap = (arch.get("ctx_train") or 10**9)
        res["max_ctx"] = max(0, min(cap, int((usable - w - over) / per_tok)))
    ram_free = max(0, hw["ram_avail_mb"] - 1536 - cpu_mb)  # keep some RAM for the OS
    if cpu_mb > max(0, hw["ram_avail_mb"] - 1536):  # parts kept on the CPU must fit in RAM whatever the GPU verdict
        return {**res, "verdict": "no", "reason": f"Not enough RAM for the parts kept on the CPU (about {round(cpu_mb / 1024, 1)} GB)."}
    if need <= 0.9 * usable:
        v, why = "fits", "Runs fully on the GPU with room to spare."
    elif need <= usable:
        v, why = "tight", "Fits, but with little headroom: other GPU apps or a longer context could break it."
    elif need - usable <= ram_free:
        res["ram_needed_mb"] = round(need - usable)
        v, why = "offload", f"Does not fit in VRAM. llama.cpp will keep part on the CPU (about {round((need - usable) / 1024, 1)} GB of RAM): it runs, but slower."
    else:
        v, why = "no", f"Not enough VRAM + RAM (short by about {round((need - usable - ram_free) / 1024, 1)} GB). Pick a smaller quant."
    return {**res, "verdict": v, "reason": why}


async def check_repo(repo: str) -> dict:
    files = await cached(("files", repo), 300, lambda: hf_files(repo))
    quants = group_quants(files)
    if not quants:
        raise HTTPException(404, "no GGUF files found in this repo")
    try:
        arch = await cached(("arch", repo), 3600, lambda: gguf_header(repo, quants[0]["files"][0]))
    except (httpx.HTTPError, ValueError) as e:
        arch = {"error": str(e)[:120]}
    if "error" in arch:
        arch_err, arch = arch["error"], None
    else:
        arch_err = None
    hw = await machine()
    out = []
    for q in quants:
        a = assess(q["size"], arch, hw)
        a.update(tag=q["tag"], id=f"{repo}:{q['tag']}", size_mb=round(q["size"] / MB), shards=len(q["files"]),
                 disk_ok=hw["disk_free_mb"] >= q["size"] / MB + 1024)
        out.append(a)
    return {"repo": repo, "ctx": CTX_SIZE, "kv_type": KV_TYPE, "arch": arch, "arch_error": arch_err, "hardware": hw, "quants": out}


# ---------- API ----------
router = APIRouter(prefix="/api")


class IdBody(BaseModel):
    id: str


class DownloadBody(BaseModel):
    repo: str
    quant: str
    force: bool = False


@router.get("/hardware")
async def api_hardware():
    return await machine()


@router.get("/models")
async def api_models(reload: int = 0):
    return {"models": await list_models(bool(reload)), "selected": selected_model(), "ctx": CTX_SIZE}


@router.get("/hf/quants")
async def api_quants(repo: str):
    return await check_repo(valid_repo(repo))


@router.post("/models/select")
async def api_select(body: IdBody):
    m = next((x for x in await list_models() if x["id"].lower() == body.id.lower()), None)
    if not m or m["status"] == "downloading":
        raise HTTPException(404, "that model is not installed")
    save_settings({**load_settings(), "model": m["id"]})
    if m["status"] in ("loaded", "loading"):
        return {"selected": m["id"]}
    try:
        await call_router("POST", "/models/load", json={"model": m["id"]})
    except HTTPException as e:  # still selected; the chat request will surface the real error
        return {"selected": m["id"], "load_error": e.detail}
    return {"selected": m["id"]}


@router.post("/models/download")
async def api_download(body: DownloadBody):
    repo = valid_repo(body.repo)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", body.quant):
        raise HTTPException(400, "bad quant")
    q = next((x for x in (await check_repo(repo))["quants"] if x["tag"].lower() == body.quant.lower()), None)
    if not q:
        raise HTTPException(404, "that quant is not in the repo")
    if not q["disk_ok"]:
        raise HTTPException(409, "not enough free disk space for this download")
    if q["verdict"] == "no" and not body.force:
        raise HTTPException(409, "this model does not fit in memory; confirm to download anyway")
    await call_router("POST", "/models", json={"model": q["id"]})
    LIVE[q["id"]] = {"state": "downloading", "done": 0, "total": q["size_mb"] * MB}
    return {"id": q["id"]}


@router.post("/models/cancel")
async def api_cancel(body: IdBody):
    await call_router("POST", "/models/unload", json={"model": body.id})  # also cancels a running download
    LIVE.pop(body.id, None)
    return {"ok": True}


@router.post("/models/dismiss")
async def api_dismiss(body: IdBody):
    LIVE.pop(body.id, None)
    return {"ok": True}


@router.delete("/models")
async def api_delete(id: str):
    await call_router("DELETE", "/models", params={"model": id})
    LIVE.pop(id, None)
    s = load_settings()
    if s.get("model") == id:
        s.pop("model")
        save_settings(s)
    return {"ok": True}
