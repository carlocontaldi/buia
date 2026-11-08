"""BUIA agent: a thin tool-calling loop between a web UI and llama-server.

Virtual file system (container paths under DATA_DIR, default /data):
  workspace/  -> your buia folder (read/write)
  vault/      -> your Obsidian vault (mounted read-only by default)
"""
import asyncio
import base64
import json
import os
import re
import secrets
import tempfile
import time
from contextlib import aclosing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import models
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

LLAMA_URL = os.getenv("LLAMA_URL", "http://llama:8080")
DATA = Path(os.getenv("DATA_DIR", "/data")).resolve()
ROOTS = [(DATA / "workspace").resolve(), (DATA / "vault").resolve()]
MEMORY_FILES = [p.strip() for p in os.getenv("MEMORY_FILES", "workspace/BUIA.md").split(",") if p.strip()]
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "12000"))  # ~3k tokens of always-loaded memory
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))  # tool-call rounds per user message
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))  # chat messages kept per request
TOOL_OUT_MAX = 8000  # chars returned to the model per tool call
CTX_SIZE = int(os.getenv("CTX_SIZE", "16384"))  # keep equal to llama-server's -c
PROMPT_BUDGET = int(CTX_SIZE * 3 * 0.7)  # chars: assume ~3 chars/token, keep 30% free for tool schemas + the reply
TZ = ZoneInfo(os.getenv("TZ", "UTC"))
CHATS = Path(os.getenv("CHATS_DIR", str(DATA / "workspace" / "chats")))  # one JSON file per chat

BASE_PROMPT = """You are a personal assistant running locally on the user's laptop. Be concise and direct, and say so when you are unsure.
Today is {today}.

You have tools on a file system with two roots:
- workspace/  the user's buia folder (read/write)
- vault/      the user's Obsidian vault (may be read-only)

Rules:
- Never guess what a note says. Search or read it first.
- Before changing an existing file, read it. Prefer append_file over write_file. Do not overwrite a file unless asked.
- Memory: the files below are loaded on every request. When the user tells you something worth remembering across chats, or asks you to remember something, append one short line to the end of workspace/BUIA.md (its last section is Memory). Only record what the user said, not your own guesses. Never store passwords or secrets.
- After changing a file, say which file you changed.
- Images and video: when the user asks for a picture, image or video, call generate_image or generate_video once with a detailed English prompt. The result is shown to the user automatically: answer in one short sentence and never write image links.
"""


# ---------- file tools ----------
def protected(p: Path) -> bool:
    """True for everything inside a protected top-level folder of either root:
    - .obsidian: plugins/ and community-plugins.json there are code that Obsidian runs on your PC;
    - any clone of this repo, whatever it is called: it holds .env (HF token) and the scripts you run. A clone is recognised by
      its install.ps1 and docker-compose.yml, so a renamed folder is still covered.
    A Windows bind mount treats .OBSIDIAN and .obsidian (or BUIA-REPO and buia-repo) as one folder while resolve() does not,
    hence the case-insensitive name test and the file test, which the mount answers for any spelling."""
    for r in ROOTS:
        if p.is_relative_to(r):
            parts = p.relative_to(r).parts
            if parts:
                top = r / parts[0]
                if parts[0].casefold() == ".obsidian" or ((top / "install.ps1").exists() and (top / "docker-compose.yml").exists()):
                    return True
    return False


def safe(rel: str) -> Path:
    """Resolve a virtual path; refuse anything outside workspace/ or vault/ (incl. .. and symlinks)."""
    p = (DATA / rel.replace("\\", "/").lstrip("/")).resolve()
    if not any(p == r or p.is_relative_to(r) for r in ROOTS):
        raise ValueError("path must be inside workspace/ or vault/")
    if protected(p):
        raise ValueError("that folder is protected (a clone of this repo holds .env and the install scripts, .obsidian/ holds Obsidian plugins)")
    return p


def rel(p: Path) -> str:
    return p.relative_to(DATA).as_posix()


def list_dir(path: str = "") -> str:
    if not path.strip("/ "):
        return "workspace/\nvault/"
    p = safe(path)
    if not p.is_dir():
        return "error: not a directory"
    items = sorted((i for i in p.iterdir() if not i.name.startswith(".")), key=lambda i: (i.is_file(), i.name.lower()))
    lines = [i.name + ("/" if i.is_dir() else "") for i in items[:200]]
    if len(items) > 200:
        lines.append(f"... {len(items) - 200} more")
    return "\n".join(lines) or "(empty)"


def read_file(path: str, offset: int = 0) -> str:
    p = safe(path)
    if not p.is_file():
        return "error: not a file"
    text = p.read_text(encoding="utf-8", errors="replace")
    end = offset + TOOL_OUT_MAX - 200  # leave room for the footer
    chunk = text[offset:end]
    if end < len(text):
        chunk += f"\n[chars {offset}-{end} of {len(text)}; call read_file again with offset={end} for more]"
    return chunk


def search_files(query: str, path: str = "") -> str:
    roots = [safe(path)] if path.strip("/ ") else ROOTS
    terms = query.lower().split()
    if not terms:
        return "error: empty query"
    hits = []
    for root in roots:
        if not root.is_dir():
            return "error: path is not a directory"
        for f in sorted(root.rglob("*.md")):
            parts = f.relative_to(root).parts
            if any(x.startswith(".") for x in parts) or not f.is_file():
                continue
            if protected(f):
                continue
            if all(t in f.stem.lower() for t in terms):
                hits.append(f"{rel(f)}  (file name match)")
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for n, line in enumerate(lines, 1):
                if all(t in line.lower() for t in terms):
                    hits.append(f"{rel(f)}:{n}: {line.strip()[:200]}")
            if len(hits) >= 40:
                return "\n".join(hits[:40]) + "\n[first 40 hits shown; narrow the query or path]"
    return "\n".join(hits) or "no matches"


def write_file(path: str, content: str, overwrite: bool = False) -> str:
    p = safe(path)
    if p.exists() and not overwrite:
        return "error: file exists. Use append_file, or call again with overwrite=true if the user asked for it."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {rel(p)} ({len(content)} chars)"


def append_file(path: str, content: str) -> str:
    p = safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write("\n" + content.rstrip() + "\n")
    return f"appended to {rel(p)}"


SD_URL = os.getenv("SD_URL", "http://sd:1234")  # stable-diffusion.cpp server (compose service `sd`, profile sd)
SD_VIDEO_URL = os.getenv("SD_VIDEO_URL", "http://sd-video:1234")  # sd-server loaded with a video model (compose service sd-video)
# Measured on the RTX 5050 laptop (8 GB): 512x512, 33 frames, 10 steps took 153 s and peaked at 6.1 GB of VRAM. Raise only after measuring.
VIDEO_MAX_SIDE, VIDEO_MAX_FRAMES, VIDEO_MAX_STEPS = 512, 33, 20
IMAGE_MAX_SIDE = 768  # not measured yet: raise only after measuring
VIDEO_TIMEOUT = 900  # seconds to wait for one job
GENERATED = DATA / "workspace" / "generated"  # images and videos made by the tools
MEDIA_MIME = {"png": "image/png", "webp": "image/webp", "webm": "video/webm"}  # what the chat UI can show
MEDIA_RE = re.compile(r"\d{8}-\d{6}-[a-z0-9-]+\.(?:png|webp|webm)")


def _sd_job(kind: str, body: dict) -> tuple[str, list]:
    """Run one job on sd-server (kind: img_gen or vid_gen). Returns (text for the model, media entries for the chat UI)."""
    label = "video" if kind == "vid_gen" else "image"
    base = SD_VIDEO_URL if kind == "vid_gen" else SD_URL
    try:  # check the server before unloading the chat model
        caps = httpx.get(f"{base}/sdcpp/v1/capabilities", timeout=5)
        caps.raise_for_status()
    except httpx.HTTPError as e:
        return f"error: {label} server not reachable at {base} ({type(e).__name__}). Start it with: docker compose --profile sd up -d", []
    modes = caps.json().get("supported_modes")
    if modes and kind not in modes:  # one model at a time: an image request on a video model (or the reverse) cannot work
        return f"error: the sd server runs a model that does not support {kind} (it supports: {', '.join(modes)}). Change the model in SD_ARGS (.env) and restart the sd service.", []
    asyncio.run(models.unload_all())  # the diffusion model needs the GPU; the router reloads the chat model on the next message
    t0 = time.time()
    r = httpx.post(f"{base}/sdcpp/v1/{kind}", json=body, timeout=30)
    if r.status_code != 202:
        return f"error: {label} server refused the job: HTTP {r.status_code} {r.text[:200]}", []
    poll_url = r.json()["poll_url"]
    while time.time() - t0 < VIDEO_TIMEOUT:
        j = httpx.get(base + poll_url, timeout=30).json()
        if j["status"] == "completed":
            res = j["result"]
            if kind == "vid_gen":
                data, detail = res["b64_json"], f"{res['frame_count']} frames at {res['fps']} fps, "
            else:
                data, detail = res["images"][0]["b64_json"], ""
            GENERATED.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^a-z0-9]+", "-", body["prompt"].lower())[:40].strip("-") or label
            path = GENERATED / f"{datetime.now(TZ).strftime('%Y%m%d-%H%M%S')}-{slug}.{res['output_format']}"
            path.write_bytes(base64.b64decode(data))
            out = f"saved {rel(path)} ({detail}{body['width']}x{body['height']}, {time.time() - t0:.0f} s)"
            mime = MEDIA_MIME.get(res["output_format"])
            if not mime:  # e.g. avi: saved, but the browser cannot show it
                return out, []
            return out + ". It is already shown to the user in the chat: reply with one short sentence and no links.", \
                [{"kind": label, "url": f"/api/media/{path.name}", "mime": mime, "prompt": body["prompt"]}]
        if j["status"] in ("failed", "cancelled"):
            return f"error: {label} job {j['status']}: {j.get('error')}", []
        time.sleep(2)
    return f"error: {label} job not finished after {VIDEO_TIMEOUT} s (the server may still be working on it)", []


def generate_video(prompt: str, width: int = 256, height: int = 256, frames: int = 17, steps: int = 10, seed: int = 42) -> str | tuple[str, list]:
    prompt, width, height, frames, steps, seed = prompt.strip(), int(width), int(height), int(frames), int(steps), int(seed)
    if not prompt:
        return "error: empty prompt"
    if not (128 <= width <= VIDEO_MAX_SIDE and 128 <= height <= VIDEO_MAX_SIDE and width % 16 == 0 and height % 16 == 0):
        return f"error: width and height must be multiples of 16 between 128 and {VIDEO_MAX_SIDE}"
    if not (5 <= frames <= VIDEO_MAX_FRAMES and (frames - 1) % 4 == 0):
        return f"error: frames must be 5, 9, 13, ... up to {VIDEO_MAX_FRAMES}"
    if not 1 <= steps <= VIDEO_MAX_STEPS:
        return f"error: steps must be between 1 and {VIDEO_MAX_STEPS}"
    body = {"prompt": prompt, "width": width, "height": height, "video_frames": frames, "fps": 16, "seed": seed,
            "sample_params": {"sample_steps": steps}, "output_format": "webm"}
    return _sd_job("vid_gen", body)


def generate_image(prompt: str, width: int = 512, height: int = 512, seed: int = -1) -> str | tuple[str, list]:
    prompt, width, height, seed = prompt.strip(), int(width), int(height), int(seed)
    if not prompt:
        return "error: empty prompt"
    if not (256 <= width <= IMAGE_MAX_SIDE and 256 <= height <= IMAGE_MAX_SIDE and width % 16 == 0 and height % 16 == 0):
        return f"error: width and height must be multiples of 16 between 256 and {IMAGE_MAX_SIDE}"
    # no steps / cfg here: distilled models need specific values, which belong in the sd server's own arguments (SD_ARGS)
    body = {"prompt": prompt, "width": width, "height": height, "seed": seed, "output_format": "png"}
    return _sd_job("img_gen", body)


IMPL = {f.__name__: f for f in (list_dir, read_file, search_files, write_file, append_file, generate_video, generate_image)}
S = {"type": "string"}
I = {"type": "integer"}


def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    fn("list_dir", "List a directory. Empty path lists the roots (workspace/, vault/).", {"path": S}, []),
    fn("read_file", "Read a text file. Long files are paged with offset.", {"path": S, "offset": {"type": "integer"}}, ["path"]),
    fn("search_files", "Search .md files. Every word must appear in a line or in the file name. Empty path searches workspace and vault.",
       {"query": S, "path": S}, ["query"]),
    fn("write_file", "Create a new file. Fails if it exists unless overwrite is true.",
       {"path": S, "content": S, "overwrite": {"type": "boolean"}}, ["path", "content"]),
    fn("append_file", "Append text to the end of a file (created if missing). Use this for memory notes.",
       {"path": S, "content": S}, ["path", "content"]),
    fn("generate_video", "Make a short video from a text prompt with the local video model. Slow: about 20 s to 3 min. "
       "Saves a .webm in workspace/generated/ and returns its path. It unloads the chat model while it runs, so the next reply may start slowly. "
       "Defaults 256x256, 17 frames, 10 steps; width/height are multiples of 16 up to 512, frames are 5, 9, 13, ... up to 33, steps up to 20.",
       {"prompt": S, "width": I, "height": I, "frames": I, "steps": I, "seed": I}, ["prompt"]),
    fn("generate_image", "Make an image from a text prompt with the local image model. Slow: seconds to minutes. "
       "Saves a .png in workspace/generated/. It unloads the chat model while it runs, so the next reply may start slowly. "
       "Defaults 512x512; width/height are multiples of 16 up to 768. Write the prompt in English: subject, setting, style, light.",
       {"prompt": S, "width": I, "height": I, "seed": I}, ["prompt"]),
]


def call_tool(name: str, raw_args) -> tuple[str, list]:
    """Returns (text for the model, media entries for the chat UI). Only the generate_* tools return media."""
    if name not in IMPL:
        return f"error: unknown tool {name}", []
    try:
        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        out = IMPL[name](**args)
    except Exception as e:  # report to the model so it can correct itself
        out = f"error: {type(e).__name__}: {e}"
    out, media = out if isinstance(out, tuple) else (out, [])
    return out[:TOOL_OUT_MAX], media


# ---------- context budget ----------
TOOLS_CHARS = len(json.dumps(TOOLS))
STUB = "[old tool output removed to save context; call the tool again if needed]"


def prompt_chars(msgs: list) -> int:
    return TOOLS_CHARS + sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "")
                             + (len(json.dumps(m["tool_calls"])) if "tool_calls" in m else 0) for m in msgs)


def fit_context(msgs: list) -> None:
    """Shrink msgs in place until the estimated prompt fits PROMPT_BUDGET: oldest chat history first
    (whole turns), then this turn's oldest tool outputs. Never touches the system prompt, the current
    user message or the newest message."""
    cur = max(i for i, m in enumerate(msgs) if m["role"] == "user")  # current user message
    while prompt_chars(msgs) > PROMPT_BUDGET and cur > 1:
        del msgs[1]
        cur -= 1
        while cur > 1 and msgs[1]["role"] != "user":  # keep a user message first
            del msgs[1]
            cur -= 1
    for m in msgs[cur + 1:-1]:
        if prompt_chars(msgs) <= PROMPT_BUDGET:
            break
        if m["role"] == "tool":
            m["content"] = STUB


# ---------- memory ----------
def system_prompt() -> str:
    # date only: a changing clock in the prompt prefix would defeat llama-server's prompt cache
    parts = [BASE_PROMPT.format(today=datetime.now(TZ).strftime("%A %Y-%m-%d"))]
    budget = MEMORY_MAX_CHARS
    files = [f for pattern in MEMORY_FILES for f in sorted(DATA.glob(pattern)) if f.is_file()]
    for f in files:
        if budget <= 0:
            break
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        cut = text[:budget]
        budget -= len(cut)
        parts.append(f"# Memory file: {rel(f)}\n{cut}" + ("\n[truncated]" if len(cut) < len(text) else ""))
    return "\n\n".join(parts)


# ---------- chat storage (JSON files under CHATS_DIR) ----------
ID_RE = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def chat_path(cid: str) -> Path:
    if not ID_RE.fullmatch(cid):  # also blocks path traversal
        raise HTTPException(404, "no such chat")
    return CHATS / f"{cid}.json"


def new_chat() -> dict:
    cid = datetime.now(TZ).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    return {"id": cid, "title": "", "created": now_iso(), "updated": now_iso(), "messages": []}


def load_chat(cid: str) -> dict:
    p = chat_path(cid)
    if not p.is_file():
        raise HTTPException(404, "no such chat")
    return json.loads(p.read_text(encoding="utf-8"))


def save_chat(chat: dict) -> None:
    p = chat_path(chat["id"])
    if not chat["messages"]:  # never keep empty chats
        p.unlink(missing_ok=True)
        return
    chat["updated"] = now_iso()
    CHATS.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CHATS, suffix=".tmp")  # write + rename: no half-written files
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


# ---------- streaming from llama-server ----------
async def llm_stream(msgs, think: bool | None = None):
    """Yield ("reasoning", text) / ("delta", text) while tokens arrive, ("length", "") if the context
    filled up, then ("message", assistant_msg). think=None leaves the model's own default."""
    content, reasoning, calls, finish = "", "", {}, None
    model = await models.resolve_model()  # router mode: the request must name the model
    payload = {"model": model, "messages": msgs, "tools": TOOLS, "stream": True}
    if think is not None:  # a template argument, applied by llama-server (--jinja) on every request
        payload["chat_template_kwargs"] = {"enable_thinking": think}
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=10)) as client:
        async with client.stream("POST", f"{LLAMA_URL}/v1/chat/completions",
                                 json=payload) as r:
            if r.status_code != 200:
                raise httpx.HTTPError(f"HTTP {r.status_code}: {(await r.aread()).decode(errors='replace')[:300]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                d = (choices[0].get("delta") or {}) if choices else {}
                finish = (choices[0].get("finish_reason") if choices else None) or finish
                if d.get("reasoning_content"):
                    reasoning += d["reasoning_content"]
                    yield "reasoning", d["reasoning_content"]
                if d.get("content"):
                    content += d["content"]
                    yield "delta", d["content"]
                for tc in d.get("tool_calls") or []:  # fragments, merged by index
                    slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                    f = tc.get("function") or {}
                    slot["id"] = slot["id"] or tc.get("id") or ""
                    slot["name"] = slot["name"] or f.get("name") or ""
                    a = f.get("arguments") or ""
                    slot["args"] += a if isinstance(a, str) else json.dumps(a)
    msg = {"role": "assistant", "content": content or None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [{"id": c["id"] or f"call_{i}", "type": "function",
                              "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                             for i, c in sorted(calls.items())]
    if finish == "length":
        yield "length", ""
    yield "message", msg


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def run(chat: dict, think: bool | None = None):
    """One user turn: stream tokens, run tools, persist the chat. Events are SSE `data:` JSON lines."""
    yield sse({"type": "chat", "id": chat["id"], "title": chat["title"]})
    history = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]][-MAX_HISTORY:]
    while history and history[0]["role"] != "user":  # some chat templates reject a leading assistant turn
        history.pop(0)
    msgs = [{"role": "system", "content": system_prompt()}] + history
    text, thought, trace, media, finished = "", "", [], [], False
    try:
        for _ in range(MAX_STEPS):
            fit_context(msgs)
            msg, sep, tsep, cut = None, "\n\n" if text else "", "\n\n" if thought else "", False
            async with aclosing(llm_stream(msgs, think)) as stream:  # closing it aborts the upstream request
                async for kind, val in stream:
                    if kind == "delta":
                        val, sep = sep + val, ""
                        text += val
                        yield sse({"type": "delta", "text": val})
                    elif kind == "reasoning":
                        val, tsep = tsep + val, ""
                        thought += val
                        yield sse({"type": "reasoning", "text": val})
                    elif kind == "length":
                        cut = True
                    else:
                        msg = val
            if cut:  # generation stopped because the context window is full
                note = ("\n\n" if text else "") + "[cut off: the context window is full. Start a new chat or ask for less.]"
                text += note
                finished = True
                yield sse({"type": "delta", "text": note})
                break
            if not msg.get("tool_calls"):
                finished = True
                break
            msgs.append(msg)
            for c in msg["tool_calls"]:
                name, args = c["function"]["name"], c["function"]["arguments"]
                out, new_media = await asyncio.to_thread(call_tool, name, args)
                t = {"tool": name, "args": args, "result": out[:300]}
                trace.append(t)
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
                yield sse({"type": "tool", **t})
                for m in new_media:
                    media.append(m)
                    yield sse({"type": "media", **m})
        else:
            note = "\n\n(stopped: too many tool steps. Try a more specific request.)"
            text += note
            finished = True
            yield sse({"type": "delta", "text": note})
    except Exception as e:  # llama-server down, malformed stream, ...
        yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:  # also runs when the client disconnects (Stop button, closed tab)
        if finished:
            reply = text or "(empty reply)"
        elif text:
            reply = text + "\n[interrupted]"
        else:
            reply = None
        if reply is None:
            chat["messages"].pop()  # drop the unanswered user turn so roles keep alternating
        else:
            chat["messages"].append({"role": "assistant", "content": reply, "trace": trace,
                                     **({"thinking": thought} if thought else {}),
                                     **({"media": media} if media else {})})
        save_chat(chat)
    yield sse({"type": "done"})


# ---------- API ----------
app = FastAPI(lifespan=models.lifespan)
app.include_router(models.router)


class Ask(BaseModel):
    message: str
    chat_id: str | None = None
    think: bool | None = None  # None: leave the model's own default


@app.get("/api/health")
def health():
    try:
        return {"ready": httpx.get(f"{LLAMA_URL}/health", timeout=3).status_code == 200}
    except httpx.HTTPError:
        return {"ready": False}


@app.get("/api/chats")
def list_chats():
    files = sorted(CHATS.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:50] if CHATS.is_dir() else []
    out = []
    for f in files:
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"id": f.stem, "title": c.get("title") or "(untitled)", "updated": c.get("updated", "")})
    return out


@app.get("/api/chats/{cid}")
def get_chat(cid: str):
    return load_chat(cid)


@app.delete("/api/chats/{cid}")
def delete_chat(cid: str):
    chat_path(cid).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/media/{name}")
def get_media(name: str):
    if not MEDIA_RE.fullmatch(name):  # names are made by _sd_job; also blocks path traversal
        raise HTTPException(404, "no such file")
    p = GENERATED / name
    if not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p)


@app.post("/api/chat")
def chat_turn(body: Ask):
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "empty message")
    chat = new_chat()
    if body.chat_id:
        if chat_path(body.chat_id).is_file():
            chat = load_chat(body.chat_id)
        else:  # id of a chat that was never saved (e.g. its first turn failed): start it under that id
            chat["id"] = body.chat_id
    chat["title"] = chat["title"] or " ".join(text.split())[:60]
    chat["messages"].append({"role": "user", "content": text})
    return StreamingResponse(run(chat, body.think), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
