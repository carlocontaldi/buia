# BUIA

A local personal assistant: llama.cpp (router mode, CUDA) plus a small FastAPI agent and a web UI.
It reads and writes your notes through file tools, remembers things in `BUIA.md`, streams replies,
keeps chat history as JSON files, and can be reached from your phone over Tailscale.

## Requirements

- Windows 11, Docker Desktop (WSL2 backend) with NVIDIA GPU support
- NVIDIA driver recent enough for CUDA 13 (the compose file uses `server-cuda13`; for an older driver change the image to `server-cuda`)
- Git. Tailscale if you want phone access.

## Install

```powershell
cd C:\Users\conta\buia
git clone <repo-url> buia-stack
cd buia-stack
.\install.ps1              # add -Tailscale to expose it to your tailnet, -Sd for image/video generation
```

`install.ps1` checks Docker and GPU access, creates `.env` from `.env.example` (with `BUIA_PATH` set to the folder
above the repo), creates `BUIA.md` from `templates/BUIA.md` if it does not exist, builds and starts the stack,
and waits until it answers. It never overwrites an existing `.env` or `BUIA.md`.

Then open http://127.0.0.1:8000, click **Models**, check a Hugging Face repo, download a quant marked `fits`, click **Use**.

## Update

```powershell
cd C:\Users\conta\buia\buia-stack
.\update.ps1               # git pull + rebuild + restart
.\update.ps1 -Llama        # same, and pull the newest llama.cpp server image
```

## Phone (Tailscale)

`.\install.ps1 -Tailscale`, or by hand: `tailscale serve --bg 8000`. Open the `https://...ts.net` address from
`tailscale serve status` on a phone signed in to the same tailnet. Use `serve`, never `funnel`: the agent has no login
and can overwrite files.

## What lives where

| In the repo | Outside the repo (never committed) |
|---|---|
| `agent/` code and web UI, `docker-compose.yml`, scripts, `templates/BUIA.md` | `.env` (paths, token), `BUIA.md`, `chats/`, `generated/`, `.buia-settings.json`, your notes, downloaded chat models (Docker volume `models`), `sdcpp/` and `sdmodels/` |

## Settings

All settings are in `.env` (see `.env.example` for each one). Common ones: `CTX_SIZE`, `KV_CACHE_TYPE`,
`MEMORY_FILES` (which `.md` files are loaded on every request), `GPU_VRAM_MB` (only if the Models panel cannot read the GPU).

## Troubleshooting

- `docker compose config` prints the resolved configuration and catches path or syntax mistakes.
- `docker compose logs llama --tail 50` and `docker compose logs agent --tail 50` show what each container is doing.
- Models panel says the GPU is not detected: set `GPU_VRAM_MB` in `.env`, then `docker compose up -d`.

## Image and video generation (optional)

`.\install.ps1 -Sd` clones stable-diffusion.cpp (pinned release, with submodules) into `sdcpp\`, creates `sdmodels\`,
and builds the `sd` service from `docker-compose.yml`. The first build compiles it for CUDA and takes a while.

1. Put the model files named in `SD_ARGS` (`.env`) into `sdmodels\`. `.env.example` has a Z-Image Turbo (images) line and a Wan 2.1 (video) line.
2. `docker compose --profile sd up -d` after changing `SD_ARGS`. The sd server runs one model at a time: an image model
   cannot make videos and the other way round.
3. Ask the assistant for an image or a video. Before each job the chat model is unloaded to free VRAM and loads again on your next message.
   Results are saved in `generated\` inside your workspace folder and shown inline in the chat.

`tools\sd_smoke.py` runs one small video job against the sd server to check it end to end (instructions at the top of the file).
`.\update.ps1` rebuilds the sd service too once `sdcpp\` exists.
