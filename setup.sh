#!/usr/bin/env bash
# Installs or updates BUIA on Linux, whichever applies: no core/.env yet means install, otherwise update (git pull, rebuild, restart).
# Run it from anywhere:  ./setup.sh [--sd] [--cogvideo] [--tailscale] [--llama] [--no-pull]
#   --sd          also set up image/video generation (clones stable-diffusion.cpp and builds the sd service; the first build is long)
#   --cogvideo    also build the cogvideo service (CogVideoX-2b text-to-video, PyTorch; the first build downloads several GB)
#   --tailscale   also expose the web UI to your tailnet (phone access)
#   --llama       also show the digest of the newest llama.cpp server image (docker-compose.yml pins one; README > Update)
#   --no-pull     update without git pull (you have local commits, or this is not a git clone)
# An existing .env or BUIA.md is never overwritten. Run it as your normal user, not with sudo: it needs the docker group, not root.
# Windows has install.cmd and update.cmd for the same two jobs.

set -eu
set -o pipefail

SD_TAG=master-920-2f88688                                       # stable-diffusion.cpp release the sd service is built from (keep equal to $SdTag in core/install.ps1)
LLAMA_DEFAULT=ghcr.io/ggml-org/llama.cpp:server-cuda13         # keep equal to the tag in core/docker-compose.yml

say() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

main() {
  sd=0 cogvideo=0 tailscale=0 llama=0 pull=1
  for arg in "$@"; do
    case $arg in
      --sd) sd=1 ;;
      --cogvideo) cogvideo=1 ;;
      --tailscale) tailscale=1 ;;
      --llama) llama=1 ;;
      --no-pull) pull=0 ;;
      -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; return 0 ;;
      *) die "unknown option: $arg (./setup.sh --help lists them)" ;;
    esac
  done

  root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  cd "$root/core"
  [ -f docker-compose.yml ] || die "core/docker-compose.yml is missing next to this script: run it from a clone of the repository."

  # a NAME=value line of .env, the first one with a value (like docker compose, no quotes); empty when there is none
  env_value() { sed -n "/^$1=[^[:space:]]/{s/^$1=//;p;q;}" .env | tr -d '\r'; }
  # NAME=value in .env: replaces the line, or adds it
  set_env() {
    if grep -q "^$1=" .env; then
      sed -i "s|^$1=.*|$1=$2|" .env
    else
      [ -z "$(tail -c 1 .env)" ] || echo >> .env   # the last line may lack its newline
      printf '%s=%s\n' "$1" "$2" >> .env
    fi
  }

  # 1. Install or update?
  if [ -f .env ]; then mode=update; else mode=install; fi
  say "BUIA $mode ($root)"

  # 2. Update: get the newest version first, so the rest of this run is about it. A zip download has no .git and cannot update.
  if [ "$mode" = update ] && [ "$pull" -eq 1 ]; then
    if git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git -C "$root" pull --ff-only || die "git pull failed. Fix that, or run ./setup.sh --no-pull to rebuild what you have."
    else
      say "Not a git clone: nothing to pull."
    fi
  fi

  # 3. Docker must work for this user
  command -v docker >/dev/null 2>&1 || die "Docker is not installed. Install Docker Engine: https://docs.docker.com/engine/install/"
  docker compose version >/dev/null 2>&1 || die "The Docker Compose plugin is missing (docker compose version fails). Install docker-compose-plugin, or Docker Engine from docker.com."
  if ! out=$(docker info --format '{{.ServerVersion}}' 2>&1); then
    case $out in
      *"permission denied"*) die "Docker refuses this user. Add yourself to the docker group: sudo usermod -aG docker \$USER, then log out and in (or: newgrp docker)." ;;
      *) die "Docker is not running: sudo systemctl start docker. Docker said: $out" ;;
    esac
  fi

  # 4. Warn early if containers cannot use the GPU
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi is not here: install the NVIDIA driver (the llama.cpp image needs an NVIDIA GPU)."
  else
    if ! docker run --rm --gpus all ubuntu nvidia-smi >/dev/null 2>&1; then
      warn "Docker cannot see the NVIDIA GPU. Install the NVIDIA Container Toolkit, then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
  fi

  # 5. .env: created once from .env.example, never overwritten
  if [ ! -f .env ]; then
    cp .env.example .env
    say "Created .env"
  fi

  # 5a. The pinned llama.cpp image is built for CUDA 13, which needs NVIDIA driver 580 or newer
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n 1p || true)
  if [ -z "${LLAMA_IMAGE:-}" ] && [ -z "$(env_value LLAMA_IMAGE)" ] && [ "${driver%%.*}" -lt 580 ] 2>/dev/null; then
    warn "NVIDIA driver $driver is older than 580, which the pinned CUDA 13 llama.cpp image needs. Put LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda in core/.env (README > Requirements), then run this again."
  fi

  # 5b. Web UI password: a random one if .env has none, never overwritten
  if ! grep -q '^BUIA_PASSWORD=' .env; then
    pw=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')   # 32 hex characters: no $ or # for Compose to trip over
    echo >> .env
    set_env BUIA_PASSWORD "$pw"
    say "Web UI password: $pw   (saved in .env as BUIA_PASSWORD; change it there)"
  elif ! grep -q '^BUIA_PASSWORD=.' .env; then
    warn "BUIA_PASSWORD in .env is empty, so nobody can log in. Put a password after the = and run this again."
  fi

  # 5c. The agent container runs as you, so the files it writes in your buia folder are yours (docker-compose.yml: user:). As root it stays root.
  if [ "$(id -u)" -ne 0 ]; then
    grep -q '^BUIA_UID=' .env || set_env BUIA_UID "$(id -u)"
    grep -q '^BUIA_GID=' .env || set_env BUIA_GID "$(id -g)"
  else
    warn "Running as root: the files BUIA writes will belong to root. Run it as your normal user (in the docker group) instead."
  fi

  # 6. Folders and memory file the containers mount. As in docker-compose.yml, BUIA_PATH defaults to the folder above core and VAULT_PATH to .vault inside it.
  # sdmodels/ is made here, always: if Docker made it, it would belong to root and the agent could not download into it.
  buia=$(env_value BUIA_PATH); [ -n "$buia" ] || buia=$(cd .. && pwd)
  vault=$(env_value VAULT_PATH); [ -n "$vault" ] || vault="$buia/.vault"
  mkdir -p "$buia" "$vault" sdmodels
  if [ ! -f "$buia/BUIA.md" ]; then
    cp ../templates/BUIA.md "$buia/BUIA.md"
    say "Created $buia/BUIA.md from the template"
  fi
  # Files an earlier run as root left behind cannot be written by the agent now
  if [ "$(id -u)" -ne 0 ]; then
    stuck=$(find "$buia/chats" "$buia/generated" "$buia/memory" "$buia/.buia-settings.json" sdmodels -not -writable -print -quit 2>/dev/null || true)
    if [ -n "$stuck" ]; then
      warn "BUIA cannot write to $stuck (and maybe more): it belongs to another user, probably root from an earlier run. Fix: sudo chown -R $(id -u):$(id -g) \"$buia/chats\" \"$buia/generated\" \"$buia/memory\" \"$buia/.buia-settings.json\" \"$PWD/sdmodels\""
    fi
  fi

  # 7. Optional: image/video generation. The sd service is rebuilt whenever sdcpp/ exists, so an update keeps it.
  profiles=()
  if [ "$sd" -eq 1 ] && [ ! -d sdcpp ]; then
    command -v git >/dev/null 2>&1 || die "git is needed to fetch stable-diffusion.cpp."
    if ! git -c advice.detachedHead=false clone --branch "$SD_TAG" --depth 1 --recurse-submodules --shallow-submodules https://github.com/leejet/stable-diffusion.cpp sdcpp; then
      rm -rf sdcpp
      die "git clone of stable-diffusion.cpp failed"
    fi
  fi
  if [ -d sdcpp ]; then
    profiles+=(--profile sd)
    # the sd image is compiled for one GPU architecture (docker-compose.yml: SD_CUDA_ARCHITECTURES, default 120 = RTX 50 series)
    if ! grep -q '^SD_CUDA_ARCHITECTURES=' .env; then
      cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n 1p | tr -d '. \r' || true)
      if [ -n "$cap" ]; then
        set_env SD_CUDA_ARCHITECTURES "$cap"
        say "Set SD_CUDA_ARCHITECTURES=$cap in .env (your GPU's compute capability)."
      else
        warn "Could not read the GPU's compute capability: the sd image is built for 120 (RTX 50 series). Set SD_CUDA_ARCHITECTURES in .env if yours differs."
      fi
    fi
    if [ "$sd" -eq 1 ]; then
      say "Download an image and a video model in the web UI (Settings > Image and video models), or put the files named in SD_ARGS / SD_VIDEO_ARGS (.env) into $PWD/sdmodels"
    fi
  fi

  # 7b. Optional: CogVideoX-2b. The flag lives in .env so that the next update keeps rebuilding the service.
  if [ "$cogvideo" -eq 1 ]; then set_env COGVIDEO on; fi
  if [ "$(env_value COGVIDEO)" = on ]; then
    profiles+=(--profile cogvideo)
    if [ "$cogvideo" -eq 1 ]; then
      say "Download CogVideoX 2B in the web UI (Settings > Image and video models), then click Use. See README > CogVideoX before the first video."
    fi
  fi

  # 8. Optional: what the newest llama.cpp image is now (the compose file keeps the pinned one)
  if [ "$llama" -eq 1 ]; then
    image=$(env_value LLAMA_IMAGE); image=${image:-$LLAMA_DEFAULT}; latest=${image%@*}
    docker pull "$latest" || die "docker pull $latest failed"
    say "Newest $latest is:"
    docker image inspect "$latest" --format '{{index .RepoDigests 0}}'
  fi

  # 9. Build and start
  docker compose "${profiles[@]}" up -d --build || die "docker compose up failed"

  # 10. Wait until llama-server answers (up to 2 minutes)
  if command -v curl >/dev/null 2>&1; then
    probe() { curl -fsS --max-time 3 http://127.0.0.1:8000/api/health 2>/dev/null; }
  elif command -v wget >/dev/null 2>&1; then
    probe() { wget -qO- -T 3 http://127.0.0.1:8000/api/health 2>/dev/null; }
  else
    probe() { return 1; }
  fi
  say "Waiting for BUIA..."
  ready=0
  deadline=$((SECONDS + 120))
  while [ "$SECONDS" -lt "$deadline" ]; do
    body=$(probe || true)
    case $body in *'"ready":true'*|*'"ready": true'*) ready=1; break ;; esac
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    warn "BUIA is not ready after 120 seconds (the model may still be loading, or curl and wget are missing here). Current services and recent logs:"
    docker compose "${profiles[@]}" ps
    docker compose "${profiles[@]}" logs --tail 50 llama agent
  fi

  # 11. Optional: phone access over Tailscale
  if [ "$tailscale" -eq 1 ]; then
    tailscale serve --bg 8000 || die "tailscale serve failed (a non-root user may need: sudo tailscale set --operator=\$USER)"
    tailscale serve status
  fi

  say ""
  if [ "$mode" = update ]; then
    docker compose "${profiles[@]}" ps
    say "Updated. If a release note says new settings were added, compare core/.env with core/.env.example."
  fi
  say "BUIA is at http://127.0.0.1:8000  -  open Settings to download and select a model."
}

# Everything above is one function and the script ends here, so bash has read it all before git pull can replace this file.
main "$@"
exit
