"""One CogVideoX-2b job, run in its own process by server.py: when it ends, all of its VRAM and RAM go back to the system.

usage: python generate.py <job.json>      job: {"prompt": str, "seed": int, "steps": int, "out": "<path>.webm"}
On failure it writes the message to "<out>.error" and exits with 1.

Two stages, so that it fits an 8 GB GPU:
  1. the T5-XXL text encoder (9.5 GB in fp16, bigger than the GPU) turns the prompt into embeddings on the CPU, then is freed;
  2. the 3.4 GB transformer and the VAE take turns on the GPU (enable_model_cpu_offload) to denoise and decode.
The model fixes the format: 720x480, 49 frames at 8 fps (6 s), prompts of at most 226 tokens, English only.
"""
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL = Path(os.getenv("COG_MODEL_DIR", "/models/CogVideoX-2b"))  # the folder the Models panel downloads (sdmodels/CogVideoX-2b)
WIDTH, HEIGHT, FRAMES, FPS, GUIDANCE = 720, 480, 49, 8, 6.0      # model card values
MAX_TOKENS = 226


def embed(prompt: str):
    """(prompt, empty negative prompt) -> embeddings, fp16 as the transformer wants them."""
    import torch
    from diffusers import CogVideoXPipeline

    pipe = CogVideoXPipeline.from_pretrained(MODEL, transformer=None, vae=None, dtype=torch.bfloat16)  # text encoder only
    with torch.no_grad():
        pos, neg = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, num_videos_per_prompt=1,
                                      max_sequence_length=MAX_TOKENS, device="cpu", dtype=torch.bfloat16)
    return pos.to(torch.float16), neg.to(torch.float16)


def render(pos, neg, steps: int, seed: int, device: str = "cuda"):
    """Embeddings -> uint8 frames (FRAMES, HEIGHT, WIDTH, 3)."""
    import numpy as np
    import torch
    from diffusers import CogVideoXPipeline

    pipe = CogVideoXPipeline.from_pretrained(MODEL, text_encoder=None, tokenizer=None, dtype=torch.float16)  # no text encoder
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # transformer and VAE take turns on the GPU
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    def progress(_pipe, i, _t, kwargs):
        print(f"step {i + 1}/{steps}", flush=True)  # shows up in: docker compose logs cogvideo
        return kwargs

    dev = torch.device("cuda")
    pos = pos.to(dev)
    neg = neg.to(dev)

    frames = pipe(prompt_embeds=pos, negative_prompt_embeds=neg, height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                  num_inference_steps=steps, guidance_scale=GUIDANCE, generator=torch.Generator(device).manual_seed(seed),
                  output_type="np", callback_on_step_end=progress).frames[0]
    return (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)


def write_webm(frames, path: Path) -> None:
    import imageio_ffmpeg

    n, h, w, _ = frames.shape
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0", "-crf", "30", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


def run(job: dict, device: str = "cuda") -> None:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("no GPU visible in the cogvideo container (check the NVIDIA driver and Docker Desktop's WSL2 backend)")
    pos, neg = embed(job["prompt"])
    gc.collect()  # the text encoder is out of scope here: give its 9.5 GB back before the transformer loads
    write_webm(render(pos, neg, job["steps"], job["seed"], device), Path(job["out"]))


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        run(job)
    except Exception as e:  # the server shows this message to the agent
        err = Path(job["out"] + ".error")
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(f"{type(e).__name__}: {e}"[:500], encoding="utf-8")
        sys.exit(1)
