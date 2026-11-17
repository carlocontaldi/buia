"""One CogVideoX-2b job, run in its own process by server.py: when it ends, all of its VRAM and RAM go back to the system.

usage: python generate.py <job.json>      job: {"prompt": str, "seed": int, "steps": int, "out": "<path>.webm", optional "negative": str, "guidance": float}
On failure it writes the message to "<out>.error" and exits with 1. While it runs it keeps "<out>.progress" up to date (server.py shows it to the dashboard).

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


def report(out: str, phase: str, step: int = 0, steps: int = 0) -> None:
    """Tells server.py where this job is. The file is replaced in one step, so a reader never sees half of it."""
    tmp = out + ".progress.tmp"
    try:
        Path(tmp).write_text(json.dumps({"phase": phase, "step": step, "steps": steps}), encoding="utf-8")
        os.replace(tmp, out + ".progress")
    except OSError:
        pass  # progress is only information: a full disk must not cost the video


def embed(prompt: str, negative: str = ""):
    """(prompt, negative prompt) -> embeddings, fp16 as the transformer wants them."""
    import torch
    from diffusers import CogVideoXPipeline

    pipe = CogVideoXPipeline.from_pretrained(MODEL, transformer=None, vae=None, dtype=torch.bfloat16)  # text encoder only
    with torch.no_grad():
        pos, neg = pipe.encode_prompt(prompt, negative_prompt=negative, do_classifier_free_guidance=True, num_videos_per_prompt=1,
                                      max_sequence_length=MAX_TOKENS, device="cpu", dtype=torch.bfloat16)
    return pos.to(torch.float16), neg.to(torch.float16)


def render(pos, neg, steps: int, seed: int, device: str = "cuda", guidance: float = GUIDANCE, report=lambda *a: None):
    """Embeddings -> uint8 frames (FRAMES, HEIGHT, WIDTH, 3). report(phase, step, steps) is called as it goes."""
    import numpy as np
    import torch
    from diffusers import CogVideoXPipeline

    report("loading")
    pipe = CogVideoXPipeline.from_pretrained(MODEL, text_encoder=None, tokenizer=None, dtype=torch.float16)  # no text encoder
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # transformer and VAE take turns on the GPU
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    def progress(_pipe, i, _t, kwargs):
        print(f"step {i + 1}/{steps}", flush=True)  # shows up in: docker compose logs cogvideo
        report("sampling", i + 1, steps)
        if i + 1 == steps:
            report("decoding")  # the VAE decodes inside the pipe call, after the last step
        return kwargs

    dev = torch.device("cuda")
    pos = pos.to(dev)
    neg = neg.to(dev)

    report("sampling", 0, steps)
    frames = pipe(prompt_embeds=pos, negative_prompt_embeds=neg, height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                  num_inference_steps=steps, guidance_scale=guidance, generator=torch.Generator(device).manual_seed(seed),
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
    def say(phase: str, step: int = 0, steps: int = 0) -> None:
        report(job["out"], phase, step, steps)

    say("text")
    pos, neg = embed(job["prompt"], job.get("negative", ""))
    gc.collect()  # the text encoder is out of scope here: give its 9.5 GB back before the transformer loads
    frames = render(pos, neg, job["steps"], job["seed"], device, job.get("guidance", GUIDANCE), say)
    say("saving")
    write_webm(frames, Path(job["out"]))


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        run(job)
    except Exception as e:  # the server shows this message to the agent
        err = Path(job["out"] + ".error")
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(f"{type(e).__name__}: {e}"[:500], encoding="utf-8")
        sys.exit(1)
