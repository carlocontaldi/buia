"""The Video panel's logic, with no server in it: which parameters each video engine takes, how a form becomes the body of an sd-server /
cogvideo job, and how progress becomes a phase and a percentage.

Two engines, same job API (agent.py _sd_job):
  "sd"        stable-diffusion.cpp's sd-server (Wan 2.1 / 2.2 and any other video model it runs). Its job API has no progress: the sampling
              steps are only printed in its output, which sd-run.sh copies to a log file the agent can read (LogTail + SdProgress below).
  "cogvideo"  the cogvideo service, which reports phase and step in the job it returns (see cogvideo/server.py).
The limits (largest size, frames, steps) are the ones agent.py measured on the 8 GB GPU; agent.py passes them in as `caps`.
"""
import random
import re
from pathlib import Path

LOG_NAME = "active-video.log"  # sd-run.sh names the log after its active file (active-video.args)
SAMPLERS = ("euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd",
            "res_multistep", "res_2s", "er_sde", "euler_cfg_pp", "euler_a_cfg_pp", "euler_ge", "dpm++2m_sde", "dpm++2m_sde_bt", "lms")  # sd.cpp master-920
MAX_NEGATIVE = 2000  # characters, as in cogvideo/server.py
MAX_SEED = 2**32 - 1


class BadValue(ValueError):
    """A form value the server will not run; the message is shown to the user as it is."""


# ---------- what the form shows ----------
def number(key, label, kind, lo, hi, step, default, help, hint=None):
    """default None = optional: left empty, nothing is sent and the model's own setting stays (hint says what that is)."""
    return {"key": key, "label": label, "type": kind.__name__, "min": lo, "max": hi, "step": step, "default": default, "help": help,
            "placeholder": hint or ("model default" if default is None else "")}


def fields(engine: str, caps: dict, hints: dict | None = None) -> list[dict]:
    """The parameters of one engine, in the order of the form. hints: what the model's own flags set (cfg_scale, sampler, flow_shift)."""
    hints = hints or {}
    negative = {"key": "negative_prompt", "label": "Negative prompt", "type": "text", "default": "", "placeholder": "",
                "help": "What to keep out of the video."}
    seed = number("seed", "Seed", int, -1, MAX_SEED, 1, -1, "-1 picks a random one. The same seed and settings make the same video.")
    if engine == "cogvideo":
        lo, hi = caps["cog_steps"]
        return [negative,
                number("steps", "Steps", int, lo, hi, 1, caps["cog_default_steps"], "More steps take longer and usually look cleaner (the model card uses 50)."),
                number("guidance", "Guidance scale", float, 1, 20, 0.5, 6.0, "How closely the video follows the prompt (the model card uses 6)."),
                seed]
    side, frames, steps = caps["side"], caps["frames"], caps["steps"]
    return [negative,
            number("width", "Width", int, 128, side, 16, 256, f"Multiple of 16, up to {side}."),
            number("height", "Height", int, 128, side, 16, 256, f"Multiple of 16, up to {side}."),
            number("frames", "Frames", int, 5, frames, 4, 17, f"5, 9, 13 ... up to {frames}. Frames divided by fps is the length."),
            number("fps", "Frames per second", int, 1, 60, 1, 16, "Playback speed only: it does not change the picture."),
            number("steps", "Steps", int, 1, steps, 1, 10, f"Up to {steps}. More steps take longer."),
            number("cfg_scale", "Guidance (cfg) scale", float, 1, 20, 0.5, None, "How closely the video follows the prompt.", hints.get("cfg_scale")),
            {"key": "sampler", "label": "Sampler", "type": "select", "options": list(SAMPLERS), "default": None,
             "placeholder": hints.get("sampler") or "model default", "help": "Empty keeps the sampler the model was started with."},
            number("flow_shift", "Flow shift", float, 1, 20, 0.5, None, "Timestep shift of flow models such as Wan.", hints.get("flow_shift")),
            seed]


# ---------- what the server gets ----------
def _blank(raw) -> bool:
    return raw is None or (isinstance(raw, str) and not raw.strip())


def _num(values: dict, f: dict, kind):
    """The value of field f as int or float (None when the field is optional and empty), or BadValue."""
    raw = values.get(f["key"])
    if _blank(raw):
        if f["default"] is None:
            return None
        return f["default"]
    try:
        if isinstance(raw, bool):
            raise ValueError
        n = float(raw)
    except (TypeError, ValueError):
        raise BadValue(f"{f['label']} must be a number") from None
    if not f["min"] <= n <= f["max"]:  # also false for nan
        raise BadValue(f"{f['label']} must be between {f['min']} and {f['max']}")
    if kind is int:
        if n != int(n):
            raise BadValue(f"{f['label']} must be a whole number")
        return int(n)
    return n


def build_body(engine: str, values: dict, caps: dict) -> dict:
    """The JSON body of the vid_gen job for these form values. Raises BadValue with a message for the user. The seed is always set, so the
    caller can show which one was used."""
    if not isinstance(values, dict):
        raise BadValue("the form values must be an object")
    prompt = values.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadValue("write a prompt first")
    spec = {f["key"]: f for f in fields(engine, caps)}
    negative = values.get("negative_prompt")
    negative = "" if negative is None else negative
    if not isinstance(negative, str) or len(negative) > MAX_NEGATIVE:
        raise BadValue(f"Negative prompt must be text of at most {MAX_NEGATIVE} characters")
    seed = _num(values, spec["seed"], int)
    body = {"prompt": prompt.strip(), "seed": seed if seed >= 0 else random.randrange(MAX_SEED + 1), "output_format": "webm"}
    if negative.strip():
        body["negative_prompt"] = negative.strip()
    steps = _num(values, spec["steps"], int)
    if engine == "cogvideo":
        w, h, n = caps["cog_format"]  # the one format the model makes
        body.update(width=w, height=h, video_frames=n,
                    sample_params={"sample_steps": steps, "guidance": {"txt_cfg": _num(values, spec["guidance"], float)}})
        return body
    width, height = _num(values, spec["width"], int), _num(values, spec["height"], int)
    if width % 16 or height % 16:
        raise BadValue("Width and height must be multiples of 16")
    frames = _num(values, spec["frames"], int)
    if (frames - 1) % 4:
        raise BadValue(f"Frames must be 5, 9, 13 ... up to {caps['frames']}")
    sample = {"sample_steps": steps}
    cfg, flow = _num(values, spec["cfg_scale"], float), _num(values, spec["flow_shift"], float)
    if cfg is not None:
        sample["guidance"] = {"txt_cfg": cfg}
    if flow is not None:
        sample["flow_shift"] = flow
    sampler = values.get("sampler")
    if not _blank(sampler):
        if sampler not in SAMPLERS:
            raise BadValue(f"Sampler must be one of: {', '.join(SAMPLERS)}")  # sd-server would ignore an unknown name without a word
        sample["sample_method"] = sampler
    body.update(width=width, height=height, video_frames=frames, fps=_num(values, spec["fps"], int), sample_params=sample)
    return body


# ---------- how far a job is ----------
# One bar for all phases: each phase owns a slice of it, and inside the slice the bar follows the real step counter when the phase has one.
# The slices are only a layout (not measured shares of the time): the text under the bar always shows the real counter.
SHARE = {"queued": (0, 0), "starting": (0, 10), "text": (0, 10), "loading": (10, 20), "sampling": (20, 80), "decoding": (80, 98), "saving": (98, 100)}


def percent(phase: str, step: int = 0, steps: int = 0) -> int:
    lo, hi = SHARE[phase]
    return round(lo + (hi - lo) * min(step, steps) / steps) if steps > 0 else lo


# sd.cpp prints  \r  |=====>     | 3/10 - 1.52s/it\x1b[K  per sampling step and per VAE tile, and the same with # and MB/s while it loads
# weights (examples: src/core/util.cpp print_progress_line). Only the unit tells the two apart.
LINE = re.compile(r"\|[=>#\s]*\|\s*(\d+)/(\d+)\s*-\s*[\d.]+(s/it|it/s|MB/s|GB/s)")


class SdProgress:
    """Reads sd-server's output in the order it was printed and says where the job is: loading weights, sampling (the first series of
    step counters), then decoding (a later series: VAE tiles) or, with no tiles, the time after the last sampling step."""

    def __init__(self):
        self.loading = self.sampling = self.decoding = None  # (step, total) of the latest counter of that kind
        self.series = 0                                      # run-series seen so far (sampling = 1, VAE tiles = 2)
        self.last = None                                     # (step, total) of the latest run-series line

    def feed(self, text: str) -> None:
        for m in LINE.finditer(text):
            step, total, unit = int(m[1]), int(m[2]), m[3]
            if unit.endswith("B/s"):
                self.loading = (step, total)  # shown only until sampling starts (snapshot prefers the later phases)
                continue
            if self.last and total == self.last[1] and step <= self.last[0]:
                if step < self.last[0]:  # the counter started again: a new series
                    self.series += 1
                else:  # the same line twice
                    continue
            elif not self.last or total != self.last[1]:
                self.series += 1
            self.last = (step, total)
            if self.series == 1:
                self.sampling = self.last
            else:
                self.decoding = self.last

    def snapshot(self) -> tuple[str, int, int]:
        """(phase, step, steps)"""
        if self.decoding:
            return "decoding", *self.decoding
        if self.sampling:
            step, total = self.sampling
            return ("sampling", step, total) if step < total else ("decoding", 0, 0)
        if self.loading:
            return "loading", *self.loading
        return "starting", 0, 0


class LogTail:
    """New text of a log file since the object was made (what sd-server printed during this job). A file that got shorter was emptied
    (sd-run.sh does that when the server restarts): reading starts again from its beginning."""

    def __init__(self, path):
        self.path = Path(path)
        self.pos = self._size()

    def _size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> str:
        size = self._size()
        if size < self.pos:
            self.pos = 0
        if size == self.pos:
            return ""
        try:
            with self.path.open("rb") as f:
                f.seek(self.pos)
                data = f.read(min(size - self.pos, 1_000_000))
        except OSError:
            return ""
        cut = data.rfind(b"\r")  # a counter line still being written is read next time, whole
        if cut >= 0 and not LINE.search(data[cut:].decode("utf-8", "replace")):
            data = data[:cut]
        self.pos += len(data)
        return data.decode("utf-8", "replace")
