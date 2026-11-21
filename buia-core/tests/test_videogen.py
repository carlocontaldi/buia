"""Tests for agent/videogen.py: the parameters of each video engine, how a form becomes a job body, and how progress is read (standard library only).

Run from buia-core:  python -m unittest discover -s tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if AGENT_DIR.is_dir():
    sys.path.insert(0, str(AGENT_DIR))
import videogen  # noqa: E402

CAPS = {"side": 512, "frames": 33, "steps": 20, "cog_steps": (10, 50), "cog_default_steps": 50, "cog_format": (720, 480, 49)}


def counter(step, total, unit="s/it", char="=", head=True):
    """One counter line exactly as stable-diffusion.cpp prints it (src/core/util.cpp build_progress_bar + print_progress_line)."""
    cur = int(step * 50 / total)
    bar = "".join(" " if i > cur else ">" if head and i == cur and i != 49 else char for i in range(50))
    return f"\r  |{bar}| {step}/{total} - 1.52{unit}\x1b[K" + ("\n" if step == total else "")


def loading(step, total):
    return counter(step, total, "MB/s", "#", head=False)


def series(total, **kw):
    return "".join(counter(i, total, **kw) for i in range(1, total + 1))


class FieldTests(unittest.TestCase):
    def keys(self, engine):
        return [f["key"] for f in videogen.fields(engine, CAPS)]

    def test_each_engine_has_its_own_parameters(self):
        self.assertEqual(self.keys("sd"), ["negative_prompt", "width", "height", "frames", "fps", "steps", "cfg_scale", "sampler", "flow_shift", "seed"])
        self.assertEqual(self.keys("cogvideo"), ["negative_prompt", "steps", "guidance", "seed"])

    def test_limits_come_from_the_caps(self):
        f = {f["key"]: f for f in videogen.fields("sd", CAPS)}
        self.assertEqual((f["width"]["max"], f["frames"]["max"], f["steps"]["max"]), (512, 33, 20))
        c = {f["key"]: f for f in videogen.fields("cogvideo", CAPS)}
        self.assertEqual((c["steps"]["min"], c["steps"]["max"], c["steps"]["default"]), (10, 50, 50))

    def test_defaults_are_the_ones_generate_video_uses(self):
        f = {f["key"]: f["default"] for f in videogen.fields("sd", CAPS)}
        self.assertEqual((f["width"], f["height"], f["frames"], f["fps"], f["steps"]), (256, 256, 17, 16, 10))

    def test_optional_fields_say_what_the_model_uses(self):
        f = {f["key"]: f for f in videogen.fields("sd", CAPS, {"cfg_scale": "6.0", "sampler": "euler"})}
        self.assertEqual((f["cfg_scale"]["default"], f["cfg_scale"]["placeholder"], f["sampler"]["placeholder"]), (None, "6.0", "euler"))
        self.assertEqual(f["flow_shift"]["placeholder"], "model default")


class BuildBodyTests(unittest.TestCase):
    def body(self, engine="sd", **values):
        return videogen.build_body(engine, {"prompt": "a quiet forest"} | values, CAPS)

    def bad(self, engine="sd", **values):
        with self.assertRaises(videogen.BadValue) as e:
            self.body(engine, **values)
        return str(e.exception)

    def test_defaults(self):
        b = self.body(seed=7)
        self.assertEqual(b, {"prompt": "a quiet forest", "seed": 7, "output_format": "webm", "width": 256, "height": 256, "video_frames": 17,
                             "fps": 16, "sample_params": {"sample_steps": 10}})

    def test_form_strings_are_converted_and_optional_ones_sent_only_when_filled(self):
        b = self.body(width="320", height="192.0", frames="9", fps="24", steps="12", cfg_scale="5.5", sampler="euler_a", flow_shift="3",
                      negative_prompt="  blurry ", seed="42")
        self.assertEqual((b["width"], b["height"], b["video_frames"], b["fps"], b["seed"], b["negative_prompt"]), (320, 192, 9, 24, 42, "blurry"))
        self.assertEqual(b["sample_params"], {"sample_steps": 12, "guidance": {"txt_cfg": 5.5}, "flow_shift": 3.0, "sample_method": "euler_a"})
        b = self.body(cfg_scale="", sampler="", flow_shift=None, negative_prompt="  ")
        self.assertEqual(b["sample_params"], {"sample_steps": 10})
        self.assertNotIn("negative_prompt", b)

    def test_random_seed_is_chosen_here_so_it_can_be_shown(self):
        seeds = {self.body()["seed"] for _ in range(5)} | {self.body(seed="-1")["seed"]}
        self.assertTrue(all(0 <= s <= videogen.MAX_SEED for s in seeds))
        self.assertGreater(len(seeds), 1)

    def test_what_the_gpu_cannot_do_is_refused(self):
        for values, word in (({"width": 640}, "Width"), ({"height": 100}, "Height"), ({"width": 200}, "multiples of 16"), ({"frames": 37}, "Frames"),
                             ({"frames": 6}, "5, 9, 13"), ({"steps": 21}, "Steps"), ({"steps": 0}, "Steps"), ({"steps": 2.5}, "whole number")):
            with self.subTest(values=values):
                self.assertIn(word, self.bad(**values))

    def test_junk_is_refused_with_a_message(self):
        for values, word in (({"width": "wide"}, "Width"), ({"width": True}, "Width"), ({"steps": "nan"}, "Steps"), ({"cfg_scale": "inf"}, "cfg"),
                             ({"sampler": "magic"}, "Sampler"), ({"negative_prompt": "x" * 2001}, "Negative"), ({"negative_prompt": 5}, "Negative"),
                             ({"seed": -2}, "Seed"), ({"flow_shift": 0}, "Flow shift")):
            with self.subTest(values=values):
                self.assertIn(word, self.bad(**values))

    def test_a_prompt_is_needed(self):
        for prompt in (None, "", "   ", 5):
            with self.subTest(prompt=prompt):
                with self.assertRaises(videogen.BadValue):
                    videogen.build_body("sd", {"prompt": prompt}, CAPS)
        with self.assertRaises(videogen.BadValue):
            videogen.build_body("sd", ["not", "an", "object"], CAPS)

    def test_cogvideo_keeps_its_format_and_takes_steps_guidance_negative_prompt(self):
        b = self.body("cogvideo", steps="30", guidance="4.5", negative_prompt="blurry", seed="3", width="64", frames="5")  # size fields are not its own
        self.assertEqual(b, {"prompt": "a quiet forest", "seed": 3, "output_format": "webm", "negative_prompt": "blurry", "width": 720, "height": 480,
                             "video_frames": 49, "sample_params": {"sample_steps": 30, "guidance": {"txt_cfg": 4.5}}})
        b = self.body("cogvideo", seed=1)
        self.assertEqual(b["sample_params"], {"sample_steps": 50, "guidance": {"txt_cfg": 6.0}})

    def test_cogvideo_limits(self):
        for values, word in (({"steps": 9}, "Steps"), ({"steps": 51}, "Steps"), ({"guidance": 0.5}, "Guidance"), ({"guidance": 21}, "Guidance")):
            with self.subTest(values=values):
                self.assertIn(word, self.bad("cogvideo", **values))


class PercentTests(unittest.TestCase):
    def test_the_bar_only_goes_forward_through_the_phases(self):
        walk = [("starting", 0, 0), ("loading", 1, 3), ("loading", 3, 3), ("sampling", 0, 10), ("sampling", 5, 10), ("sampling", 10, 10),
                ("decoding", 0, 0), ("decoding", 1, 4), ("decoding", 4, 4), ("saving", 0, 0)]
        values = [videogen.percent(*w) for w in walk]
        self.assertEqual(values, sorted(values))
        self.assertEqual((values[0], values[-1]), (0, 98))

    def test_counters_move_the_bar_inside_their_phase(self):
        self.assertEqual([videogen.percent("sampling", s, 10) for s in (0, 5, 10)], [20, 50, 80])
        self.assertEqual(videogen.percent("sampling", 99, 10), 80)  # never past the end of the phase
        self.assertEqual(videogen.percent("sampling", 3, 0), 20)    # no total yet


class SdProgressTests(unittest.TestCase):
    def test_nothing_printed_yet(self):
        self.assertEqual(videogen.SdProgress().snapshot(), ("starting", 0, 0))

    def test_the_test_helper_prints_what_sd_cpp_prints(self):
        self.assertEqual(counter(3, 10), "\r  |" + "=" * 15 + ">" + " " * 34 + "| 3/10 - 1.52s/it\x1b[K")
        self.assertTrue(counter(10, 10).endswith("| 10/10 - 1.52s/it\x1b[K\n"))

    def test_loading_then_sampling_then_tiles(self):
        p = videogen.SdProgress()
        p.feed("some log line\n" + loading(5, 40))
        self.assertEqual(p.snapshot(), ("loading", 5, 40))
        p.feed(loading(40, 40) + "[INFO] sampling\n" + counter(1, 10) + counter(4, 10))
        self.assertEqual(p.snapshot(), ("sampling", 4, 10))
        p.feed("".join(counter(i, 10) for i in range(5, 11)))
        self.assertEqual(p.snapshot(), ("decoding", 0, 0))  # sampled, nothing counts the decode
        p.feed("[INFO] decoding\n" + counter(1, 6, "it/s"))
        self.assertEqual(p.snapshot(), ("decoding", 1, 6))
        p.feed(series(6)[len(counter(1, 6)):])
        self.assertEqual(p.snapshot(), ("decoding", 6, 6))

    def test_tiles_with_as_many_counts_as_steps_are_still_a_new_series(self):
        p = videogen.SdProgress()
        p.feed(series(8) + "[INFO] decode\n" + counter(1, 8) + counter(2, 8))
        self.assertEqual(p.snapshot(), ("decoding", 2, 8))

    def test_a_line_printed_twice_counts_once(self):
        p = videogen.SdProgress()
        p.feed(counter(1, 10) + counter(2, 10) + counter(2, 10))
        self.assertEqual(p.snapshot(), ("sampling", 2, 10))

    def test_weight_loading_after_sampling_started_is_not_shown(self):
        p = videogen.SdProgress()
        p.feed(counter(2, 10) + loading(1, 9))
        self.assertEqual(p.snapshot(), ("sampling", 2, 10))

    def test_text_that_only_looks_like_progress_is_ignored(self):
        p = videogen.SdProgress()
        p.feed("[DEBUG] 3/10 - done\n| 3/10 | 5/10 - fast\n")
        self.assertEqual(p.snapshot(), ("starting", 0, 0))


class LogTailTests(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = Path(d.name) / "active-video.log"

    def append(self, text):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)

    def test_only_what_is_written_after_it_was_made(self):
        self.append("old job " + counter(9, 10))
        tail = videogen.LogTail(self.path)
        self.assertEqual(tail.read(), "")
        self.append("new")
        self.assertEqual(tail.read(), "new")
        self.assertEqual(tail.read(), "")

    def test_a_file_that_does_not_exist_yet(self):
        tail = videogen.LogTail(self.path)
        self.assertFalse(tail.exists())
        self.assertEqual(tail.read(), "")
        self.append("hello")
        self.assertTrue(tail.exists())
        self.assertEqual(tail.read(), "hello")

    def test_an_emptied_file_is_read_from_its_start_again(self):  # sd-run.sh empties it when the server restarts or the log is too big
        self.append("x" * 100)
        tail = videogen.LogTail(self.path)
        self.path.write_text("", encoding="utf-8")
        self.append("fresh")
        self.assertEqual(tail.read(), "fresh")

    def test_a_counter_line_cut_in_half_is_read_whole_next_time(self):
        tail = videogen.LogTail(self.path)
        line = counter(3, 10)
        self.append("start\n" + counter(2, 10) + line[:20])
        first = tail.read()
        self.assertIn("2/10", first)
        self.assertNotIn("3/10", first)
        self.append(line[20:])
        self.assertEqual(tail.read(), line)

    def test_real_progress_through_the_tail(self):
        tail, p = videogen.LogTail(self.path), videogen.SdProgress()
        for chunk in (loading(1, 2), loading(2, 2), counter(1, 3), counter(2, 3), counter(3, 3), counter(1, 2, "it/s")):
            self.append(chunk)
            p.feed(tail.read())
        self.assertEqual(p.snapshot(), ("decoding", 1, 2))


if __name__ == "__main__":
    unittest.main()
