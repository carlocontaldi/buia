"""Tests for sd-run.sh with a fake sd-server (a shell script that only notes how it was started), and for sdmodels.restart_server, which
makes the script restart the server. They need a POSIX sh and GNU stat (Linux, as in the sd containers); sdmodels needs fastapi and httpx.
The script polls every 2 s, so each test takes a few seconds.

Run from the repo root:  python -m unittest discover -s tests
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
except ImportError:
    raise unittest.SkipTest("fastapi and httpx are not installed here; run these tests in the agent container") from None
if not sys.platform.startswith("linux") or not shutil.which("sh"):
    raise unittest.SkipTest("sd-run.sh needs Linux (GNU stat) and sh")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import sdmodels  # noqa: E402


class SdRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.active = self.tmp / "active-video.args"
        self.starts = self.tmp / "starts"  # one line per start of the fake server: the flags it got
        server = self.tmp / "fake-sd-server"
        server.write_text(f'#!/bin/sh\necho "$@" >> {self.starts}\nexec sleep 300\n', encoding="utf-8")
        server.chmod(0o755)
        self.env = os.environ | {"SD_SERVER": str(server), "SD_LOG_DIR": str(self.tmp / "logs")}
        self.procs = []
        self.addCleanup(self.stop_all)

    def stop_all(self):
        for p in self.procs:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        subprocess.run(["pkill", "-f", f"sleep 300.*{self.tmp.name}"], check=False)  # a fake server that outlived its script

    def run_script(self, *defaults):
        out = (self.tmp / "out").open("w")
        self.addCleanup(out.close)
        p = subprocess.Popen(["sh", str(ROOT / "sd-run.sh"), str(self.active), *defaults], env=self.env, stdout=out, stderr=subprocess.STDOUT)
        self.procs.append(p)
        return p

    def started(self):
        return self.starts.read_text(encoding="utf-8").splitlines() if self.starts.exists() else []

    def wait_starts(self, n, seconds=12):
        end = time.time() + seconds
        while time.time() < end and len(self.started()) < n:
            time.sleep(0.1)
        self.assertEqual(len(self.started()), n, (self.tmp / "out").read_text(encoding="utf-8"))

    def test_waits_for_a_model_file_that_is_not_there_yet_and_then_starts_by_itself(self):
        model = self.tmp / "m.gguf"
        p = self.run_script("--diffusion-model", str(model), "--lora-model-dir", str(self.tmp / "loras"))  # a missing folder is no reason to wait
        time.sleep(3)
        self.assertEqual((self.started(), p.poll()), ([], None))  # not started, and the container does not end (Docker would restart it)
        out = (self.tmp / "out").read_text(encoding="utf-8")
        self.assertEqual(out.count("waiting for"), 1)  # said once, not every 2 s
        self.assertIn("m.gguf", out)
        model.write_bytes(b"x")  # the download finished
        self.wait_starts(1)
        self.assertIn(str(model), self.started()[0])

    def test_picking_a_model_while_waiting_starts_it(self):
        self.run_script("--diffusion-model", str(self.tmp / "missing.safetensors"))
        time.sleep(1)
        (self.tmp / "ok.gguf").write_bytes(b"x")
        self.active.write_text(f"--diffusion-model {self.tmp / 'ok.gguf'}\n", encoding="utf-8")
        self.wait_starts(1)
        self.assertIn("ok.gguf", self.started()[0])

    def test_restart_server_restarts_it_with_the_same_flags_whether_or_not_the_file_exists(self):
        with patch.object(sdmodels, "SD_DIR", self.tmp):
            self.run_script("--steps", "8")  # no active file: the defaults
            self.wait_starts(1)
            sdmodels.restart_server("video")  # creates the (empty) file: the defaults run again
            self.wait_starts(2)
            self.assertEqual(self.active.read_text(encoding="utf-8"), "")
            self.assertTrue(all("--steps 8" in line for line in self.started()))
            sdmodels.put(self.active, "--steps 9")  # a model picked in the panel
            self.wait_starts(3)
            sdmodels.restart_server("video")  # touches it: same flags again, within the same second or not
            self.wait_starts(4)
            self.assertEqual([line.split()[-2:] for line in self.started()[2:]], [["--steps", "9"]] * 2)


if __name__ == "__main__":
    unittest.main()
