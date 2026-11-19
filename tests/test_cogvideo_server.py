"""Tests for cogvideo/server.py (standard library only, no GPU: generate.py is replaced by a fake).

Run from the repo root:  python -m unittest discover -s tests
"""
import importlib.util
import json
import socket
import tempfile
import threading
import time
import types
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("cogvideo_server", Path(__file__).resolve().parents[1] / "cogvideo" / "server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)
GSPEC = importlib.util.spec_from_file_location("cogvideo_generate", Path(__file__).resolve().parents[1] / "cogvideo" / "generate.py")
generate = importlib.util.module_from_spec(GSPEC)
GSPEC.loader.exec_module(generate)  # imports only the standard library at the top: torch and diffusers load inside the job


class FakeGenerate:
    """Stands in for subprocess.Popen(["python", "generate.py", "<job.json>"]): wait() writes the video the job asks for, or an error.
    While it waits on the gate, terminate() ends it like SIGTERM would: return code -15 and no video."""

    def __init__(self, size=None, rc=0, error=None, gate=None, progress=None):
        self.size, self.rc, self.error, self.gate, self.progress = size, rc, error, gate, progress
        self.job = None  # the job file as generate.py would have read it
        self.terminated = threading.Event()

    def __call__(self, cmd, *args, **kwargs):
        job = self.job = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        if self.progress:  # what generate.py's report() would have written by now
            Path(job["out"] + ".progress").write_text(json.dumps(self.progress), encoding="utf-8")
        return types.SimpleNamespace(wait=self.wait, terminate=self.terminated.set)

    def wait(self):
        if self.gate:
            for _ in range(1000):  # up to 10 s
                if self.gate.is_set() or self.terminated.is_set():
                    break
                time.sleep(0.01)
        if self.terminated.is_set():
            return -15
        out = Path(self.job["out"])
        if self.rc == 0:
            with out.open("wb") as f:
                if self.size:
                    f.truncate(self.size)  # a sparse file: big, but free
                else:
                    f.write(b"webm:" + self.job["prompt"].encode())
        elif self.error:
            Path(str(out) + ".error").write_text(self.error, encoding="utf-8")
        return self.rc


class CheckTests(unittest.TestCase):
    def test_valid_request_gets_defaults(self):
        job, why = server.check({"prompt": "  a quiet forest  "})
        self.assertEqual(why, "")
        self.assertEqual((job["prompt"], job["steps"]), ("a quiet forest", server.DEFAULT_STEPS))

    def test_refuses_what_the_model_cannot_make(self):
        for req, word in (({"prompt": "x", "width": 256}, "720x480"), ({"prompt": "x", "output_format": "mp4"}, "webm"),
                          ({"prompt": "x", "sample_params": 5}, "object"), ({"prompt": "x", "sample_params": {"sample_steps": 99}}, "sample_steps"),
                          ({"prompt": "x", "sample_params": {"sample_steps": True}}, "sample_steps")):  # True is an int in Python: not "1 step"
            with self.subTest(req=req):
                job, why = server.check(req)
                self.assertIsNone(job)
                self.assertIn(word, why)


class ParameterTests(unittest.TestCase):
    def test_negative_prompt_and_guidance_reach_the_job(self):
        job, why = server.check({"prompt": "x", "negative_prompt": "  blurry ", "sample_params": {"sample_steps": 20, "guidance": {"txt_cfg": 4}}})
        self.assertEqual(why, "")
        self.assertEqual((job["negative"], job["guidance"], job["steps"]), ("blurry", 4.0, 20))

    def test_defaults_are_the_model_cards(self):
        job, _ = server.check({"prompt": "x"})
        self.assertEqual((job["negative"], job["guidance"]), ("", 6.0))

    def test_refuses_bad_negative_prompt_and_guidance(self):
        for req, word in (({"negative_prompt": 5}, "negative_prompt"), ({"negative_prompt": "x" * 2001}, "negative_prompt"),
                          ({"sample_params": {"guidance": 6}}, "object"), ({"sample_params": {"guidance": {"txt_cfg": "6"}}}, "txt_cfg"),
                          ({"sample_params": {"guidance": {"txt_cfg": True}}}, "txt_cfg"), ({"sample_params": {"guidance": {"txt_cfg": 0.5}}}, "txt_cfg"),
                          ({"sample_params": {"guidance": {"txt_cfg": 21}}}, "txt_cfg")):
            with self.subTest(req=req):
                job, why = server.check({"prompt": "x"} | req)
                self.assertIsNone(job)
                self.assertIn(word, why)


class ReportTests(unittest.TestCase):
    def test_report_replaces_the_progress_file_whole(self):
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "job.webm")
            generate.report(out, "sampling", 3, 50)
            generate.report(out, "decoding")
            self.assertEqual(json.loads(Path(out + ".progress").read_text(encoding="utf-8")), {"phase": "decoding", "step": 0, "steps": 0})
            self.assertEqual([f.name for f in Path(d).iterdir()], ["job.webm.progress"])  # no temp file left behind

    def test_report_never_fails_the_job(self):
        generate.report("/nonexistent-folder/job.webm", "text")


class ApiTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        for p in (patch.object(server.tempfile, "gettempdir", return_value=tmp.name), patch.object(server, "JOBS", {}), patch.object(server, "JOB_FILES", {}), patch.object(server, "PROCS", {})):
            p.start()
            self.addCleanup(p.stop)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def request(self, method, path, body=None):
        c = HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.getheader("Content-Type"), r.read()

    def make_video(self, prompt="a quiet forest"):
        """POST a job and wait until it is no longer running. Returns (job id, job status JSON)."""
        status, _, data = self.request("POST", "/sdcpp/v1/vid_gen", {"prompt": prompt})
        self.assertEqual(status, 202, data)
        jid = json.loads(data)["id"]
        for _ in range(100):
            job = json.loads(self.request("GET", f"/sdcpp/v1/jobs/{jid}")[2])
            if job["status"] != "running":
                return jid, job
            time.sleep(0.05)
        self.fail("the job did not finish")

    def test_finished_video_is_downloaded_from_result_url(self):
        with patch.object(server.subprocess, "Popen", FakeGenerate()):
            jid, job = self.make_video("hello")
        self.assertEqual(job["status"], "completed")
        self.assertNotIn("b64_json", job["result"])
        status, ctype, data = self.request("GET", job["result"]["result_url"])
        self.assertEqual((status, ctype, data), (200, "video/webm", b"webm:hello"))

    def test_a_running_job_shows_its_progress_and_leaves_no_file_after(self):
        gate = threading.Event()
        fake = FakeGenerate(gate=gate, progress={"phase": "sampling", "step": 7, "steps": 50})
        with patch.object(server.subprocess, "Popen", fake):
            status, _, data = self.request("POST", "/sdcpp/v1/vid_gen", {"prompt": "x"})
            jid = json.loads(data)["id"]
            for _ in range(100):  # the fake writes the file as soon as its thread runs
                job = json.loads(self.request("GET", f"/sdcpp/v1/jobs/{jid}")[2])
                if "progress" in job:
                    break
                time.sleep(0.05)
            self.assertEqual(job, {"status": "running", "progress": {"phase": "sampling", "step": 7, "steps": 50}})
            gate.set()
            for _ in range(100):
                job = json.loads(self.request("GET", f"/sdcpp/v1/jobs/{jid}")[2])
                if job["status"] != "running":
                    break
                time.sleep(0.05)
        self.assertEqual(job["status"], "completed")
        self.assertNotIn("progress", job)
        self.assertEqual(sorted(f.name for f in self.tmp.iterdir()), [f"{jid}.webm"])  # the video only: progress files are cleaned up

    def test_garbled_progress_file_is_ignored(self):
        gate = threading.Event()
        with patch.object(server.subprocess, "Popen", FakeGenerate(gate=gate)):
            _, _, data = self.request("POST", "/sdcpp/v1/vid_gen", {"prompt": "x"})
            jid = json.loads(data)["id"]
            for text in ("{not json", "[1, 2]"):
                (self.tmp / f"{jid}.webm.progress").write_text(text, encoding="utf-8")
                self.assertEqual(json.loads(self.request("GET", f"/sdcpp/v1/jobs/{jid}")[2]), {"status": "running"})
            gate.set()

    def test_the_job_gets_the_negative_prompt_and_guidance(self):
        fake = FakeGenerate()
        with patch.object(server.subprocess, "Popen", fake):
            status, _, data = self.request("POST", "/sdcpp/v1/vid_gen", {"prompt": "x", "negative_prompt": "blurry", "seed": 5,
                                                                        "sample_params": {"sample_steps": 12, "guidance": {"txt_cfg": 3.5}}})
            self.assertEqual(status, 202, data)
            for _ in range(100):
                if fake.job:
                    break
                time.sleep(0.05)
        self.assertEqual({k: fake.job[k] for k in ("prompt", "negative", "guidance", "steps", "seed")},
                         {"prompt": "x", "negative": "blurry", "guidance": 3.5, "steps": 12, "seed": 5})

    def test_no_result_for_unknown_or_running_jobs(self):
        gate = threading.Event()
        with patch.object(server.subprocess, "Popen", FakeGenerate(gate=gate)):
            status, _, data = self.request("POST", "/sdcpp/v1/vid_gen", {"prompt": "x"})
            jid = json.loads(data)["id"]
            self.assertEqual(self.request("GET", f"/sdcpp/v1/jobs/{jid}/result")[0], 404)  # still running
            self.assertEqual(self.request("POST", "/sdcpp/v1/vid_gen", {"prompt": "y"})[0], 409)
            gate.set()
        self.assertEqual(self.request("GET", "/sdcpp/v1/jobs/nope/result")[0], 404)
        self.assertEqual(self.request("GET", f"/sdcpp/v1/jobs/{jid}/other")[0], 404)

    def test_failed_job_reports_the_error_and_leaves_no_file(self):
        with patch.object(server.subprocess, "Popen", FakeGenerate(rc=1, error="RuntimeError: boom")):
            jid, job = self.make_video()
        self.assertEqual(job, {"status": "failed", "error": "RuntimeError: boom"})
        self.assertEqual(self.request("GET", f"/sdcpp/v1/jobs/{jid}/result")[0], 404)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_dropping_a_job_deletes_its_video(self):
        with patch.object(server.subprocess, "Popen", FakeGenerate()):
            ids = [self.make_video(f"v{i}")[0] for i in range(5)]
        # posting the 5th job dropped the 1st: the newest few stay
        self.assertEqual(self.request("GET", f"/sdcpp/v1/jobs/{ids[0]}/result")[0], 404)
        self.assertFalse((self.tmp / f"{ids[0]}.webm").exists())
        for jid in ids[1:]:
            self.assertEqual(self.request("GET", f"/sdcpp/v1/jobs/{jid}/result")[0], 200)
            self.assertTrue((self.tmp / f"{jid}.webm").exists())

    def post_video(self):
        status, _, data = self.request("POST", "/sdcpp/v1/vid_gen", {"prompt": "x"})
        self.assertEqual(status, 202, data)
        return json.loads(data)["id"]

    def wait_for(self, jid, status):
        for _ in range(200):
            job = json.loads(self.request("GET", f"/sdcpp/v1/jobs/{jid}")[2])
            if job["status"] == status:
                return job
            time.sleep(0.02)
        self.fail(f"the job never became {status}")

    def test_cancel_stops_a_running_job_and_frees_the_server(self):
        fake = FakeGenerate(gate=threading.Event())  # never opened: only terminate() ends it
        with patch.object(server.subprocess, "Popen", fake):
            jid = self.post_video()
            for _ in range(200):  # the process is running
                if jid in server.PROCS:
                    break
                time.sleep(0.02)
            status, _, data = self.request("POST", f"/sdcpp/v1/jobs/{jid}/cancel")
            self.assertEqual(status, 200)
            job = self.wait_for(jid, "cancelled")
            self.assertTrue(fake.terminated.is_set())
            self.assertIn("cancelled", job["error"])
            self.assertEqual(self.request("GET", f"/sdcpp/v1/jobs/{jid}/result")[0], 404)
            self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), [])  # no job file, no half video
            self.assertEqual(server.PROCS, {})
            self.post_video()  # the server took the next job: no 409

    def test_a_cancel_that_comes_before_the_process_exists_still_stops_it(self):
        fake = FakeGenerate(gate=threading.Event())
        started, release = threading.Event(), threading.Event()

        def slow_popen(cmd, *args, **kwargs):
            proc = fake(cmd)
            started.set()
            release.wait(5)
            return proc
        with patch.object(server.subprocess, "Popen", slow_popen):
            jid = self.post_video()
            self.assertTrue(started.wait(5))
            self.assertEqual(self.request("POST", f"/sdcpp/v1/jobs/{jid}/cancel")[0], 200)  # PROCS has nothing to terminate yet
            release.set()
            self.wait_for(jid, "cancelled")
            self.assertTrue(fake.terminated.is_set())

    def test_cancel_of_an_unknown_job_is_a_404(self):
        self.assertEqual(self.request("POST", "/sdcpp/v1/jobs/nope/cancel")[0], 404)
        self.assertEqual(self.request("POST", "/sdcpp/v1/jobs//cancel")[0], 404)

    def test_cancel_of_a_finished_job_changes_nothing(self):
        with patch.object(server.subprocess, "Popen", FakeGenerate()):
            jid, job = self.make_video("hello")
        self.assertEqual(self.request("POST", f"/sdcpp/v1/jobs/{jid}/cancel")[0], 200)
        self.assertEqual(json.loads(self.request("GET", f"/sdcpp/v1/jobs/{jid}")[2])["status"], "completed")
        self.assertEqual(self.request("GET", f"/sdcpp/v1/jobs/{jid}/result")[0], 200)

    def test_a_client_that_stops_reading_does_not_block_the_next_job(self):
        with patch.object(server.subprocess, "Popen", FakeGenerate(size=64 * 1024 * 1024)):
            jid, _ = self.make_video()
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            s.connect(("127.0.0.1", self.port))
            self.addCleanup(s.close)
            s.sendall(f"GET /sdcpp/v1/jobs/{jid}/result HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            time.sleep(0.5)  # the server is now stuck writing to this client
            started = time.time()
            status, _, data = self.request("POST", "/sdcpp/v1/vid_gen", {"prompt": "next"})
            self.assertEqual(status, 202, data)
            self.assertLess(time.time() - started, 3)


if __name__ == "__main__":
    unittest.main()
