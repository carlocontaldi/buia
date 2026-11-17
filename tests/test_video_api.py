"""Tests for the Video panel's routes in agent/agent.py (/api/video/*): parameters per model, starting a job, and the progress it reports while a
fake sd-server (progress only in its log file, as the real one) or a fake cogvideo service (progress in the job) works.

Like test_sd_job.py they need fastapi and httpx (skipped without them). From the repo root:  python -m unittest discover -s tests
"""
import atexit
import base64
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
except ImportError:
    raise unittest.SkipTest("fastapi and httpx are not installed here; run these tests in the agent container") from None

DATA = tempfile.mkdtemp()  # agent.py reads these when it is imported (test_sd_job.py does the same: whichever imports first wins)
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
os.environ.update(DATA_DIR=DATA, BUIA_PASSWORD="test")
AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if AGENT_DIR.is_dir():
    sys.path.insert(0, str(AGENT_DIR))
import agent  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

VIDEO = b"fake-webm-bytes"
DONE = {"status": "completed", "result": {"output_format": "webm", "frame_count": 17, "fps": 16, "b64_json": base64.b64encode(VIDEO).decode()}}


def bar(step, total, unit="s/it", char="=", head=True):  # sd.cpp's progress line, see test_videogen.counter
    cur = int(step * 50 / total)
    body = "".join(" " if i > cur else ">" if head and i == cur and i != 49 else char for i in range(50))
    return f"\r  |{body}| {step}/{total} - 1.52{unit}\x1b[K" + ("\n" if step == total else "")


class Fake(BaseHTTPRequestHandler):
    """sd-server / cogvideo. Poll number i answers script[i] (the last one repeats) after appending logs[i] to the log file, if there is one."""
    script: list = []
    logs: list = []
    log_path: Path | None = None
    polls = 0
    posted: list = []

    def do_GET(self):
        if self.path == "/sdcpp/v1/capabilities":
            return self.send_json(200, {"supported_modes": ["img_gen", "vid_gen"]})
        if self.path == "/sdcpp/v1/jobs/1":
            i, Fake.polls = Fake.polls, Fake.polls + 1
            if self.log_path and i < len(self.logs) and self.logs[i]:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(self.logs[i])
            return self.send_json(200, self.script[min(i, len(self.script) - 1)])
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        Fake.posted.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
        self.send_json(202, {"poll_url": "/sdcpp/v1/jobs/1"})

    def send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


class VideoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.httpd.server_port}"
        cls.client = TestClient(agent.app, base_url="http://localhost")
        assert cls.client.post("/api/login", json={"password": "test"}).status_code == 200

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.client.close()

    def setUp(self):
        async def no_unload():
            return []
        self.logdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.logdir, ignore_errors=True)
        self.seen = []  # what the job showed after each poll: the sleep between two polls is where it is recorded, so the test never waits
        self.saved = (agent.SD_VIDEO_URL, agent.sdmodels.COG_URL, agent.models.unload_all, agent.SD_LOG_DIR)
        agent.SD_VIDEO_URL = agent.sdmodels.COG_URL = self.url
        agent.models.unload_all, agent.SD_LOG_DIR = no_unload, self.logdir
        Fake.script, Fake.logs, Fake.log_path, Fake.polls, Fake.posted = [DONE], [], self.logdir / "active-video.log", 0, []
        agent.VIDEO_JOBS.clear()
        shutil.rmtree(agent.GENERATED, ignore_errors=True)
        self.sleeps = patch.object(agent.time, "sleep", self.record)
        self.sleeps.start()
        self.addCleanup(self.sleeps.stop)

    def tearDown(self):
        agent.SD_VIDEO_URL, agent.sdmodels.COG_URL, agent.models.unload_all, agent.SD_LOG_DIR = self.saved
        agent.VIDEO_JOBS.clear()

    def record(self, seconds):
        with agent.VIDEO_LOCK:
            for job in agent.VIDEO_JOBS.values():
                self.seen.append((job["phase"], job["step"], job["steps"], job["percent"], job["queue"]))

    def start(self, engine_cog=False, model="", **values):
        with patch.object(agent.sdmodels, "cog_active", lambda: engine_cog):
            return self.client.post("/api/video/jobs", json={"model": model, "values": {"prompt": "a quiet forest"} | values})

    def finish(self, jid):
        for _ in range(400):
            j = self.client.get(f"/api/video/jobs/{jid}").json()
            if j["state"] != "running":
                return j
            threading.Event().wait(0.02)
        self.fail("the job did not finish")

    # ---- the form of a model
    def test_params_of_a_wan_model_carry_what_its_flags_set(self):
        r = self.client.get("/api/video/params", params={"model": "wan22-ti2v-5b"}).json()
        self.assertEqual(r["engine"], "sd")
        f = {x["key"]: x for x in r["fields"]}
        self.assertEqual((f["cfg_scale"]["placeholder"], f["sampler"]["placeholder"], f["flow_shift"]["placeholder"]), ("6.0", "euler", "3.0"))
        self.assertEqual(f["width"]["max"], agent.VIDEO_MAX_SIDE)

    def test_params_of_cogvideox_are_its_own(self):
        r = self.client.get("/api/video/params", params={"model": "cogvideox-2b"}).json()
        self.assertEqual((r["engine"], [x["key"] for x in r["fields"]]), ("cogvideo", ["negative_prompt", "steps", "guidance", "seed"]))
        self.assertIn("720x480", r["note"])

    def test_params_of_an_unknown_or_image_model_are_refused(self):
        self.assertEqual(self.client.get("/api/video/params", params={"model": "nope"}).status_code, 404)
        self.assertEqual(self.client.get("/api/video/params", params={"model": "z-image-turbo"}).status_code, 404)

    def test_everything_needs_the_login(self):
        with TestClient(agent.app, base_url="http://localhost") as anon:
            for method, path in (("get", "/api/video/params"), ("get", "/api/video/jobs"), ("post", "/api/video/jobs")):
                self.assertEqual(getattr(anon, method)(path).status_code, 401, path)

    # ---- sd-server: the steps are read from its log
    def test_sd_job_follows_the_log_and_ends_with_the_video(self):
        run = {"status": "running"}
        Fake.script = [{"status": "queued", "queue_position": 2}, run, run, run, run, run, run, run, DONE]
        Fake.logs = ["", "[INFO] start\n", bar(1, 3, "MB/s", "#", False), bar(1, 4) + bar(2, 4), bar(3, 4), bar(4, 4), bar(1, 2, "it/s"), bar(2, 2, "it/s")]
        # the job is recorded in the sleep after each poll, so the poll that says "completed" is not (nothing shows after it)
        r = self.start(width=320, steps=4, cfg_scale="5", sampler="euler", seed=9, negative_prompt="blurry")
        self.assertEqual(r.status_code, 200, r.text)
        j = self.finish(r.json()["id"])
        self.assertEqual(self.seen, [("queued", 0, 0, 0, 2), ("starting", 0, 0, 0, 0), ("loading", 1, 3, 13, 0), ("sampling", 2, 4, 50, 0),
                                     ("sampling", 3, 4, 65, 0), ("decoding", 0, 0, 80, 0), ("decoding", 1, 2, 89, 0), ("decoding", 2, 2, 98, 0)])
        self.assertEqual((j["state"], j["percent"], j["seed"]), ("done", 100, 9))
        self.assertTrue(j["result"]["url"].startswith("/api/media/"), j)
        self.assertEqual(self.client.get(j["result"]["url"]).content, VIDEO)
        self.assertEqual(Fake.posted[0], {"prompt": "a quiet forest", "negative_prompt": "blurry", "seed": 9, "output_format": "webm", "width": 320,
                                         "height": 256, "video_frames": 17, "fps": 16,
                                         "sample_params": {"sample_steps": 4, "guidance": {"txt_cfg": 5.0}, "sample_method": "euler"}})

    def test_lines_of_an_earlier_job_are_not_this_jobs_progress(self):
        (self.logdir / "active-video.log").write_text("old " + bar(9, 10), encoding="utf-8")
        Fake.script = [{"status": "running"}, {"status": "running"}, DONE]
        Fake.logs = ["", bar(1, 5), ""]
        self.finish(self.start().json()["id"])
        self.assertEqual(self.seen, [("starting", 0, 0, 0, 0), ("sampling", 1, 5, 32, 0)])

    def test_without_the_log_volume_the_bar_cannot_move_but_the_video_still_comes(self):
        Fake.script, Fake.log_path = [{"status": "running"}, DONE], None
        j = self.finish(self.start().json()["id"])
        self.assertEqual(self.seen, [("running", 0, 0, None, 0)])
        self.assertEqual(j["state"], "done")

    # ---- cogvideo: the steps are in the job
    def test_cogvideo_job_reports_phase_and_step(self):
        run = {"status": "running"}
        Fake.script = [run, run | {"progress": {"phase": "text", "step": 0, "steps": 0}}, run | {"progress": {"phase": "loading", "step": 0, "steps": 0}},
                       run | {"progress": {"phase": "sampling", "step": 1, "steps": 3}}, run | {"progress": {"phase": "sampling", "step": 3, "steps": 3}},
                       run | {"progress": {"phase": "decoding", "step": 0, "steps": 0}}, run | {"progress": {"phase": "saving", "step": 0, "steps": 0}},
                       run | {"progress": {"phase": "from-the-future"}}, DONE]
        r = self.start(engine_cog=True, steps=30, guidance=4, negative_prompt="blurry", seed=1)
        self.assertEqual(r.status_code, 200, r.text)
        j = self.finish(r.json()["id"])
        self.assertEqual([s[:4] for s in self.seen], [("starting", 0, 0, 0), ("text", 0, 0, 0), ("loading", 0, 0, 10), ("sampling", 1, 3, 40),
                                                      ("sampling", 3, 3, 80), ("decoding", 0, 0, 80), ("saving", 0, 0, 98), ("starting", 0, 0, 0)])
        self.assertEqual(j["state"], "done")
        self.assertEqual(Fake.posted[0], {"prompt": "a quiet forest", "negative_prompt": "blurry", "seed": 1, "output_format": "webm", "width": 720,
                                         "height": 480, "video_frames": 49, "sample_params": {"sample_steps": 30, "guidance": {"txt_cfg": 4.0}}})

    # ---- when it goes wrong
    def test_a_failed_job_is_an_error_with_the_servers_message(self):
        Fake.script = [{"status": "running"}, {"status": "failed", "error": "out of memory"}]
        j = self.finish(self.start().json()["id"])
        self.assertEqual(j["state"], "error")
        self.assertIn("out of memory", j["error"])
        self.assertNotIn("error:", j["error"][:6])

    def test_server_that_is_not_running_is_an_error_not_a_hang(self):
        with patch.object(agent, "SD_VIDEO_URL", "http://127.0.0.1:9"):
            j = self.finish(self.start().json()["id"])
        self.assertEqual(j["state"], "error")
        self.assertIn("docker compose", j["error"])

    def test_bad_values_are_a_400_and_start_nothing(self):
        r = self.start(width=100)
        self.assertEqual((r.status_code, "Width" in r.json()["detail"]), (400, True))
        self.assertEqual((self.start(prompt="  ").status_code, agent.VIDEO_JOBS, Fake.posted), (400, {}, []))

    def test_a_model_that_is_not_in_use_is_a_409(self):
        self.assertEqual(self.start(model="wan21-t2v-1.3b").status_code, 409)
        self.assertEqual(self.start(model="nope").status_code, 404)
        self.assertEqual(self.start(model="z-image-turbo").status_code, 409)

    def test_one_video_at_a_time_and_the_chat_waits_for_it(self):
        agent.VIDEO_JOBS["busy"] = {"id": "busy", "state": "running", "started": 0}
        self.assertEqual(self.start().status_code, 409)
        r = self.client.post("/api/chat", json={"message": "hello"})
        self.assertEqual(r.status_code, 409, r.text)

    def test_only_the_newest_jobs_are_kept_and_listed_newest_first(self):
        ids = []
        for _ in range(agent.VIDEO_KEEP + 2):
            Fake.polls = 0
            ids.append(self.start().json()["id"])
            self.finish(ids[-1])
        listed = [j["id"] for j in self.client.get("/api/video/jobs").json()]
        self.assertEqual(listed, ids[::-1][:agent.VIDEO_KEEP])
        self.assertEqual(self.client.get(f"/api/video/jobs/{ids[0]}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
