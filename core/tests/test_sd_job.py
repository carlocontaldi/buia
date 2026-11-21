"""Tests for how agent/agent.py's _sd_job reads a finished image or video job (a fake sd-server / cogvideo service answers).

They need the agent's own dependencies (fastapi, httpx), so they are skipped where those are missing. Run the full suite from core:
  python -m unittest discover -s tests
The _sd_job tests can also run inside the agent container, where agent.py is in the working directory. Other tests load source files relative to
the repo layout and should be run from a checkout:
  docker compose cp tests agent:/tmp/tests
  docker compose exec agent python -m unittest discover -s /tmp/tests -p test_sd_job.py
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

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
except ImportError:
    raise unittest.SkipTest("fastapi and httpx are not installed here; run these tests in the agent container") from None

DATA = tempfile.mkdtemp()  # agent.py reads these when it is imported
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
os.environ.update(DATA_DIR=DATA, BUIA_PASSWORD="test")
AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if AGENT_DIR.is_dir():
    sys.path.insert(0, str(AGENT_DIR))
import agent  # noqa: E402

BODY = {"prompt": "a quiet forest", "width": 720, "height": 480}
VIDEO = b"fake-webm-bytes"


class FakeBackend(BaseHTTPRequestHandler):
    result: dict = {}
    fetched: list = []  # paths that were requested

    def do_GET(self):
        self.fetched.append(self.path)
        if self.path == "/sdcpp/v1/capabilities":
            return self.send_json(200, {"supported_modes": ["img_gen", "vid_gen"]})
        if self.path == "/sdcpp/v1/jobs/1":
            return self.send_json(200, {"status": "completed", "result": self.result})
        if self.path == "/files/ok":
            self.send_response(200)
            self.send_header("Content-Length", str(len(VIDEO)))
            self.end_headers()
            return self.wfile.write(VIDEO)
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_json(202, {"poll_url": "/sdcpp/v1/jobs/1"})

    def send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


class SdJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackend)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        async def no_unload():
            return []
        self.saved = (agent.SD_URL, agent.SD_VIDEO_URL, agent.sdmodels.COG_URL, agent.models.unload_all)
        agent.SD_URL = agent.SD_VIDEO_URL = agent.sdmodels.COG_URL = self.url
        agent.models.unload_all = no_unload
        FakeBackend.fetched = []
        shutil.rmtree(agent.GENERATED, ignore_errors=True)

    def tearDown(self):
        agent.SD_URL, agent.SD_VIDEO_URL, agent.sdmodels.COG_URL, agent.models.unload_all = self.saved

    def run_job(self, kind, result, cog=False):
        FakeBackend.result = result
        return agent._sd_job(kind, BODY, cog=cog)

    def saved_files(self):
        return sorted(agent.GENERATED.glob("*")) if agent.GENERATED.is_dir() else []

    def test_video_embedded_as_base64_is_saved(self):  # sd-server
        out, media = self.run_job("vid_gen", {"output_format": "webm", "frame_count": 33, "fps": 16, "b64_json": base64.b64encode(VIDEO).decode()})
        (f,) = self.saved_files()
        self.assertEqual(f.read_bytes(), VIDEO)
        self.assertIn("33 frames at 16 fps", out)
        self.assertEqual([(m["kind"], m["mime"]) for m in media], [("video", "video/webm")])

    def test_video_with_result_url_is_downloaded_from_the_same_service(self):  # cogvideo
        out, media = self.run_job("vid_gen", {"output_format": "webm", "frame_count": 49, "fps": 8, "result_url": "/files/ok"}, cog=True)
        (f,) = self.saved_files()
        self.assertEqual(f.read_bytes(), VIDEO)
        self.assertIn("49 frames at 8 fps", out)
        self.assertEqual(len(media), 1)

    def test_failed_download_is_an_error_and_saves_nothing(self):
        out, media = self.run_job("vid_gen", {"output_format": "webm", "frame_count": 49, "fps": 8, "result_url": "/files/gone"}, cog=True)
        self.assertTrue(out.startswith("error: could not download the video"), out)
        self.assertEqual((media, self.saved_files()), ([], []))

    def test_result_url_that_is_not_a_path_is_never_fetched(self):
        for url in ("http://127.0.0.1:1/files/ok", "files/ok", ""):
            with self.subTest(url=url):
                out, media = self.run_job("vid_gen", {"output_format": "webm", "frame_count": 49, "fps": 8, "result_url": url}, cog=True)
                self.assertIn("unusable result address", out)
                self.assertEqual((media, self.saved_files()), ([], []))
        self.assertNotIn("/files/ok", FakeBackend.fetched)

    def test_image_is_read_from_images_list(self):
        out, media = self.run_job("img_gen", {"output_format": "png", "images": [{"b64_json": base64.b64encode(b"png-bytes").decode()}]})
        (f,) = self.saved_files()
        self.assertEqual((f.suffix, f.read_bytes()), (".png", b"png-bytes"))
        self.assertEqual([m["kind"] for m in media], ["image"])


if __name__ == "__main__":
    unittest.main()
