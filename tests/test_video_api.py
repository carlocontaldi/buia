"""Tests for the Video panel's routes in agent/agent.py (/api/video/*): parameters per model, starting a job, and the progress it reports while a
fake sd-server (progress only in its log file, as the real one) or a fake cogvideo service (progress in the job) works.

Like test_sd_job.py they need fastapi and httpx (skipped without them). From the repo root:  python -m unittest discover -s tests
"""
import atexit
import asyncio
import base64
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from datetime import datetime
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
    fail_polls = 0  # the first this many polls answer HTTP 500
    posted: list = []
    cancels: list = []  # paths of the cancel requests
    cancel_code = 200  # 200: the job is cancelled (queued sd job, cogvideo); 409: sd-server cannot interrupt a job that is generating
    dropped = False  # the old server went away (it was restarted): the job is unknown to the new one
    dropped_polls = 0

    def do_GET(self):
        if self.path == "/sdcpp/v1/capabilities":
            return self.send_json(200, {"supported_modes": ["img_gen", "vid_gen"]})
        if self.path == "/sdcpp/v1/jobs/1":
            if Fake.dropped:  # a restarted server is up again at once, but does not know the job
                Fake.dropped_polls += 1
                return self.send_json(404, {"error": "job not found"})
            i, Fake.polls = Fake.polls, Fake.polls + 1
            if i < Fake.fail_polls:
                return self.send_json(500, {"error": "boom"})
            if Fake.cancels and Fake.cancel_code == 200:
                return self.send_json(200, {"status": "cancelled", "error": "job cancelled by client"})
            if self.log_path and i < len(self.logs) and self.logs[i]:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(self.logs[i])
            return self.send_json(200, self.script[min(i, len(self.script) - 1)])
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.endswith("/cancel"):
            Fake.cancels.append(self.path)
            return self.send_json(Fake.cancel_code, {})
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
        Fake.script, Fake.logs, Fake.log_path, Fake.polls, Fake.posted, Fake.fail_polls = [DONE], [], self.logdir / "active-video.log", 0, [], 0
        Fake.cancels, Fake.cancel_code, Fake.dropped, Fake.dropped_polls = [], 200, False, 0
        agent.VIDEO_JOBS.clear()
        agent.ACTIVE_CHATS.clear()
        with agent.GPU_LOCK:
            agent.GPU_OWNER = None
        shutil.rmtree(agent.GENERATED, ignore_errors=True)
        self.sleeps = patch.object(agent.time, "sleep", self.record)
        self.sleeps.start()
        self.addCleanup(self.sleeps.stop)

    def tearDown(self):
        agent.SD_VIDEO_URL, agent.sdmodels.COG_URL, agent.models.unload_all, agent.SD_LOG_DIR = self.saved
        agent.VIDEO_JOBS.clear()
        agent.ACTIVE_CHATS.clear()
        with agent.GPU_LOCK:
            agent.GPU_OWNER = None

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

    def test_chat_turn_is_saved_before_streaming_and_same_chat_is_exclusive(self):
        async def fake_llm_stream(msgs, think=None):
            yield "delta", "first reply"
            yield "message", {"role": "assistant", "content": "first reply"}

        async def consume(iterator):
            async for _ in iterator:
                pass

        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)), \
                patch.object(agent, "llm_stream", fake_llm_stream):
            response = agent.chat_turn(agent.Ask(message="first message"))
            path = next(Path(chats).glob("*.json"))
            chat = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(chat["messages"], [{"role": "user", "content": "first message", "in_progress": {"boot": agent.BOOT}}])

            with self.assertRaises(agent.HTTPException) as e:
                agent.chat_turn(agent.Ask(message="second message", chat_id=chat["id"]))
            self.assertEqual(e.exception.status_code, 409)

            asyncio.run(consume(response.body_iterator))
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([m["role"] for m in saved["messages"]], ["user", "assistant"])
            self.assertNotIn("in_progress", saved["messages"][0])
            self.assertEqual(saved["messages"][1]["content"], "first reply")
            self.assertNotIn(chat["id"], agent.ACTIVE_CHATS)

    def test_chat_load_occurs_after_serialization_lock_is_acquired(self):
        async def fake_llm_stream(msgs, think=None):
            yield "message", {"role": "assistant", "content": "reply"}

        async def consume(iterator):
            async for _ in iterator:
                pass

        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)), \
                patch.object(agent, "llm_stream", fake_llm_stream):
            original = agent.new_chat()
            original["messages"].append({"role": "user", "content": "earlier"})
            original["messages"].append({"role": "assistant", "content": "saved reply"})
            agent.save_chat(original)
            entered = threading.Event()
            actual_load = agent.load_chat

            def observed_load(cid):
                entered.set()
                return actual_load(cid)

            result = []
            with patch.object(agent, "load_chat", observed_load):
                with agent.CHAT_LOCK:
                    worker = threading.Thread(target=lambda: result.append(agent.chat_turn(agent.Ask(message="next", chat_id=original["id"]))))
                    worker.start()
                    self.assertFalse(entered.wait(0.1), "the chat snapshot was loaded before the turn lock")
                worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertTrue(entered.is_set())
            response = result[0]
            asyncio.run(consume(response.body_iterator))
            saved = json.loads((Path(chats) / f"{original['id']}.json").read_text(encoding="utf-8"))
            self.assertEqual([m["content"] for m in saved["messages"] if m["role"] == "user"], ["earlier", "next"])

    def test_unstarted_chat_stream_releases_its_reservation_after_response_cleanup(self):
        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)):
            response = agent.chat_turn(agent.Ask(message="response never started"))
            cid = next(Path(chats).glob("*.json")).stem
            owner = agent.GPU_OWNER
            self.assertIn(cid, agent.ACTIVE_CHATS)
            asyncio.run(response.background())
            self.assertNotIn(cid, agent.ACTIVE_CHATS)
            self.assertIsNone(agent.GPU_OWNER)
            self.assertTrue(owner.startswith("chat:"))
            saved = json.loads(next(Path(chats).glob("*.json")).read_text(encoding="utf-8"))
            self.assertNotIn("in_progress", saved["messages"][-2])
            self.assertIn("interrupted", saved["messages"][-1]["content"])
            self.assertTrue(saved["messages"][-1]["notice"])

    def test_completed_chat_does_not_release_gpu_transferred_to_generation_job(self):
        cid = "20261119-120000-0001"
        owner = "chat:test-owner"
        with agent.CHAT_LOCK:
            agent.ACTIVE_CHATS.discard(cid)
        with agent.GPU_LOCK:
            agent.GPU_OWNER = owner
        agent.release_unstarted_chat(cid, owner)
        self.assertEqual(agent.GPU_OWNER, owner)
        agent.release_gpu(owner)

    def test_late_cleanup_of_an_earlier_turn_leaves_the_next_turn_of_the_same_chat_alone(self):
        async def fake_llm_stream(msgs, think=None):
            yield "delta", "reply"
            yield "message", {"role": "assistant", "content": "reply"}

        async def consume(iterator):
            async for _ in iterator:
                pass

        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)), \
                patch.object(agent, "llm_stream", fake_llm_stream):
            first = agent.chat_turn(agent.Ask(message="one"))
            cid = next(Path(chats).glob("*.json")).stem
            first_owner = agent.GPU_OWNER
            asyncio.run(consume(first.body_iterator))  # finished: its reservation is released
            second = agent.chat_turn(agent.Ask(message="two", chat_id=cid))
            second_owner = agent.GPU_OWNER
            self.assertNotEqual(first_owner, second_owner)

            agent.release_unstarted_chat(cid, first_owner)  # the first response's cleanup task, running late

            self.assertIn(cid, agent.ACTIVE_CHATS)
            self.assertEqual(agent.GPU_OWNER, second_owner)
            saved = json.loads((Path(chats) / f"{cid}.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["messages"][-1]["in_progress"]["boot"], agent.BOOT)
            asyncio.run(consume(second.body_iterator))
            saved = json.loads((Path(chats) / f"{cid}.json").read_text(encoding="utf-8"))
            self.assertEqual([m["content"] for m in saved["messages"]], ["one", "reply", "two", "reply"])

    def test_a_turn_the_model_never_answered_is_not_replayed_to_it(self):
        async def failing_llm_stream(msgs, think=None):
            raise RuntimeError("llama is down")
            yield  # makes this an async generator

        async def consume(iterator):
            async for _ in iterator:
                pass

        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)), \
                patch.object(agent, "llm_stream", failing_llm_stream):
            response = agent.chat_turn(agent.Ask(message="lost question"))
            asyncio.run(consume(response.body_iterator))
            saved = json.loads(next(Path(chats).glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual([m["role"] for m in saved["messages"]], ["user", "assistant"])  # the chat still shows what happened
        self.assertTrue(saved["messages"][-1]["notice"])
        self.assertEqual(agent.history_for_model(saved["messages"]), [])
        around = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}] + saved["messages"] + [{"role": "user", "content": "c"}]
        self.assertEqual([m["content"] for m in agent.history_for_model(around)], ["a", "b", "c"])

    def test_unfinished_saved_user_turn_is_recovered_after_restart(self):
        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)):
            agent.chat_turn(agent.Ask(message="durable prompt"))
            path = next(Path(chats).glob("*.json"))
            saved = json.loads(path.read_text(encoding="utf-8"))
            cid = saved["id"]
            self.assertEqual(saved["messages"][-1]["content"], "durable prompt")
            self.assertEqual(saved["messages"][-1]["in_progress"]["boot"], agent.BOOT)

            with agent.CHAT_LOCK:
                agent.ACTIVE_CHATS.clear()  # simulate the in-memory state being lost on process restart
            with patch.object(agent, "BOOT", agent.BOOT + "-next"):
                recovered = agent.load_chat(cid)
            self.assertEqual([m["role"] for m in recovered["messages"]], ["user", "assistant"])
            self.assertIn("restarted", recovered["messages"][-1]["content"])
            self.assertTrue(recovered["messages"][-1]["notice"])
            self.assertNotIn("in_progress", recovered["messages"][0])

    def test_chat_cannot_be_deleted_while_its_turn_is_running(self):
        async def fake_llm_stream(msgs, think=None):
            yield "delta", "reply"
            yield "message", {"role": "assistant", "content": "reply"}

        async def consume(iterator):
            async for _ in iterator:
                pass

        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)), \
                patch.object(agent, "llm_stream", fake_llm_stream):
            response = agent.chat_turn(agent.Ask(message="keep this chat"))
            path = next(Path(chats).glob("*.json"))
            cid = json.loads(path.read_text(encoding="utf-8"))["id"]
            rejected = self.client.delete(f"/api/chats/{cid}")
            self.assertEqual(rejected.status_code, 409)
            self.assertTrue(path.exists())

            asyncio.run(consume(response.body_iterator))
            deleted = self.client.delete(f"/api/chats/{cid}")
            self.assertEqual(deleted.status_code, 200)
            self.assertFalse(path.exists())

    def test_video_job_cannot_start_during_an_active_chat_turn(self):
        async def fake_llm_stream(msgs, think=None):
            yield "delta", "reply"
            yield "message", {"role": "assistant", "content": "reply"}

        async def consume(iterator):
            async for _ in iterator:
                pass

        with patch.object(agent, "llm_stream", fake_llm_stream):
            response = agent.chat_turn(agent.Ask(message="keep GPU for chat"))
            self.assertTrue(agent.GPU_OWNER.startswith("chat:"))
            with self.assertRaises(agent.HTTPException) as e:
                agent.video_start(agent.VideoAsk(values={"prompt": "a quiet forest"}))
            self.assertEqual(e.exception.status_code, 409)
            asyncio.run(consume(response.body_iterator))
            self.assertIsNone(agent.GPU_OWNER)

    def test_chat_listing_pages_and_exposes_corrupt_chats_for_download(self):
        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)):
            ids = ["20261119-120000-0001", "20261119-120001-0002", "20261119-120002-0003"]
            for i, cid in enumerate(ids):
                chat = {"id": cid, "title": f"Chat {i}", "updated": "now", "messages": [{"role": "user", "content": "hello"}]}
                (Path(chats) / f"{cid}.json").write_text(json.dumps(chat) if i != 1 else "{broken", encoding="utf-8")
                os.utime(Path(chats) / f"{cid}.json", (i + 1, i + 1))

            first = self.client.get("/api/chats?limit=2").json()
            self.assertEqual(first["next_offset"], 2)
            self.assertEqual([x["id"] for x in first["chats"]], [ids[2], ids[1]])
            corrupt = first["chats"][1]
            self.assertTrue(corrupt["corrupt"])
            self.assertEqual(self.client.get(f"/api/chats/{ids[1]}").status_code, 422)
            backup = self.client.get(f"/api/chats/{ids[1]}/raw")
            self.assertEqual((backup.status_code, backup.text), (200, "{broken"))
            second = self.client.get("/api/chats?offset=2&limit=2").json()
            self.assertIsNone(second["next_offset"])
            self.assertEqual(second["chats"][0]["id"], ids[0])

            legacy = self.client.get("/api/chats")
            self.assertIsInstance(legacy.json(), list)

            invalid_shape = {"id": ids[0], "messages": [{"role": "system", "content": "not a chat role"}]}
            (Path(chats) / f"{ids[0]}.json").write_text(json.dumps(invalid_shape), encoding="utf-8")
            self.assertEqual(self.client.get(f"/api/chats/{ids[0]}").status_code, 422)
            listing = self.client.get("/api/chats?limit=3").json()["chats"]
            self.assertTrue(next(x for x in listing if x["id"] == ids[0])["corrupt"])

    def test_media_routes_do_not_crash_on_malformed_media_fields(self):
        with tempfile.TemporaryDirectory() as chats, tempfile.TemporaryDirectory() as generated, \
                patch.object(agent, "CHATS", Path(chats)), patch.object(agent, "GENERATED", Path(generated)):
            name = "20261119-120000-a-test-0123456789abcdef.webm"
            (Path(generated) / name).write_bytes(VIDEO)
            chat = agent.new_chat()
            chat["messages"].append({"role": "assistant", "content": "broken data", "media": 1})
            agent.save_chat(chat)

            listing = self.client.get("/api/media")
            self.assertEqual(listing.status_code, 200)
            self.assertFalse(listing.json()["media"][0]["deletable"])
            rejected = self.client.delete(f"/api/media/{name}")
            self.assertEqual(rejected.status_code, 409)
            self.assertTrue((Path(generated) / name).exists())

    def test_media_gallery_prevents_deleting_files_used_by_chats(self):
        name = "20261119-120000-a-test-0123456789abcdef.webm"
        with tempfile.TemporaryDirectory() as chats, tempfile.TemporaryDirectory() as generated, \
                patch.object(agent, "CHATS", Path(chats)), patch.object(agent, "GENERATED", Path(generated)):
            (Path(generated) / name).write_bytes(VIDEO)
            chat = agent.new_chat()
            chat["messages"].append({"role": "assistant", "content": "video", "media": [{"url": f"/api/media/{name}"}]})
            agent.save_chat(chat)

            page = self.client.get("/api/media").json()
            self.assertEqual(page["media"][0]["name"], name)
            self.assertTrue(page["media"][0]["referenced"])
            self.assertFalse(page["media"][0]["deletable"])
            self.assertEqual(self.client.delete(f"/api/media/{name}").status_code, 409)

            self.client.delete(f"/api/chats/{chat['id']}")
            self.assertEqual(self.client.delete(f"/api/media/{name}").status_code, 200)
            self.assertFalse((Path(generated) / name).exists())

            pending_name = "20261119-120001-another-0123456789abcdef.webm"
            (Path(generated) / pending_name).write_bytes(VIDEO)
            pending = agent.new_chat()
            pending["messages"].extend([
                {"role": "user", "content": "make a video"},
                {"role": "assistant", "content": "", "trace": [], "pending": {"job": "job1", "boot": agent.BOOT, "tool": "generate_video"}},
            ])
            agent.save_chat(pending)
            self.assertEqual(self.client.delete(f"/api/media/{pending_name}").status_code, 409)

    def test_health_reports_not_ready_as_unhealthy(self):
        with patch.object(agent.httpx, "get", side_effect=httpx.ConnectError("offline")):
            response = self.client.get("/api/health")
        self.assertEqual((response.status_code, response.json()), (503, {"ready": False}))

    def test_health_reports_ready_as_healthy(self):
        with patch.object(agent.httpx, "get", return_value=httpx.Response(200)):
            response = self.client.get("/api/health")
        self.assertEqual((response.status_code, response.json()), (200, {"ready": True}))

    def test_chat_rejects_a_prompt_that_cannot_fit_context(self):
        oversized = "x" * (agent.PROMPT_BUDGET + 1)
        response = self.client.post("/api/chat", json={"message": oversized})
        self.assertEqual(response.status_code, 413)

    def test_file_tools_page_list_and_search_txt_and_overwrite_atomically(self):
        with tempfile.TemporaryDirectory() as data:
            data_path = Path(data)
            workspace, vault = data_path / "workspace", data_path / "vault"
            workspace.mkdir(); vault.mkdir()
            for i in range(5):
                (workspace / f"note-{i}.txt").write_text(f"needle {i}", encoding="utf-8")
            with patch.object(agent, "DATA", data_path), patch.object(agent, "ROOTS", [workspace, vault]):
                self.assertIn("offset=2", agent.list_dir("workspace", limit=2))
                results = agent.search_files("needle", "workspace", limit=2)
                self.assertIn("note-0.txt", results)
                self.assertIn("offset=2", results)
                past_the_end = agent.search_files("needle", "workspace", offset=9)
                self.assertIn("5 in total", past_the_end)
                self.assertNotIn("showing", past_the_end)
                self.assertEqual(agent.search_files("absent", "workspace"), "no matches")
                target = workspace / "note-0.txt"
                with patch.object(agent.os, "replace", side_effect=OSError("replace failed")):
                    with self.assertRaisesRegex(OSError, "replace failed"):
                        agent.write_file("workspace/note-0.txt", "replacement", overwrite=True)
                self.assertEqual(target.read_text(encoding="utf-8"), "needle 0")
                self.assertEqual(list(workspace.glob("*.tmp")), [])

    def post_chat_with_sse(self, body):
        real_async_client = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))

        def fake_async_client(**kwargs):
            return real_async_client(transport=transport, **kwargs)

        async def resolve_model():
            return "test-model"

        with tempfile.TemporaryDirectory() as chats, patch.object(agent, "CHATS", Path(chats)), \
                patch.object(agent.models, "resolve_model", resolve_model), \
                patch.object(agent.httpx, "AsyncClient", fake_async_client):
            response = self.client.post("/api/chat", json={"message": "stream test"})
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            cid = next(event["id"] for event in events if event["type"] == "chat")
            return response, events, agent.load_chat(cid)

    def test_eof_without_completion_signal_saves_partial_reply_as_interrupted(self):
        chunk = {"choices": [{"delta": {"content": "partial answer"}, "finish_reason": None}]}
        response, events, chat = self.post_chat_with_sse(f"data: {json.dumps(chunk)}\n\n".encode())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(event["type"] == "error" and "completion signal" in event["message"] for event in events))
        self.assertTrue(chat["messages"][-1]["content"].endswith("[interrupted]"))

    def test_finish_reason_without_done_marker_is_a_complete_response(self):
        chunks = [
            {"choices": [{"delta": {"content": "complete answer"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode()
        response, events, chat = self.post_chat_with_sse(body)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(any(event["type"] == "error" for event in events))
        self.assertEqual(chat["messages"][-1]["content"], "complete answer")

    # ---- sd-server: the steps are read from its log
    def test_same_prompt_generated_twice_in_one_second_never_overwrites(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 19, 12, 0, 0, tzinfo=tz)

        body = {"prompt": "a quiet forest", "width": 256, "height": 256, "video_frames": 17, "fps": 16,
                "seed": 42, "sample_params": {"sample_steps": 10}, "output_format": "webm"}
        with patch.object(agent, "datetime", FixedDateTime):
            Fake.polls = 0
            first_text, first_media = agent._sd_job("vid_gen", body)
            Fake.polls = 0
            second_text, second_media = agent._sd_job("vid_gen", body)
        self.assertIn("saved ", first_text)
        self.assertIn("saved ", second_text)
        self.assertNotEqual(first_media[0]["url"], second_media[0]["url"])
        self.assertEqual(self.client.get(first_media[0]["url"]).content, VIDEO)
        self.assertEqual(self.client.get(second_media[0]["url"]).content, VIDEO)

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

    def test_a_poll_that_fails_once_does_not_end_the_job(self):
        Fake.fail_polls = 2
        j = self.finish(self.start().json()["id"])
        self.assertEqual((j["state"], Fake.polls), ("done", 3))

    def test_a_server_that_stays_silent_is_an_error_after_a_while(self):
        Fake.fail_polls = 10**6
        with patch.object(agent, "POLL_LOST", 0.3):
            j = self.finish(self.start().json()["id"])
        self.assertEqual(j["state"], "error")
        self.assertIn("lost contact", j["error"])

    # ---- Cancel
    GENERATING = {"status": "generating"}

    def wait_polled(self):
        for _ in range(400):
            if Fake.polls:
                return
            threading.Event().wait(0.02)
        self.fail("the job never polled the server")

    def test_cancelling_a_cogvideo_job_asks_the_service_and_ends_as_cancelled(self):
        Fake.script = [{"status": "running", "progress": {"phase": "sampling", "step": 3, "steps": 50}}]
        jid = self.start(engine_cog=True).json()["id"]
        self.wait_polled()
        with patch.object(agent.sdmodels, "restart_server") as restart:
            r = self.client.post(f"/api/video/jobs/{jid}/cancel")
            self.assertEqual((r.status_code, r.json()["cancelling"]), (200, True))
            j = self.finish(jid)
        self.assertEqual((j["state"], j["cancelling"], j["error"]), ("cancelled", False, None))
        self.assertEqual(Fake.cancels, ["/sdcpp/v1/jobs/1/cancel"])
        restart.assert_not_called()  # the service stops its own process
        self.assertFalse(agent.video_busy())

    def test_cancelling_a_queued_sd_job_needs_no_restart(self):
        Fake.script = [{"status": "queued", "queue_position": 1}]
        jid = self.start().json()["id"]
        self.wait_polled()
        with patch.object(agent.sdmodels, "restart_server") as restart:
            self.client.post(f"/api/video/jobs/{jid}/cancel")
            j = self.finish(jid)
        self.assertEqual(j["state"], "cancelled")
        restart.assert_not_called()

    def test_cancelling_a_generating_sd_job_restarts_the_server_and_waits_until_it_is_gone(self):
        Fake.script, Fake.cancel_code = [self.GENERATING], 409  # sd-server: "cannot be interrupted yet"

        def restart(kind):
            Fake.dropped = True
            self.restarted = kind
        jid = self.start().json()["id"]
        self.wait_polled()
        with patch.object(agent.sdmodels, "restart_server", restart):
            self.client.post(f"/api/video/jobs/{jid}/cancel")
            j = self.finish(jid)
        self.assertEqual((j["state"], self.restarted), ("cancelled", "video"))
        self.assertGreaterEqual(Fake.dropped_polls, 1)  # the job was not reported as cancelled before the old server was gone
        self.assertFalse(agent.video_busy())

    def test_cancelling_twice_or_a_finished_job_is_harmless(self):
        jid = self.start().json()["id"]
        self.finish(jid)
        r = self.client.post(f"/api/video/jobs/{jid}/cancel")
        self.assertEqual((r.status_code, r.json()["state"], r.json()["cancelling"]), (200, "done", False))
        self.assertEqual(Fake.cancels, [])  # nothing to stop: the server was not asked
        self.assertEqual(self.client.post("/api/video/jobs/nope/cancel").status_code, 404)

    def test_a_video_that_is_already_done_is_kept_even_if_cancel_was_pressed(self):
        calls = []
        text, media = agent._sd_job("vid_gen", {"prompt": "x", "width": 256, "height": 256}, stop=lambda: calls.append(1) or len(calls) > 1)
        self.assertTrue(media, text)
        self.assertEqual(Fake.cancels, [])

    def test_a_cancel_before_the_job_was_sent_sends_nothing(self):
        text, media = agent._sd_job("vid_gen", {"prompt": "x", "width": 256, "height": 256}, stop=lambda: True)
        self.assertEqual((text, media, Fake.posted), ("error: video job cancelled", [], []))

    def test_cancel_needs_the_login(self):
        with TestClient(agent.app, base_url="http://localhost") as anon:
            self.assertEqual(anon.post("/api/video/jobs/x/cancel").status_code, 401)

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
