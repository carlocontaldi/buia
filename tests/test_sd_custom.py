"""Custom image/video models: fit estimate, family templates, repo check/add/delete.
Run: python -m unittest discover -s tests"""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import fastapi  # noqa: F401
    import httpx    # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


def load_sdmodels():
    stub = types.ModuleType("models")
    stub.MB = 1024 * 1024
    stub.HF = "https://huggingface.co"
    stub._hf_headers = lambda: {}
    async def cached(key, ttl, fn): return await fn()
    stub.cached = cached
    path = Path(__file__).resolve().parents[1] / "agent" / "sdmodels.py"
    spec = importlib.util.spec_from_file_location("sdmodels_custom_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"models": stub}):
        spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE, "needs fastapi and httpx")
class CustomModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = mock.patch.dict(os.environ, {"SD_MODELS_DIR": cls.tmp.name, "GPU_VRAM_MB": "8192"})
        cls.env.start()
        cls.sd = load_sdmodels()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.tmp.cleanup()

    def test_fit_result(self):
        self.assertIsNone(self.sd.fit_result(0, 8192, "estimate", "ref"))
        self.assertIsNone(self.sd.fit_result(1000, 0, "estimate", "ref"))
        self.assertTrue(self.sd.fit_result(1000, 8192, "estimate", "ref")["fits"])
        self.assertFalse(self.sd.fit_result(9000, 8192, "estimate", "ref")["fits"])

    def test_make_custom_wan_substitutes_model_and_flags(self):
        p = self.sd.make_custom("someone/Wan2.1-T2V-1.3B-GGUF", "wan2.1_t2v_1.3B_Q4_K_M.gguf", "wan21-t2v-1.3b")
        self.assertEqual(p["files"][0][0], "someone/Wan2.1-T2V-1.3B-GGUF")
        self.assertEqual(p["files"][0][1], "wan2.1_t2v_1.3B_Q4_K_M.gguf")
        self.assertIn("wan2.1_t2v_1.3B_Q4_K_M.gguf", p["args"])
        self.assertIn("/models/custom/", p["args"])
        self.assertNotIn("wan2.1_t2v_1.3B_fp16.safetensors", p["args"])
        self.assertIn("umt5-xxl-encoder-Q4_K_S.gguf", p["args"])
        self.assertEqual(p["family"], "wan21-t2v-1.3b")

    def test_make_custom_cogvideo_keeps_layout(self):
        p = self.sd.make_custom("mirror/CogVideoX-2b", "model_index.json", "cogvideox-2b")
        self.assertEqual(p["engine"], "cogvideo")
        self.assertTrue(all(f[0] == "mirror/CogVideoX-2b" and len(f) == 3 for f in p["files"]))

    def test_check_add_delete(self):
        tree = [{"type": "file", "path": "wan2.1_t2v_1.3B_Q4_K_M.gguf", "lfs": {"size": 900 * 1024 * 1024}},
                {"type": "file", "path": "README.md", "size": 10}]
        async def fake_tree(repo): return tree
        async def no_infos(p): raise self.sd.HTTPException(502, "offline")
        with mock.patch.object(self.sd, "repo_tree", fake_tree), mock.patch.object(self.sd, "infos", no_infos):
            chk = asyncio.run(self.sd.api_check(self.sd.RepoBody(repo="https://huggingface.co/someone/Wan2.1-T2V-1.3B-GGUF")))
            self.assertEqual(chk["repo"], "someone/Wan2.1-T2V-1.3B-GGUF")
            self.assertEqual(chk["families"][0]["family"], "wan21-t2v-1.3b")
            added = asyncio.run(self.sd.api_add(self.sd.AddBody(repo="someone/Wan2.1-T2V-1.3B-GGUF", path="wan2.1_t2v_1.3B_Q4_K_M.gguf")))
            self.assertTrue(any(x["id"] == added["id"] and x.get("custom") for x in self.sd.PRESETS))
            self.assertTrue((self.sd.SD_DIR / "custom.json").exists())
            asyncio.run(self.sd.api_delete(added["id"]))
            self.assertFalse(any(x["id"] == added["id"] for x in self.sd.PRESETS))

    def test_load_custom_rejects_traversal(self):
        bad = [{"id": "evil", "custom": True, "family": "wan21-t2v-1.3b", "kind": "video", "args": "",
                "files": [["owner/repo", "x", "../../.env"]]}]
        (self.sd.SD_DIR / "custom.json").write_text(json.dumps(bad), encoding="utf-8")
        self.sd.load_custom()
        self.assertFalse(any(p["id"] == "evil" for p in self.sd.PRESETS))


if __name__ == "__main__":
    unittest.main()
