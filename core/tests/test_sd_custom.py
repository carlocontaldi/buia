"""Custom image/video models: fit estimate, family templates, repo check/add/delete.
Run: python -m unittest discover -s tests"""
import asyncio
import importlib.util
import json
import os
import re
import struct
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


def fake_header(kind, in_channels=4, middle=True):
    """A safetensors header (names, shapes, sizes only) shaped like a real checkpoint of that kind."""
    MB = 1024 * 1024
    h, off = {"__metadata__": {"format": "pt"}}, [0]
    def add(name, shape, mb, dtype="F16"):
        h[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off[0], off[0] + mb * MB]}
        off[0] += mb * MB
    if kind == "flux":
        add("model.diffusion_model.double_blocks.0.img_attn.qkv.weight", [9216, 3072], 100)
        return h
    if kind == "diffusers":
        add("conv_in.weight", [320, 4, 3, 3], 1)
        add("down_blocks.0.attentions.0.proj_in.weight", [320, 320], 1)
        return h
    add("model.diffusion_model.input_blocks.0.0.weight", [320, in_channels, 3, 3], 1)
    add("model.diffusion_model.input_blocks.1.0.weight", [320, 320, 3, 3], 3000 if kind == "sdxl" else 1500)
    if middle:
        add("model.diffusion_model.middle_block.1.proj_in.weight", [1280, 1280], 500)
    add("first_stage_model.decoder.conv_in.weight", [512, 4, 3, 3], 160 if kind == "sdxl" else 80)
    if kind == "sdxl":
        add("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight", [49408, 768], 235)
        add("conditioner.embedders.1.model.token_embedding.weight", [49408, 1280], 1500)
    elif kind == "sd15":
        add("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight", [49408, 768], 235)
    elif kind == "sd2":
        add("cond_stage_model.model.token_embedding.weight", [49408, 1024], 300)
    return h


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

    def setUp(self):
        self.sd.RUN.clear()

    def tearDown(self):
        self.sd.RUN.clear()

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

    def test_custom_path_rejects_shell_word_splitting_and_wildcards(self):
        for path in ("folder/model file.gguf", "folder/model\tfile.gguf", "folder/model*.gguf", "folder/model[1].gguf"):
            with self.subTest(path=path), self.assertRaises(self.sd.HTTPException) as e:
                self.sd.clean_path(path)
            self.assertEqual(e.exception.status_code, 400)

    def test_saved_custom_model_with_an_unsafe_path_is_kept_but_not_started(self):
        legacy = self.sd.make_custom("someone/repo", "folder/model file.gguf", "wan21-t2v-1.3b")
        with tempfile.TemporaryDirectory() as models_dir, mock.patch.object(self.sd, "SD_DIR", Path(models_dir)), \
                mock.patch.object(self.sd, "PRESETS", [*self.sd.PRESETS, legacy]):
            self.assertTrue(self.sd.custom_ok(legacy))  # saved before the rule: not dropped from custom.json by the next save
            for f in legacy["files"]:
                self.sd.dest(f).parent.mkdir(parents=True, exist_ok=True)
                self.sd.dest(f).write_text("x")
            with self.assertRaises(self.sd.HTTPException) as e:
                asyncio.run(self.sd.api_use(self.sd.IdBody(id=legacy["id"])))
            self.assertEqual(e.exception.status_code, 409)
            self.assertIn("space", e.exception.detail)
            self.assertFalse((Path(models_dir) / f"active-{legacy['kind']}.args").exists())  # sd-run.sh was not handed the broken flags

    def test_cogvideo_has_no_customs(self):
        # the cogvideo service reads one fixed folder (sdmodels/CogVideoX-2b): a second repo would share files with the preset
        base_files = self.sd.preset("cogvideox-2b")["files"]
        tree = [{"type": "file", "path": f[1], "size": 10} for f in base_files]
        async def fake_tree(repo): return tree
        with mock.patch.object(self.sd, "repo_tree", fake_tree):
            with self.assertRaises(self.sd.HTTPException) as e:
                asyncio.run(self.sd.api_check(self.sd.RepoBody(repo="mirror/CogVideoX-2b")))
            self.assertEqual(e.exception.status_code, 400)
            self.assertNotIn("CogVideoX", e.exception.detail)
            with self.assertRaises(self.sd.HTTPException):
                asyncio.run(self.sd.api_add(self.sd.AddBody(repo="mirror/CogVideoX-2b", path="model_index.json")))
        entry = {"id": "cog-x", "custom": True, "family": "cogvideox-2b", "kind": "video", "args": "", "engine": "cogvideo",
                 "files": [["mirror/CogVideoX-2b", "model_index.json", "CogVideoX-2b/model_index.json"]]}
        self.assertFalse(self.sd.custom_ok(entry))

    def test_verdicts(self):
        r = lambda need, ram=None: self.sd.fit_result(need, 8192, "estimate", "ref", ram)  # usable VRAM 7680 MB
        self.assertEqual(r(5000)["verdict"], "fits")
        self.assertEqual(r(7500)["verdict"], "tight")
        self.assertTrue(r(7500)["fits"])
        self.assertEqual(r(9000, ram=20000)["verdict"], "offload")
        self.assertFalse(r(9000, ram=20000)["fits"])
        self.assertEqual(r(9000)["verdict"], "offload")            # free RAM unknown: no claim that it cannot run
        self.assertEqual(r(30000, ram=8000)["verdict"], "no")
        self.assertIn("short by", r(30000, ram=8000)["reason"])

    def test_flux_schnell_template(self):
        for name in ("flux1-schnell-q4_0.gguf", "FLUX.1-schnell-Q4_K_S.gguf", "flux_1_schnell-Q8_0.gguf"):
            self.assertTrue(re.search(self.sd.TEMPLATES["flux1-schnell"]["pattern"], name, re.I), name)
        self.assertFalse(re.search(self.sd.TEMPLATES["flux1-schnell"]["pattern"], "flux1-dev-q4_0.gguf", re.I))
        p = self.sd.make_custom("city96/FLUX.1-schnell-gguf", "flux1-schnell-Q4_K_S.gguf", "flux1-schnell")
        self.assertIn("/models/custom/", p["args"])
        self.assertNotIn("flux1-schnell-q4_0.gguf", p["args"])
        self.assertEqual([f[1] for f in p["files"][1:]], ["clip_l.safetensors", "t5xxl_fp16.safetensors", "split_files/vae/ae.safetensors"])
        self.assertTrue(self.sd.custom_ok(p))

    def test_checkpoint_info_families(self):
        sdxl = fake_header("sdxl")
        info = self.sd.checkpoint_info(sdxl)
        self.assertEqual(info["family"], "sdxl")
        self.assertEqual(info["dtype"], "F16")
        self.assertEqual(info["mb"], round(sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in sdxl.items() if k != "__metadata__") / (1024 * 1024)))
        self.assertEqual(info["parts_mb"], {"diffusion": 3501, "te": 1735, "vae": 160, "other": 0})
        self.assertEqual(self.sd.checkpoint_info(fake_header("sd15"))["family"], "sd15")

    def test_checkpoint_info_rejects_other_files(self):
        self.assertIsNone(self.sd.checkpoint_info(fake_header("sdxl", in_channels=9)))     # inpaint
        self.assertIsNone(self.sd.checkpoint_info(fake_header("sdxl", middle=False)))      # tiny / distilled UNet
        self.assertIsNone(self.sd.checkpoint_info(fake_header("sd2")))                     # SD 2.x: not in these templates
        self.assertIsNone(self.sd.checkpoint_info(fake_header("diffusers")))               # Diffusers layout
        self.assertIsNone(self.sd.checkpoint_info(fake_header("flux")))                    # diffusion-only file
        self.assertIsNone(self.sd.checkpoint_info({"__metadata__": {}}))
        bad = fake_header("sdxl")
        bad["model.diffusion_model.input_blocks.1.0.weight"]["data_offsets"] = "oops"
        self.assertIsNone(self.sd.checkpoint_info(bad))

    def test_check_finds_checkpoints_by_tensor_names(self):
        MB = 1024 * 1024
        tree = [{"type": "file", "path": "juggernaut_final.safetensors", "lfs": {"size": 6600 * MB}},      # a name that says nothing
                {"type": "file", "path": "vae/diffusion_pytorch_model.safetensors", "lfs": {"size": 3000 * MB}},
                {"type": "file", "path": "small_lora.safetensors", "size": 100 * MB},
                {"type": "file", "path": "other_unknown.safetensors", "lfs": {"size": 7000 * MB}}]
        async def fake_tree(repo): return tree
        async def fake_ckpt(repo, path): return self.sd.checkpoint_info(fake_header("sdxl")) if path == "juggernaut_final.safetensors" else None
        with mock.patch.object(self.sd, "repo_tree", fake_tree), mock.patch.object(self.sd, "checkpoint_of", fake_ckpt):
            chk = asyncio.run(self.sd.api_check(self.sd.RepoBody(repo="someone/jugg")))
            self.assertEqual([f["family"] for f in chk["families"]], ["sdxl"])
            v = chk["families"][0]["variants"][0]
            self.assertEqual(v["path"], "juggernaut_final.safetensors")
            self.assertEqual(v["fit"]["source"], "header")
            self.assertIn(v["fit"]["verdict"], ("tight", "offload", "no"))   # 6.6 GB of weights + 2 GB on an 8 GB card
            self.assertNotIn("warning", v)
            added = asyncio.run(self.sd.api_add(self.sd.AddBody(repo="someone/jugg", path="juggernaut_final.safetensors")))
            p = self.sd.preset(added["id"])
            self.assertEqual(p["family"], "sdxl")
            self.assertTrue(p["args"].startswith("-m /models/custom/"))
            self.assertEqual(p["info"]["family"], "sdxl")
            self.assertEqual(added["fit"]["source"], "header")
            with self.assertRaises(self.sd.HTTPException) as e:   # a file the header check does not accept
                asyncio.run(self.sd.api_add(self.sd.AddBody(repo="someone/jugg", path="other_unknown.safetensors")))
            self.assertEqual(e.exception.status_code, 400)
            asyncio.run(self.sd.api_delete(added["id"]))

    def test_distilled_name_gets_a_warning(self):
        tree = [{"type": "file", "path": "sd_xl_turbo_1.0_fp16.safetensors", "lfs": {"size": 6600 * 1024 * 1024}}]
        async def fake_tree(repo): return tree
        async def fake_ckpt(repo, path): return self.sd.checkpoint_info(fake_header("sdxl"))
        with mock.patch.object(self.sd, "repo_tree", fake_tree), mock.patch.object(self.sd, "checkpoint_of", fake_ckpt):
            v = asyncio.run(self.sd.api_check(self.sd.RepoBody(repo="stabilityai/sdxl-turbo")))["families"][0]["variants"][0]
            self.assertIn("distilled", v["warning"])

    def test_header_read_over_http_with_range(self):
        header = fake_header("sd15")
        raw = json.dumps(header).encode()
        blob = struct.pack("<Q", len(raw)) + raw + b"\0" * 64
        seen = []
        def handler(request):
            seen.append(request.headers.get("range"))
            rng = request.headers["range"].split("=")[1].split("-")
            return httpx.Response(206, content=blob[int(rng[0]):int(rng[1]) + 1])
        real = httpx.AsyncClient
        factory = lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
        with mock.patch.object(self.sd.httpx, "AsyncClient", factory):
            info = asyncio.run(self.sd.read_checkpoint("someone/sd15", "v1-5-pruned-emaonly.safetensors"))
            self.assertEqual(info["family"], "sd15")
            self.assertEqual(seen, ["bytes=0-1048575"])       # one request: the header fits in the first MB
            big = {**header, "pad": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2], "x": "y" * (2 << 20)}}
            raw = json.dumps(big).encode()
            blob = struct.pack("<Q", len(raw)) + raw
            seen.clear()
            self.assertEqual(asyncio.run(self.sd.read_checkpoint("someone/sd15", "x.safetensors"))["family"], "sd15")
            self.assertEqual(len(seen), 2)                   # a header over 1 MB is fetched whole with a second request
            blob = struct.pack("<Q", 1 << 40) + b"x" * 100
            with self.assertRaises(ValueError):
                asyncio.run(self.sd.read_checkpoint("someone/bad", "x.safetensors"))
        async def boom(repo, path): raise httpx.ConnectError("offline")
        with mock.patch.object(self.sd, "read_checkpoint", boom):
            self.assertIsNone(asyncio.run(self.sd.checkpoint_of("someone/sd15", "x.safetensors")))

    def test_load_custom_validates_info(self):
        p = self.sd.make_custom("someone/jugg", "j.safetensors", "sdxl", {"family": "sdxl", "mb": 6600})
        self.assertTrue(self.sd.custom_ok(p))
        self.assertFalse(self.sd.custom_ok({**p, "info": "6600"}))
        self.assertFalse(self.sd.custom_ok({**p, "info": {"mb": True}}))

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

    def test_download_is_reserved_before_metadata_lookup(self):
        async def scenario():
            entered = asyncio.Event()
            release = asyncio.Event()
            calls = 0

            async def delayed_infos(p):
                nonlocal calls
                calls += 1
                entered.set()
                await release.wait()
                return [{"size": 1, "sha256": None} for _ in p["files"]]

            async def no_download(p, infs, run):
                return None

            with mock.patch.object(self.sd, "infos", delayed_infos), mock.patch.object(self.sd, "fetch_all", no_download):
                first = asyncio.create_task(self.sd.api_download(self.sd.IdBody(id="wan21-t2v-1.3b")))
                await entered.wait()
                try:
                    with self.assertRaises(self.sd.HTTPException) as e:
                        await asyncio.wait_for(self.sd.api_download(self.sd.IdBody(id="wan21-t2v-1.3b")), timeout=0.2)
                    self.assertEqual(e.exception.status_code, 409)
                except asyncio.TimeoutError:
                    self.fail("the second request waited for metadata instead of observing the active reservation")
                finally:
                    release.set()
                result = await first
                await self.sd.RUN["wan21-t2v-1.3b"]["task"]
                self.assertEqual(result, {"id": "wan21-t2v-1.3b"})
                self.assertEqual(calls, 1)

        with tempfile.TemporaryDirectory() as models_dir, \
                mock.patch.object(self.sd, "SD_DIR", Path(models_dir)), \
                mock.patch.object(self.sd.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10**12)):
            asyncio.run(scenario())

    def test_load_custom_rejects_traversal(self):
        bad = [{"id": "evil", "custom": True, "family": "wan21-t2v-1.3b", "kind": "video", "args": "",
                "files": [["owner/repo", "x", "../../.env"]]}]
        (self.sd.SD_DIR / "custom.json").write_text(json.dumps(bad), encoding="utf-8")
        self.sd.load_custom()
        self.assertFalse(any(p["id"] == "evil" for p in self.sd.PRESETS))


if __name__ == "__main__":
    unittest.main()
