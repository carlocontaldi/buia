"""Tests for setup.ps1, the Windows script that installs and updates. It runs the real script in a copy of the repository layout, with stand-ins
(PowerShell scripts called docker, git, nvidia-smi and tailscale, found first on PATH) that only write down how they were called, so nothing is
installed, pulled or started. It needs a web server on 127.0.0.1:8000, which the script polls: a small one is started here.

setup.cmd runs the script with Windows PowerShell (powershell), so that is what runs here on Windows; elsewhere PowerShell 7 (pwsh) must be installed.
Windows PowerShell 5.1 has its own habits (encodings, native command errors): the Windows run in CI is what tests them.

Run from core:  python -m unittest discover -s tests
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

POWERSHELL = shutil.which("powershell" if sys.platform == "win32" else "pwsh")
if not POWERSHELL:
    raise unittest.SkipTest("setup.ps1 needs PowerShell (powershell on Windows, pwsh elsewhere)")

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parent

SHIMS = {
    "docker.ps1": r"""Add-Content $env:FAKE_LOG "docker $($args -join ' ')"
if ($args[0] -eq 'info') { if ($env:FAKE_DOCKER -eq 'down') { exit 1 } else { exit 0 } }
if ($args[0] -eq 'run') { if ($env:FAKE_GPU -eq 'no') { exit 125 } else { exit 0 } }
if ($args[0] -eq 'image') { Write-Output 'ghcr.io/ggml-org/llama.cpp@sha256:newest'; exit 0 }
exit 0
""",
    "git.ps1": r"""Add-Content $env:FAKE_LOG "git $($args -join ' ')"
$all = $args -join ' '
if ($all -match 'rev-parse') { if ($env:FAKE_GIT_REPO -eq '0') { exit 128 } else { exit 0 } }
if ($all -match ' clone ') { New-Item -ItemType Directory -Force -Path $args[-1] | Out-Null; exit 0 }
exit 0
""",
    "nvidia-smi.ps1": r"""$all = $args -join ' '
if ($all -match 'driver_version') { if ($env:FAKE_DRIVER) { Write-Output $env:FAKE_DRIVER } else { Write-Output '580.95.05' } }
elseif ($all -match 'compute_cap') { if ($env:FAKE_CAP) { Write-Output $env:FAKE_CAP } else { Write-Output '8.9' } }
else { Write-Output 'GPU' }
exit 0
""",
    "tailscale.ps1": 'Add-Content $env:FAKE_LOG "tailscale $($args -join \' \')"\nexit 0\n',
}


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"ready": True, "version": "0"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class SetupPs1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server = HTTPServer(("127.0.0.1", 8000), Health)
        except OSError:
            raise unittest.SkipTest("port 8000 is in use: setup.ps1 polls it") from None
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = self.tmp / "repo"
        (self.repo / "core").mkdir(parents=True)
        (self.repo / "templates").mkdir()
        for name in ("setup.ps1", ".env.example", "docker-compose.yml"):
            shutil.copy(CORE / name, self.repo / "core" / name)
        shutil.copy(REPO / "templates" / "BUIA.md", self.repo / "templates" / "BUIA.md")
        self.core = self.repo / "core"
        self.env = self.core / ".env"
        self.log = self.tmp / "calls.log"
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name, text in SHIMS.items():
            (self.bin / name).write_text(text, encoding="utf-8")

    def run_setup(self, *args, **fake):
        env = {k: v for k, v in os.environ.items() if k != "LLAMA_IMAGE"}
        env.update({"PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}", "FAKE_LOG": str(self.log)}, **fake)
        cmd = [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.core / "setup.ps1"), *args]
        r = subprocess.run(cmd, env=env, cwd=self.tmp, text=True, capture_output=True, timeout=120)
        r.all = r.stdout + r.stderr   # warnings and errors go to either stream, depending on the PowerShell
        return r

    def calls(self):
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def env_text(self):
        return self.env.read_text(encoding="utf-8")

    def test_first_run_installs_and_the_second_run_updates(self):
        r = self.run_setup()
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("BUIA install", r.stdout)
        env = self.env_text()
        self.assertRegex(env, r"(?m)^BUIA_PASSWORD=[0-9a-f]{32}$")
        self.assertRegex(env, r"(?m)^BUIA_SESSION_SALT=[0-9a-f]{32}$")
        self.assertNotEqual(re.search(r"(?m)^BUIA_PASSWORD=(\w+)", env).group(1), re.search(r"(?m)^BUIA_SESSION_SALT=(\w+)", env).group(1))
        self.assertNotRegex(env, r"(?m)^BUIA_UID=", "the Linux user ids are not for Windows")
        self.assertTrue((self.repo / "BUIA.md").is_file(), "BUIA.md comes from the template in the folder above core")
        self.assertTrue((self.repo / ".vault").is_dir())
        self.assertIn("docker compose up -d --build", self.calls())
        self.assertFalse([c for c in self.calls() if "pull" in c], "a fresh install has nothing to pull")
        self.assertIn("BUIA is at http://127.0.0.1:8000", r.stdout)
        self.assertIn(re.search(r"Web UI password: (\w+)", r.stdout).group(1), env)

        before = env
        (self.repo / "BUIA.md").write_text("my note\n", encoding="utf-8")
        self.log.unlink()
        r = self.run_setup()
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("BUIA update", r.stdout)
        self.assertIn("git pull --ff-only", self.calls())
        self.assertEqual(self.env_text(), before, "an update leaves .env as it is")
        self.assertEqual((self.repo / "BUIA.md").read_text(encoding="utf-8"), "my note\n")
        self.assertNotIn("Web UI password", r.stdout)

    def test_flags_choose_the_profiles_and_the_update_keeps_them(self):
        r = self.run_setup("-Sd", "-CogVideo")
        self.assertEqual(r.returncode, 0, r.all)
        clone = [c for c in self.calls() if c.startswith("git ") and " clone " in c]
        self.assertEqual(len(clone), 1)
        self.assertIn("--branch master-920-2f88688 ", clone[0])
        self.assertTrue((self.core / "sdcpp").is_dir())
        self.assertTrue((self.core / "sdmodels").is_dir())
        self.assertIn("docker compose --profile sd --profile cogvideo up -d --build", self.calls())
        env = self.env_text()
        self.assertEqual(len(re.findall(r"(?m)^COGVIDEO=on$", env)), 1)
        self.assertEqual(env.count("COGVIDEO="), 1)
        self.assertRegex(env, r"(?m)^SD_CUDA_ARCHITECTURES=89$")  # compute_cap 8.9 from the stand-in nvidia-smi
        self.log.unlink()
        r = self.run_setup()   # no flags: sdcpp\ and COGVIDEO=on in .env are what keep the services
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("docker compose --profile sd --profile cogvideo up -d --build", self.calls())
        self.assertFalse([c for c in self.calls() if " clone " in c])

    def test_no_pull_and_a_copy_that_is_not_a_clone(self):
        self.run_setup()
        self.log.unlink()
        r = self.run_setup("-NoPull")
        self.assertEqual(r.returncode, 0, r.all)
        self.assertFalse([c for c in self.calls() if "pull" in c or "rev-parse" in c])
        self.log.unlink()
        r = self.run_setup(FAKE_GIT_REPO="0")
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("Not a git clone", r.stdout)
        self.assertFalse([c for c in self.calls() if "pull" in c])

    def test_paths_from_env_are_used_and_nothing_existing_is_overwritten(self):
        mine = self.tmp / "my notes"
        vault = self.tmp / "my vault"
        mine.mkdir()
        (mine / "BUIA.md").write_text("mine\n", encoding="utf-8")
        shutil.copy(CORE / ".env.example", self.env)
        with self.env.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"BUIA_PATH={mine.as_posix()}\nVAULT_PATH={vault.as_posix()}\nBUIA_PASSWORD=\n")
        r = self.run_setup()
        self.assertEqual(r.returncode, 0, r.all)
        self.assertEqual((mine / "BUIA.md").read_text(encoding="utf-8"), "mine\n")
        self.assertTrue(vault.is_dir())
        self.assertFalse((self.repo / "BUIA.md").exists())
        self.assertIn("BUIA_PASSWORD in .env is empty", r.all)
        self.assertEqual(self.env_text().count("BUIA_PASSWORD="), 1, "an empty password is reported, not replaced")

    def test_an_update_gives_an_old_install_a_session_salt_and_keeps_it_after(self):
        shutil.copy(CORE / ".env.example", self.env)   # an .env from before the salt existed: a password, an empty salt line
        with self.env.open("a", encoding="utf-8", newline="\n") as f:
            f.write("BUIA_PASSWORD=mine\nBUIA_SESSION_SALT=\n")
        self.assertEqual(self.run_setup().returncode, 0)
        salt = re.findall(r"(?m)^BUIA_SESSION_SALT=(.*)$", self.env_text())
        self.assertEqual(len(salt), 1, "the empty line is filled in, not repeated")
        self.assertRegex(salt[0], r"^[0-9a-f]{32}$")
        self.assertRegex(self.env_text(), r"(?m)^BUIA_PASSWORD=mine$")
        self.assertEqual(self.run_setup().returncode, 0)
        self.assertEqual(re.findall(r"(?m)^BUIA_SESSION_SALT=(.*)$", self.env_text()), salt, "a salt that is there stays: changing it logs everyone out")

    def test_non_ascii_text_in_env_survives_a_change_to_it(self):
        shutil.copy(CORE / ".env.example", self.env)
        text = self.env_text().replace("COGVIDEO=off", "COGVIDEO=off\n# d\u00e9j\u00e0 vu \u00e0 la maison")
        self.env.write_text(text + "\nBUIA_PASSWORD=mine\nBUIA_SESSION_SALT=abc\n", encoding="utf-8", newline="\n")
        r = self.run_setup("-CogVideo")
        self.assertEqual(r.returncode, 0, r.all)
        after = self.env.read_bytes().decode("utf-8")   # strict: .env must still be UTF-8
        self.assertIn("d\u00e9j\u00e0 vu \u00e0 la maison", after)
        self.assertIn("COGVIDEO=on", after)

    def test_an_old_nvidia_driver_is_pointed_at_llama_image(self):
        r = self.run_setup(FAKE_DRIVER="550.54")
        self.assertIn("older than 580", r.all)
        self.assertIn("LLAMA_IMAGE", r.all)
        self.assertNotIn("older than 580", self.run_setup(FAKE_DRIVER="580.95").all)
        with self.env.open("a", encoding="utf-8", newline="\n") as f:
            f.write("LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda\n")
        self.assertNotIn("older than 580", self.run_setup(FAKE_DRIVER="550.54").all)

    def test_docker_not_running_stops_before_anything_is_created_and_a_missing_gpu_only_warns(self):
        r = self.run_setup(FAKE_DOCKER="down")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("is Docker Desktop running", r.all)
        self.assertFalse(self.env.exists(), "nothing is created before Docker works")
        r = self.run_setup(FAKE_GPU="no")
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("cannot see the NVIDIA GPU", r.all)
        self.assertIn("docker compose up -d --build", self.calls())

    def test_tailscale_and_llama_options(self):
        r = self.run_setup("-Tailscale", "-Llama")
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("tailscale serve --bg 8000", self.calls())
        self.assertIn("docker pull ghcr.io/ggml-org/llama.cpp:server-cuda13", self.calls())   # no @sha256: the moving tag
        self.assertIn("sha256:newest", r.stdout)


if __name__ == "__main__":
    unittest.main()
