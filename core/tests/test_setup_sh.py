"""Tests for ../setup.sh, the Linux script that installs and updates. It runs the real script in a copy of the repository layout, with stand-ins
for docker, git, curl, nvidia-smi and tailscale that only write down how they were called, so nothing is installed, pulled or started.
They need bash and GNU sed/find (Linux).

Run from core:  python -m unittest discover -s tests
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if not sys.platform.startswith("linux") or not shutil.which("bash"):
    raise unittest.SkipTest("setup.sh needs Linux and bash")

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parent

SHIMS = {
    "docker": r"""#!/bin/sh
echo "docker $*" >> "$FAKE_LOG"
[ "$1 $2" = "compose version" ] && exit 0
case "$1" in
  info) case "${FAKE_DOCKER:-ok}" in
          ok) echo 27.0.0; exit 0 ;;
          denied) echo "permission denied while trying to connect to the Docker daemon socket" >&2; exit 1 ;;
          *) echo "Cannot connect to the Docker daemon" >&2; exit 1 ;;
        esac ;;
  run) [ "${FAKE_GPU:-ok}" = ok ] && exit 0; exit 125 ;;
  image) echo "ghcr.io/ggml-org/llama.cpp@sha256:newest"; exit 0 ;;
esac
exit 0
""",
    "git": r"""#!/bin/sh
echo "git $*" >> "$FAKE_LOG"
case "$*" in
  *rev-parse*) [ "${FAKE_GIT_REPO:-1}" = 1 ] && exit 0; exit 128 ;;
  *clone*) for a in "$@"; do last=$a; done; mkdir -p "$last"; exit 0 ;;
esac
exit 0
""",
    "curl": '#!/bin/sh\necho \'{"ready":true,"version":"0"}\'\n',
    "nvidia-smi": r"""#!/bin/sh
case "$*" in
  *driver_version*) echo "${FAKE_DRIVER:-580.95.05}" ;;
  *compute_cap*) echo "${FAKE_CAP:-8.9}" ;;
  *) echo "GPU" ;;
esac
""",
    "tailscale": '#!/bin/sh\necho "tailscale $*" >> "$FAKE_LOG"\n',
}


class SetupShTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = self.tmp / "repo"
        (self.repo / "core").mkdir(parents=True)
        (self.repo / "templates").mkdir()
        shutil.copy(REPO / "setup.sh", self.repo / "setup.sh")
        for name in (".env.example", "docker-compose.yml"):
            shutil.copy(CORE / name, self.repo / "core" / name)
        shutil.copy(REPO / "templates" / "BUIA.md", self.repo / "templates" / "BUIA.md")
        self.core = self.repo / "core"
        self.env = self.core / ".env"
        self.log = self.tmp / "calls.log"
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        for name, text in SHIMS.items():
            (bin_dir / name).write_text(text, encoding="utf-8")
            (bin_dir / name).chmod(0o755)
        self.bin = bin_dir

    def run_setup(self, *args, **fake):
        env = {k: v for k, v in os.environ.items() if k != "LLAMA_IMAGE"}
        env.update({"PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}", "FAKE_LOG": str(self.log)}, **fake)
        return subprocess.run(["bash", str(self.repo / "setup.sh"), *args], env=env, cwd=self.tmp, text=True, capture_output=True, timeout=60)

    def calls(self):
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def env_text(self):
        return self.env.read_text(encoding="utf-8")

    def test_first_run_installs_and_the_second_run_updates(self):
        r = self.run_setup()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("BUIA install", r.stdout)
        env = self.env_text()
        self.assertRegex(env, r"(?m)^BUIA_PASSWORD=[0-9a-f]{32}$")
        self.assertRegex(env, r"(?m)^BUIA_SESSION_SALT=[0-9a-f]{32}$")
        self.assertNotEqual(re.search(r"(?m)^BUIA_PASSWORD=(\w+)", env).group(1), re.search(r"(?m)^BUIA_SESSION_SALT=(\w+)", env).group(1))
        if os.getuid() == 0:
            self.assertNotRegex(env, r"(?m)^BUIA_UID=")
            self.assertIn("Running as root", r.stderr)
        else:
            self.assertRegex(env, rf"(?m)^BUIA_UID={os.getuid()}$")
            self.assertRegex(env, rf"(?m)^BUIA_GID={os.getgid()}$")
        self.assertTrue((self.repo / "BUIA.md").is_file(), "BUIA.md comes from the template in the folder above core")
        self.assertTrue((self.repo / ".vault").is_dir())
        self.assertTrue((self.core / "sdmodels").is_dir(), "made here so that Docker does not make it as root")
        self.assertIn("docker compose up -d --build", self.calls())
        self.assertFalse([c for c in self.calls() if "pull" in c], "a fresh install has nothing to pull")
        self.assertIn("BUIA is at http://127.0.0.1:8000", r.stdout)
        self.assertIn(re.search(r"Web UI password: (\w+)", r.stdout).group(1), env)

        before, notes = env, (self.repo / "BUIA.md").read_text(encoding="utf-8")
        (self.repo / "BUIA.md").write_text(notes + "my note\n", encoding="utf-8")
        self.log.unlink()
        r = self.run_setup()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("BUIA update", r.stdout)
        self.assertIn(f"git -C {self.repo} pull --ff-only", self.calls())
        self.assertEqual(self.env_text(), before, "an update leaves .env as it is")
        self.assertTrue((self.repo / "BUIA.md").read_text(encoding="utf-8").endswith("my note\n"))
        self.assertNotIn("Web UI password", r.stdout)

    def test_flags_choose_the_profiles_and_the_update_keeps_them(self):
        r = self.run_setup("--sd", "--cogvideo")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        clone = [c for c in self.calls() if c.startswith("git ") and " clone " in c]
        self.assertEqual(len(clone), 1)
        self.assertIn("--branch master-920-2f88688 ", clone[0])
        self.assertTrue((self.core / "sdcpp").is_dir())
        self.assertIn("docker compose --profile sd --profile cogvideo up -d --build", self.calls())
        env = self.env_text()
        self.assertRegex(env, r"(?m)^COGVIDEO=on$")
        self.assertRegex(env, r"(?m)^SD_CUDA_ARCHITECTURES=89$")  # compute_cap 8.9 from the stand-in nvidia-smi
        self.assertEqual(env.count("COGVIDEO="), 1)
        self.log.unlink()
        r = self.run_setup()   # no flags: sdcpp/ and COGVIDEO=on in .env are what keep the services
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("docker compose --profile sd --profile cogvideo up -d --build", self.calls())
        self.assertFalse([c for c in self.calls() if " clone " in c])

    def test_no_pull_and_a_copy_that_is_not_a_clone(self):
        self.run_setup()
        self.log.unlink()
        r = self.run_setup("--no-pull")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse([c for c in self.calls() if "pull" in c])
        self.log.unlink()
        r = self.run_setup(FAKE_GIT_REPO="0")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Not a git clone", r.stdout)
        self.assertFalse([c for c in self.calls() if "pull" in c])

    def test_paths_from_env_are_used_and_nothing_existing_is_overwritten(self):
        mine = self.tmp / "my notes"
        vault = self.tmp / "my vault"
        mine.mkdir()
        (mine / "BUIA.md").write_text("mine\n", encoding="utf-8")
        shutil.copy(CORE / ".env.example", self.env)
        with self.env.open("a", encoding="utf-8") as f:
            f.write(f"BUIA_PATH={mine}\nVAULT_PATH={vault}\nBUIA_PASSWORD=\n")
        r = self.run_setup()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((mine / "BUIA.md").read_text(encoding="utf-8"), "mine\n")
        self.assertTrue(vault.is_dir())
        self.assertFalse((self.repo / "BUIA.md").exists())
        self.assertIn("BUIA_PASSWORD in .env is empty", r.stderr)
        self.assertEqual(self.env_text().count("BUIA_PASSWORD="), 1, "an empty password is reported, not replaced")

    def test_an_update_gives_an_old_install_a_session_salt_and_keeps_it_after(self):
        shutil.copy(CORE / ".env.example", self.env)   # an .env from before the salt existed: a password, no salt line
        with self.env.open("a", encoding="utf-8") as f:
            f.write("BUIA_PASSWORD=mine\nBUIA_SESSION_SALT=\n")
        self.assertEqual(self.run_setup().returncode, 0)
        salt = re.findall(r"(?m)^BUIA_SESSION_SALT=(.*)$", self.env_text())
        self.assertEqual(len(salt), 1, "the empty line is filled in, not repeated")
        self.assertRegex(salt[0], r"^[0-9a-f]{32}$")
        self.assertRegex(self.env_text(), r"(?m)^BUIA_PASSWORD=mine$")
        self.assertEqual(self.run_setup().returncode, 0)
        self.assertEqual(re.findall(r"(?m)^BUIA_SESSION_SALT=(.*)$", self.env_text()), salt, "a salt that is there stays: changing it logs everyone out")

    def test_stale_uid_and_gid_are_normalized_for_the_current_user(self):
        shutil.copy(CORE / ".env.example", self.env)
        with self.env.open("a", encoding="utf-8") as f:
            f.write("BUIA_PASSWORD=mine\nBUIA_UID=1\nBUIA_GID=2\n")
        self.assertEqual(self.run_setup().returncode, 0)
        if os.getuid() == 0:
            self.assertNotRegex(self.env_text(), r"(?m)^BUIA_UID=")
            self.assertNotRegex(self.env_text(), r"(?m)^BUIA_GID=")
        else:
            self.assertRegex(self.env_text(), rf"(?m)^BUIA_UID={os.getuid()}$")
            self.assertRegex(self.env_text(), rf"(?m)^BUIA_GID={os.getgid()}$")

    def test_an_old_nvidia_driver_is_pointed_at_llama_image(self):
        r = self.run_setup(FAKE_DRIVER="550.54.14")
        self.assertIn("older than 580", r.stderr)
        self.assertIn("LLAMA_IMAGE", r.stderr)
        self.assertEqual(self.run_setup(FAKE_DRIVER="580.95.05").stderr.count("older than 580"), 0)
        with self.env.open("a", encoding="utf-8") as f:
            f.write("LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda\n")
        self.assertNotIn("older than 580", self.run_setup(FAKE_DRIVER="550.54.14").stderr)

    def test_docker_problems_stop_with_the_fix_and_a_missing_gpu_only_warns(self):
        r = self.run_setup(FAKE_DOCKER="denied")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("docker group", r.stderr)
        self.assertFalse(self.env.exists(), "nothing is created before Docker works")
        r = self.run_setup(FAKE_DOCKER="down")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("systemctl start docker", r.stderr)
        r = self.run_setup(FAKE_GPU="no")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("NVIDIA Container Toolkit", r.stderr)
        self.assertIn("docker compose up -d --build", self.calls())

    @unittest.skipIf(os.getuid() == 0, "root can write anywhere")
    def test_files_the_agent_could_not_write_are_reported(self):
        chats = self.repo / "chats"
        chats.mkdir()
        (chats / "old.json").write_text("{}", encoding="utf-8")
        chats.chmod(0o555)   # stands in for a folder an earlier run as root made
        self.addCleanup(chats.chmod, 0o755)
        r = self.run_setup()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("sudo chown -R", r.stderr)

    def test_a_snap_docker_gets_a_warning(self):
        snap = self.tmp / "snap" / "bin"
        snap.mkdir(parents=True)
        shutil.copy(self.bin / "docker", snap / "docker")
        env = {k: v for k, v in os.environ.items() if k != "LLAMA_IMAGE"}
        env.update({"PATH": f"{snap}{os.pathsep}{self.bin}{os.pathsep}{os.environ['PATH']}", "FAKE_LOG": str(self.log)})
        r = subprocess.run(["bash", str(self.repo / "setup.sh")], env=env, cwd=self.tmp, text=True, capture_output=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("comes from a snap", r.stderr)
        self.assertNotIn("comes from a snap", self.run_setup().stderr)

    def test_tailscale_and_llama_options(self):
        r = self.run_setup("--tailscale", "--llama")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("tailscale serve --bg 8000", self.calls())
        self.assertIn("docker pull ghcr.io/ggml-org/llama.cpp:server-cuda13", self.calls())   # no @sha256: the moving tag
        self.assertIn("sha256:newest", r.stdout)

    def test_unknown_option_is_refused_and_help_works(self):
        r = self.run_setup("--nope")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown option", r.stderr)
        r = self.run_setup("--help")
        self.assertEqual(r.returncode, 0)
        self.assertIn("--no-pull", r.stdout)
        self.assertFalse(self.env.exists())

    def test_values_that_must_match_other_files(self):
        script = (REPO / "setup.sh").read_text(encoding="utf-8")
        ps1 = (CORE / "setup.ps1").read_text(encoding="utf-8")
        compose = (CORE / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertEqual(re.search(r"(?m)^SD_TAG=(\S+)", script).group(1), re.search(r'\$SdTag = "([^"]+)"', ps1).group(1))
        llama = re.search(r"(?m)^LLAMA_DEFAULT=(\S+)", script).group(1)
        self.assertEqual(llama, re.search(r'\$LlamaDefault = "([^"]+)"', ps1).group(1))
        self.assertIn("${LLAMA_IMAGE:-" + llama + "@sha256:", compose)

    def test_the_script_is_stored_executable(self):
        if not shutil.which("git"):
            self.skipTest("git is not installed")
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-s", "setup.sh"], text=True, capture_output=True).stdout
        if not out:
            self.skipTest("not a git checkout")
        self.assertTrue(out.startswith("100755 "), "git must store setup.sh as executable (git update-index --chmod=+x setup.sh), or ./setup.sh fails on Linux")


if __name__ == "__main__":
    unittest.main()
