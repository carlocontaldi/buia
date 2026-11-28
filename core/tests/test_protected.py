"""safe() refuses the repo's own entries at the top of buia/ (.git, templates/, README.md), the stack folder and .obsidian/.

The buia folder is the repo root, so these are reachable by the file tools unless protected() covers them.
Like the other agent tests this needs fastapi and httpx (skipped without them). From core:  python -m unittest discover -s tests
"""
import atexit
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
except ImportError:
    raise unittest.SkipTest("fastapi and httpx are not installed here; run these tests in the agent container") from None

DATA = tempfile.mkdtemp()  # agent.py reads these when it is imported (the other test files do the same: whichever imports first wins)
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
os.environ.update(DATA_DIR=DATA, BUIA_PASSWORD="test")
AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if AGENT_DIR.is_dir():
    sys.path.insert(0, str(AGENT_DIR))
_memory_files = os.environ.pop("MEMORY_FILES", None)
try:
    import agent  # noqa: E402
finally:
    if _memory_files is not None:
        os.environ["MEMORY_FILES"] = _memory_files


class ProtectedTests(unittest.TestCase):
    def setUp(self):
        self.made = []
        for name in ("buia/stack/setup.ps1", "buia/stack/docker-compose.yml", "buia/notes/README.md", "buia/templates.md",
                     "vault/README.md", "vault/templates/daily.md"):
            f = agent.DATA / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x", encoding="utf-8")
            self.made.append(f)

    def tearDown(self):
        for f in self.made:
            f.unlink(missing_ok=True)

    def refused(self, rel):
        with self.assertRaises(ValueError, msg=rel) as cm:
            agent.safe(rel)
        self.assertIn("protected", str(cm.exception), rel)

    def test_repo_entries_at_the_top_of_the_buia_folder_are_refused(self):
        for rel in ("buia/.git/config", "buia/.git", "buia/templates/BUIA.md", "buia/templates", "buia/README.md"):
            self.refused(rel)

    def test_any_spelling_of_the_case_is_refused(self):
        for rel in ("buia/.GIT/config", "buia/Templates/BUIA.md", "buia/readme.MD", "buia/Readme.md"):
            self.refused(rel)

    def test_the_stack_folder_and_obsidian_stay_refused(self):
        for rel in ("buia/stack/.env", "buia/stack/setup.ps1", "buia/.obsidian/app.json", "vault/.obsidian/app.json"):
            self.refused(rel)

    def test_the_default_vault_folder_is_refused_in_buia_but_not_as_vault(self):
        for rel in ("buia/.vault", "buia/.vault/note.md", "buia/.Vault/note.md"):
            self.refused(rel)
        self.assertEqual(agent.safe("vault/note.md"), (agent.DATA / "vault/note.md").resolve())

    def test_other_files_are_not_caught(self):
        for rel in ("buia/notes/README.md", "buia/templates.md", "buia/memory/README.md", "buia/BUIA.md"):
            self.assertEqual(agent.safe(rel), (agent.DATA / rel).resolve(), rel)

    def test_the_vault_keeps_its_own_readme_and_templates(self):
        for rel in ("vault/README.md", "vault/templates/daily.md"):
            self.assertEqual(agent.safe(rel), (agent.DATA / rel).resolve(), rel)


class VersionTests(unittest.TestCase):
    def test_the_version_file_holds_one_release_number(self):
        self.assertRegex(agent.VERSION, r"^\d+\.\d+\.\d+$")
        self.assertEqual(agent.VERSION, (AGENT_DIR / "VERSION").read_text(encoding="utf-8").strip())

    def test_the_image_copies_the_version_file(self):
        self.assertIn("VERSION", (AGENT_DIR / "Dockerfile").read_text(encoding="utf-8"))


class MemoryFilesTests(unittest.TestCase):
    def test_the_defaults_name_buia(self):
        self.assertEqual(agent.MEMORY_FILES, ["buia/BUIA.md", "buia/memory/*.md", "buia/memory/projects/*.md", "buia/memory/Projects/*.md"])

    def test_legacy_memory_setting_also_loads_the_lowercase_project_folder(self):
        self.assertEqual(
            agent.memory_patterns("buia/BUIA.md,buia/memory/*.md,buia/memory/Projects/*.md"),
            ["buia/BUIA.md", "buia/memory/*.md", "buia/memory/Projects/*.md", "buia/memory/projects/*.md"],
        )

    def test_the_write_guard_covers_the_memory_files_only(self):
        (agent.DATA / "buia").mkdir(exist_ok=True)
        with unittest.mock.patch.object(agent, "MEMORY_FILES", ["buia/BUIA.md"]):
            self.assertTrue(agent.is_memory_file("buia/BUIA.md"))
            self.assertFalse(agent.is_memory_file("buia/other.md"))

    def test_shipped_project_memory_paths_use_the_lowercase_folder(self):
        repo = AGENT_DIR.parent.parent
        env_example = (repo / "core" / ".env.example").read_text(encoding="utf-8")
        compose = (repo / "core" / "docker-compose.yml").read_text(encoding="utf-8")
        template = (repo / "templates" / "BUIA.md").read_text(encoding="utf-8")
        self.assertIn("buia/memory/projects/*.md", env_example)
        self.assertIn("buia/memory/Projects/*.md", env_example)
        self.assertIn("buia/memory/projects/*.md", compose)
        self.assertIn("buia/memory/Projects/*.md", compose)
        self.assertIn("memory/projects/", template)

    def test_system_prompt_loads_new_and_legacy_project_memory(self):
        with tempfile.TemporaryDirectory() as data:
            root = Path(data)
            lower = root / "buia/memory/projects/lower.md"
            upper = root / "buia/memory/Projects/upper.md"
            lower.parent.mkdir(parents=True)
            upper.parent.mkdir(parents=True)
            lower.write_text("lowercase memory", encoding="utf-8")
            upper.write_text("legacy memory", encoding="utf-8")

            with unittest.mock.patch.object(agent, "DATA", root), \
                    unittest.mock.patch.object(agent, "ROOTS", [
                        (root / "buia").resolve(),
                        (root / "vault").resolve(),
                    ]), \
                    unittest.mock.patch.object(agent, "MEMORY_FILES", [
                        "buia/memory/projects/*.md",
                        "buia/memory/Projects/*.md",
                    ]):
                prompt = agent.system_prompt()
        self.assertIn("lowercase memory", prompt)
        self.assertIn("legacy memory", prompt)

    def test_system_prompt_deduplicates_case_alias_memory_files(self):
        with tempfile.TemporaryDirectory() as data:
            root = Path(data)
            shared = root / "buia/memory/projects/shared.md"
            shared.parent.mkdir(parents=True)
            shared.write_text("shared memory", encoding="utf-8")

            with unittest.mock.patch.object(agent, "DATA", root), \
                    unittest.mock.patch.object(agent, "ROOTS", [
                        (root / "buia").resolve(),
                        (root / "vault").resolve(),
                    ]), \
                    unittest.mock.patch.object(agent, "MEMORY_FILES", [
                        "buia/memory/projects/*.md",
                        "buia/memory/Projects/*.md",
                    ]):
                prompt = agent.system_prompt()
        self.assertEqual(prompt.count("shared memory"), 1)

    def test_system_prompt_rejects_files_and_symlinks_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as data:
            root = Path(data)
            outside = root / "outside.md"
            outside.write_text("outside secret", encoding="utf-8")
            buia = root / "buia"
            buia.mkdir()
            link = buia / "link.md"
            link.symlink_to(outside)

            with unittest.mock.patch.object(agent, "DATA", root), \
                    unittest.mock.patch.object(agent, "ROOTS", [
                        buia.resolve(),
                        (root / "vault").resolve(),
                    ]), \
                    unittest.mock.patch.object(agent, "MEMORY_FILES", [
                        "../outside.md",
                        "buia/link.md",
                    ]):
                prompt = agent.system_prompt()
        self.assertNotIn("outside secret", prompt)


if __name__ == "__main__":
    unittest.main()
