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
import agent  # noqa: E402


class ProtectedTests(unittest.TestCase):
    def setUp(self):
        self.made = []
        for name in ("buia/stack/install.ps1", "buia/stack/docker-compose.yml", "buia/notes/README.md", "buia/templates.md",
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
        for rel in ("buia/stack/.env", "buia/stack/install.ps1", "buia/.obsidian/app.json", "vault/.obsidian/app.json"):
            self.refused(rel)

    def test_other_files_are_not_caught(self):
        for rel in ("buia/notes/README.md", "buia/templates.md", "buia/memory/README.md", "buia/BUIA.md"):
            self.assertEqual(agent.safe(rel), (agent.DATA / rel).resolve(), rel)

    def test_the_vault_keeps_its_own_readme_and_templates(self):
        for rel in ("vault/README.md", "vault/templates/daily.md"):
            self.assertEqual(agent.safe(rel), (agent.DATA / rel).resolve(), rel)


class MemoryPatternTests(unittest.TestCase):
    def test_the_defaults_name_buia(self):
        self.assertEqual(agent.MEMORY_FILES, ["buia/BUIA.md", "buia/memory/*.md", "buia/memory/Projects/*.md"])

    def test_the_write_guard_covers_the_memory_files_only(self):
        (agent.DATA / "buia").mkdir(exist_ok=True)
        with unittest.mock.patch.object(agent, "MEMORY_FILES", ["buia/BUIA.md"]):
            self.assertTrue(agent.is_memory_file("buia/BUIA.md"))
            self.assertFalse(agent.is_memory_file("buia/other.md"))


if __name__ == "__main__":
    unittest.main()
