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
        self.assertEqual(agent.MEMORY_FILES, ["buia/BUIA.md", "buia/memory/*.md", "buia/memory/projects/*.md"])

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
        self.assertIn("buia/memory/projects/*.md", compose)
        self.assertIn("memory/projects/", template)
        for text in (env_example, compose, template):  # one spelling everywhere: the loader and the write guard ignore case
            self.assertNotIn("Projects", text)

    def prompt_for(self, files, patterns):
        """system_prompt() over a temporary DATA with the given {path: text} files, loading `patterns`."""
        with tempfile.TemporaryDirectory() as data:
            root = Path(data).resolve()  # the Windows runner's temp folder has an 8.3 short name; the agent resolves DATA the same way
            for name, text in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text, encoding="utf-8")
            roots = [(root / "buia").resolve(), (root / "vault").resolve()]
            with unittest.mock.patch.object(agent, "DATA", root), \
                    unittest.mock.patch.object(agent, "ROOTS", roots), \
                    unittest.mock.patch.object(agent, "MEMORY_FILES", patterns):
                return agent.system_prompt()

    def test_system_prompt_loads_the_project_folder_whatever_its_spelling(self):
        # one folder per case: Windows has no memory/projects and memory/Projects side by side, Linux can
        for folder, pattern in (("Projects", "buia/memory/projects/*.md"), ("projects", "buia/memory/Projects/*.md")):
            with self.subTest(folder=folder, pattern=pattern):
                prompt = self.prompt_for({f"buia/memory/{folder}/note.md": "project note"}, [pattern])
                self.assertIn("project note", prompt)

    def test_system_prompt_deduplicates_case_alias_memory_files(self):
        prompt = self.prompt_for({"buia/memory/projects/shared.md": "shared memory"},
                                 ["buia/memory/projects/*.md", "buia/memory/Projects/*.md"])
        self.assertEqual(prompt.count("shared memory"), 1)

    def test_system_prompt_keeps_to_the_folders_the_patterns_name(self):
        files = {"buia/BUIA.md": "TEXT-OF-BUIA", "buia/memory/a.md": "TEXT-OF-MEMORY", "buia/memory/deep/b.md": "TEXT-DEEP-MEMORY",
                 "vault/top.md": "TEXT-VAULT-TOP", "vault/sub/c.md": "TEXT-VAULT-DEEP"}
        prompt = self.prompt_for(files, ["buia/memory/*.md", "buia/BUIA.md", "vault/*.md"])
        self.assertLess(prompt.index("TEXT-OF-MEMORY"), prompt.index("TEXT-OF-BUIA"))  # the patterns' order is the load order
        self.assertIn("TEXT-VAULT-TOP", prompt)
        self.assertNotIn("TEXT-DEEP-MEMORY", prompt)  # * does not cross a folder, as in a glob
        self.assertNotIn("TEXT-VAULT-DEEP", prompt)

    def test_system_prompt_rejects_files_and_symlinks_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as data:
            root = Path(data).resolve()
            (root / "outside.md").write_text("outside secret", encoding="utf-8")
            (root / "buia").mkdir()
            (root / "buia" / "ok.md").write_text("inside note", encoding="utf-8")
            try:
                (root / "buia" / "link.md").symlink_to(root / "outside.md")
            except OSError:
                self.skipTest("this account cannot create symlinks")
            # the loader and read_memory_file() each refuse such a path: test the loader alone with a plain reader as well
            for reader in (agent.read_memory_file, lambda path: path.read_text(encoding="utf-8")):
                with unittest.mock.patch.object(agent, "DATA", root), \
                        unittest.mock.patch.object(agent, "ROOTS", [(root / "buia").resolve(), (root / "vault").resolve()]), \
                        unittest.mock.patch.object(agent, "MEMORY_FILES", ["../outside.md", "buia/link.md", "buia/ok.md"]), \
                        unittest.mock.patch.object(agent, "read_memory_file", reader):
                    prompt = agent.system_prompt()
                self.assertIn("inside note", prompt)  # the loader works, so the next line proves something
                self.assertNotIn("outside secret", prompt)

    def test_system_prompt_rejects_traversal_and_absolute_memory_patterns(self):
        prompt = self.prompt_for(
            {"outside.md": "OUTSIDE-SECRET", "buia/ok.md": "INSIDE-NOTE"},
            ["../outside.md", "/outside.md", "C:/outside.md", "buia/../../outside.md"],
        )
        self.assertNotIn("OUTSIDE-SECRET", prompt)
        self.assertNotIn("INSIDE-NOTE", prompt)

    def test_system_prompt_supports_fixed_and_wildcard_memory_patterns(self):
        prompt = self.prompt_for(
            {"buia/BUIA.md": "FIXED-NOTE", "buia/memory/a.md": "TOP-NOTE",
             "buia/memory/deep/b.md": "DEEP-NOTE"},
            ["buia/BUIA.md", "buia/memory/*.md"],
        )
        self.assertIn("FIXED-NOTE", prompt)
        self.assertIn("TOP-NOTE", prompt)
        self.assertNotIn("DEEP-NOTE", prompt)

    def test_system_prompt_supports_recursive_memory_patterns(self):
        prompt = self.prompt_for(
            {"buia/memory/a.md": "SHALLOW-NOTE", "buia/memory/deep/b.md": "DEEP-NOTE"},
            ["buia/memory/**/*.md"],
        )
        self.assertIn("SHALLOW-NOTE", prompt)
        self.assertIn("DEEP-NOTE", prompt)

    def test_system_prompt_keeps_case_insensitive_root_matching(self):
        prompt = self.prompt_for(
            {"buia/BUIA.md": "ROOT-CASE-NOTE"},
            ["BUIA/BUIA.md"],
        )
        self.assertIn("ROOT-CASE-NOTE", prompt)

    def test_read_memory_file_refuses_a_symlink_and_a_folder(self):
        with tempfile.TemporaryDirectory() as data:
            root = Path(data).resolve()
            (root / "buia").mkdir()
            (root / "buia" / "ok.md").write_text("inside note", encoding="utf-8")
            with unittest.mock.patch.object(agent, "ROOTS", [(root / "buia").resolve()]):
                self.assertEqual(agent.read_memory_file(root / "buia" / "ok.md"), "inside note")
                self.assertIsNone(agent.read_memory_file(root / "buia"))
                self.assertIsNone(agent.read_memory_file(root / "buia" / "missing.md"))

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW") and Path("/proc/self/fd").is_dir(), "needs Linux")
    def test_read_memory_file_refuses_a_path_that_leads_out_of_the_roots(self):
        with tempfile.TemporaryDirectory() as data:
            root = Path(data).resolve()
            (root / "buia").mkdir()
            (root / "elsewhere").mkdir()
            (root / "elsewhere" / "x.md").write_text("outside secret", encoding="utf-8")
            (root / "buia" / "folder-link").symlink_to(root / "elsewhere", target_is_directory=True)
            (root / "buia" / "file-link.md").symlink_to(root / "elsewhere" / "x.md")
            (root / "buia" / "ok.md").write_text("inside note", encoding="utf-8")
            (root / "buia" / "inner-link.md").symlink_to(root / "buia" / "ok.md")
            with unittest.mock.patch.object(agent, "ROOTS", [(root / "buia").resolve()]):
                self.assertIsNone(agent.read_memory_file(root / "buia" / "file-link.md"))  # the last part is a symlink: O_NOFOLLOW
                self.assertIsNone(agent.read_memory_file(root / "buia" / "inner-link.md"))  # even one that stays inside (the loader hands over resolved paths)
                self.assertIsNone(agent.read_memory_file(root / "buia" / "folder-link" / "x.md"))  # a folder on the way is: the fd's real path


if __name__ == "__main__":
    unittest.main()
