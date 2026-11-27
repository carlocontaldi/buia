"""Every local file that agent/static/index.html refers to (script src, link href, img src, CSS url()) must be served by the real app.

The static folder is mounted at "/", so the file agent/static/x.js is /x.js; a "static/x.js" reference answers 404 and the browser silently skips it.
The logos and the icon live in assets/ and docker-compose.yml mounts them into static/: the test app does not have those three, so their source files are checked instead.
Like test_video_api.py this needs fastapi and httpx (skipped without them). From core:  python -m unittest discover -s tests
"""
import atexit
import os
import re
import shutil
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
except ImportError:
    raise unittest.SkipTest("fastapi and httpx are not installed here; run these tests in the agent container") from None

DATA = tempfile.mkdtemp()  # agent.py reads these when it is imported (the other test files do the same: whichever imports first wins)
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
os.environ.update(DATA_DIR=DATA, BUIA_PASSWORD="test")
AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
ROOT = AGENT_DIR.parents[1]
COMPOSE = (AGENT_DIR.parent / "docker-compose.yml").read_text(encoding="utf-8")
# URL path in the page -> file in assets/, for the "../assets/X -> /app/static/Y" mounts of the compose file
MOUNTED = {"/" + target: source for source, target in re.findall(r"source: \.\./assets/([\w.-]+)\s+target: /app/static/([\w.-]+)", COMPOSE)}
if AGENT_DIR.is_dir():
    sys.path.insert(0, str(AGENT_DIR))
import agent  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PAGE = "http://localhost/"


class Refs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found = []  # what the page asks the server for, as written
        self.css = False

    def handle_starttag(self, tag, attrs):
        self.css = tag == "style"
        self.found += [v for k, v in attrs if k in ("src", "href") and v]

    def handle_endtag(self, tag):
        self.css = False

    def handle_data(self, data):
        if self.css:
            self.found += [m.strip("'\" ") for m in re.findall(r"url\(([^)]*)\)", data)]


def local(ref):
    """The path of a reference that points at this server, or None (another site, a data: URI, a #anchor)."""
    ref = ref.strip()
    u = urlsplit(urljoin(PAGE, ref))
    if ref.startswith("#") or u.scheme not in ("http", "https") or u.netloc != "localhost":
        return None
    return u.path


class StaticAssets(unittest.TestCase):
    def test_every_local_file_the_page_refers_to_is_served(self):
        c = TestClient(agent.app, base_url="http://localhost")
        page = c.get("/")
        self.assertEqual(page.status_code, 200)
        p = Refs()
        p.feed(page.text)
        paths = sorted({x for x in map(local, p.found) if x})
        self.assertTrue(paths, "found no local references in index.html, the test would pass for nothing")
        missing = {x: c.get(x).status_code for x in paths if x not in MOUNTED and c.get(x).status_code != 200}
        self.assertEqual(missing, {}, "index.html refers to files the app does not serve (static/ is mounted at /)")
        for path in paths:
            if path in MOUNTED:
                self.assertTrue((ROOT / "assets" / MOUNTED[path]).is_file(), f"{path}: its source in assets/ is missing")

    def test_a_mounted_file_is_not_also_kept_in_static(self):
        self.assertTrue(MOUNTED, "found no mounts from assets/ in docker-compose.yml, the test would pass for nothing")
        for path in MOUNTED:
            self.assertFalse((AGENT_DIR / "static" / path.lstrip("/")).exists(), f"{path} is in static/ and mounted from assets/")

    def test_the_readme_images_exist(self):
        paths = re.findall(r'<img src="([^"]+)"', (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertTrue(paths, "found no images in README.md, the test would pass for nothing")
        for rel in paths:
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_every_overlay_panel_is_positioned_by_the_base_rule(self):
        html = (AGENT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        panels = re.findall(r'<div id="(\w*panel)" hidden', html)
        self.assertTrue(panels, "found no panels in index.html, the test would pass for nothing")
        rule = re.search(r"([^{}]+)\{[^}]*position: absolute; top: 66px", html)
        self.assertIsNotNone(rule, "the rule that overlays the panels is gone")
        unpositioned = [p for p in panels if "#" + p not in rule.group(1)]
        self.assertEqual(unpositioned, [], "these panels would render in the page flow instead of over it")

    def test_header_buttons_and_the_theme_selector_are_where_the_ui_expects_them(self):
        html = (AGENT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        header = re.search(r"<header>(.*?)</header>", html, re.S)
        self.assertIsNotNone(header, "the header is gone")
        buttons = re.findall(r'<button[^>]*\bid="(\w+)"', header.group(1))
        self.assertEqual(buttons, ["chats", "models", "vopen", "new"], "header buttons: Chats, Settings, Video dashboard, New")
        for ident in ("vopen", "theme"):  # the page looks both up by id: a second one would silently win or lose
            self.assertEqual(len(re.findall(rf'\bid="{ident}"', html)), 1, ident)
        self.assertTrue(html.index('<div id="mpanel"') < html.index('id="theme"') < html.index('<div id="vpanel"'), "the theme selector belongs in the Settings panel")

if __name__ == "__main__":
    unittest.main()
