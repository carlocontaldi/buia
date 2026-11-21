"""Every local file that agent/static/index.html refers to (script src, link href, img src, CSS url()) must be served by the real app.

The static folder is mounted at "/", so the file agent/static/x.js is /x.js; a "static/x.js" reference answers 404 and the browser silently skips it.
Like test_video_api.py this needs fastapi and httpx (skipped without them). From buia-core:  python -m unittest discover -s tests
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
        missing = {x: c.get(x).status_code for x in paths if c.get(x).status_code != 200}
        self.assertEqual(missing, {}, "index.html refers to files the app does not serve (static/ is mounted at /)")


    def test_every_overlay_panel_is_positioned_by_the_base_rule(self):
        html = (AGENT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        panels = re.findall(r'<div id="(\w*panel)" hidden', html)
        self.assertTrue(panels, "found no panels in index.html, the test would pass for nothing")
        rule = re.search(r"([^{}]+)\{[^}]*position: absolute; top: 66px", html)
        self.assertIsNotNone(rule, "the rule that overlays the panels is gone")
        unpositioned = [p for p in panels if "#" + p not in rule.group(1)]
        self.assertEqual(unpositioned, [], "these panels would render in the page flow instead of over it")

if __name__ == "__main__":
    unittest.main()
