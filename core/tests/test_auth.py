"""Tests for stable password-derived session signing keys."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import fastapi  # noqa: F401
except ImportError:
    raise unittest.SkipTest("fastapi is not installed here; run these tests in the agent container") from None

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
sys.path.insert(0, str(AGENT_DIR))
import auth  # noqa: E402


class SessionKeyTests(unittest.TestCase):
    def tearDown(self):
        auth._derive_key.cache_clear()

    def test_key_is_stable_and_uses_a_32_byte_derived_key(self):
        with patch.dict(os.environ, {"BUIA_PASSWORD": "test", "BUIA_SESSION_SALT": "install-salt"}):
            first = auth.key()
            auth._derive_key.cache_clear()
            self.assertEqual(auth.key(), first)
        self.assertEqual(len(first), 32)

    def test_key_changes_when_password_or_install_salt_changes(self):
        with patch.dict(os.environ, {"BUIA_PASSWORD": "test", "BUIA_SESSION_SALT": "install-salt"}):
            original = auth.key()
        with patch.dict(os.environ, {"BUIA_PASSWORD": "other", "BUIA_SESSION_SALT": "install-salt"}):
            password_changed = auth.key()
        with patch.dict(os.environ, {"BUIA_PASSWORD": "test", "BUIA_SESSION_SALT": "other-salt"}):
            salt_changed = auth.key()

        self.assertNotEqual(original, password_changed)
        self.assertNotEqual(original, salt_changed)

    def test_missing_salt_uses_the_compatible_fallback(self):
        with patch.dict(os.environ, {"BUIA_PASSWORD": "test"}, clear=True):
            fallback = auth.key()
        with patch.dict(os.environ, {"BUIA_PASSWORD": "test", "BUIA_SESSION_SALT": "buia-session-v1"}, clear=True):
            self.assertEqual(auth.key(), fallback)

    def test_an_empty_salt_is_a_missing_salt(self):
        # docker-compose.yml passes BUIA_SESSION_SALT=${BUIA_SESSION_SALT:-}: an .env without the line reaches the agent as ""
        with patch.dict(os.environ, {"BUIA_PASSWORD": "test"}, clear=True):
            fallback = auth.key()
        with patch.dict(os.environ, {"BUIA_PASSWORD": "test", "BUIA_SESSION_SALT": ""}, clear=True):
            self.assertEqual(auth.key(), fallback)


if __name__ == "__main__":
    unittest.main()
