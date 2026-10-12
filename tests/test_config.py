"""Tests for configuration loading (.env + environment)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rva.config as config  # noqa: E402


class TestConfig(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                        "RVA_JUDGE_MODEL", "RVA_JUDGE", "RVA_DOMAIN")}
        for k in self._saved:
            os.environ.pop(k, None)
        config._loaded = False

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        config._loaded = False

    def test_env_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text(
                "OPENAI_API_KEY=sk-test-abc123456789\n"
                "RVA_JUDGE_MODEL=gpt-4o\n"
                "RVA_JUDGE=ragas\n"
                "# a comment\n"
                "RVA_DOMAIN=public-health\n",
                encoding="utf-8",
            )
            config._load_env_file(env)
            self.assertEqual(config.get_api_key(), "sk-test-abc123456789")
            self.assertEqual(config.get_judge_model(), "gpt-4o")
            self.assertEqual(config.get_default_judge(), "ragas")
            self.assertEqual(config.get_default_domain(), "public-health")

    def test_real_env_takes_precedence_over_file(self):
        os.environ["OPENAI_API_KEY"] = "from-real-env"
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
            config._load_env_file(env)
            self.assertEqual(config.get_api_key(), "from-real-env")

    def test_quotes_are_stripped(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text('OPENAI_API_KEY="sk-quoted-123456"\n', encoding="utf-8")
            config._load_env_file(env)
            self.assertEqual(config.get_api_key(), "sk-quoted-123456")

    def test_defaults_when_unset(self):
        config._loaded = True  # skip loading the real project .env
        self.assertIsNone(config.get_api_key())
        self.assertEqual(config.get_judge_model(), "gpt-4o-mini")
        self.assertEqual(config.get_default_judge(), "local")
        self.assertEqual(config.get_default_domain(), "general")

    def test_describe_redacts_key(self):
        os.environ["OPENAI_API_KEY"] = "sk-1234567890abcdef"
        config._loaded = True
        info = config.describe()
        self.assertNotIn("abcdef", info["openai_api_key"])
        self.assertIn("set", info["openai_api_key"])


if __name__ == "__main__":
    unittest.main()