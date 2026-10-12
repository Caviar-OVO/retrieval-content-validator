"""Smoke test for the Streamlit console (app/streamlit_app.py).

Runs the app headlessly with Streamlit's own AppTest and asserts the three
interactions the console promises: it renders, the example loader fills inputs,
and "run" produces a real `ValidationResult` rendered across the report tabs.

Skipped when streamlit is not installed (it is an optional dependency).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from streamlit.testing.v1 import AppTest
    _HAS_STREAMLIT = True
except Exception:  # pragma: no cover - streamlit is optional
    _HAS_STREAMLIT = False

APP = ROOT / "app" / "streamlit_app.py"


def _find(seq, needle):
    return next((x for x in seq if needle in (getattr(x, "label", "") or "")), None)


@unittest.skipUnless(_HAS_STREAMLIT, "streamlit not installed")
@unittest.skipUnless(APP.exists(), "app/streamlit_app.py missing")
class TestStreamlitConsole(unittest.TestCase):
    @staticmethod
    def _markdown(at) -> str:
        return " ".join(m.value or "" for m in at.markdown)

    def test_onboarding_shows_on_first_run_and_can_be_dismissed(self):
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        self.assertIn("新手引导", self._markdown(at), "guide not shown on first run")
        for step in ("① 填入问题", "② 填入待审回答", "③ 提供检索来源", "④ 运行校验"):
            self.assertIn(step, self._markdown(at))

        one_click = _find(at.button, "一键体验示例")
        self.assertIsNotNone(one_click, "one-click example button missing")
        one_click.click().run()
        self.assertEqual([str(e) for e in at.exception], [], "one-click example raised")
        self.assertTrue(at.session_state.get("answer_text"), "answer not filled by the guide")
        self.assertNotIn("新手引导", self._markdown(at), "guide should hide after the one-click")

        reopen = _find(at.button, "新手引导")
        self.assertIsNotNone(reopen, "sidebar '新手引导' button missing")
        reopen.click().run()
        self.assertIn("新手引导", self._markdown(at), "guide did not reopen")

    def test_render_load_and_run(self):
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        self.assertEqual([str(e) for e in at.exception], [], "initial render raised")

        btn = _find(at.button, "载入示例")
        self.assertIsNotNone(btn, "the example loader button is missing")
        btn.click().run()
        self.assertEqual([str(e) for e in at.exception], [], "example load raised")

        run = _find(at.button, "运行校验")
        self.assertIsNotNone(run, "the run button is missing")
        run.click().run()
        self.assertEqual([str(e) for e in at.exception], [], "validate raised")

        result = at.session_state["result"]
        self.assertIsNotNone(result, "no ValidationResult was produced")
        self.assertGreater(result.factuality.total_claims, 0)
        # the reported engine is one of the documented values
        self.assertIn(result.critic_backend, ("deterministic", "llm"))
        # the report tabs actually rendered content (metrics/tables present)
        self.assertGreaterEqual(len(at.metric), 3)
        self.assertGreaterEqual(len(at.dataframe), 1)


if __name__ == "__main__":
    unittest.main()
