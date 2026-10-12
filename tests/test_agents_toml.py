"""The Critic / Reviser subagent shells cannot drift from the code.

`.codex/agents/*.toml` are thin shells: a name, a description, and the persona
text. The subagent TOML format has no "prompt path" field, so each shell embeds
a verbatim copy of the role prompt. These tests pin that copy to
`rva.roles.{CRITIC,REVISER}.prompt` byte-for-byte, so editing a persona forces
the shell and this test to be updated together.
"""

from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import CRITIC, REVISER  # noqa: E402

AGENTS_DIR = ROOT / ".codex" / "agents"

CRITIC_DESCRIPTION = (
    "对模型基于网络检索生成的回答执行证据效力、可审计性、诚实性三重校验，"
    "输出 claim 级核查报告与总体评估。"
)
REVISER_DESCRIPTION = (
    "同一刊物的专业编辑：依据 Critic 核验报告对回答做最小修订，"
    "只按裁定 annotate / qualify / drop，不引入新事实，保持结构、风格与篇幅，"
    "每个动作记入 revision_log。"
)


def load(shell: str) -> dict:
    path = AGENTS_DIR / f"{shell}.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


class TestShellsExist(unittest.TestCase):
    def test_both_shells_are_present_and_parse(self):
        for shell in ("critic", "reviser"):
            path = AGENTS_DIR / f"{shell}.toml"
            self.assertTrue(path.is_file(), msg=f"missing {path}")
            self.assertIsInstance(load(shell), dict)


class TestShellContract(unittest.TestCase):
    def test_required_keys_and_name(self):
        for shell in ("critic", "reviser"):
            data = load(shell)
            for key in ("name", "description", "developer_instructions"):
                self.assertIn(key, data, msg=f"{shell}.toml missing {key}")
                self.assertTrue(data[key], msg=f"{shell}.toml has empty {key}")
            self.assertEqual(data["name"], shell)
            self.assertTrue(data["developer_instructions"].strip())

    def test_descriptions_state_the_role(self):
        self.assertEqual(load("critic")["description"], CRITIC_DESCRIPTION)
        self.assertEqual(load("reviser")["description"], REVISER_DESCRIPTION)

    def test_instructions_are_verbatim_role_prompts(self):
        self.assertEqual(load("critic")["developer_instructions"], CRITIC.prompt)
        self.assertEqual(load("reviser")["developer_instructions"], REVISER.prompt)

    def test_critic_shell_keeps_a_bounded_sandbox(self):
        data = load("critic")
        self.assertEqual(data.get("model"), "gpt-5.4")
        self.assertEqual(data.get("model_reasoning_effort"), "high")
        # The Critic only reads sources and writes a report: it never needs
        # full host access, so the narrowest useful sandbox is pinned here.
        self.assertEqual(data.get("sandbox_mode"), "workspace-write")
        raw = (AGENTS_DIR / "critic.toml").read_text(encoding="utf-8")
        self.assertNotIn("danger-full-access", raw)
        # No live search server is declared: the evidence chain is offline.
        # (The commented [mcp_servers.search-evidence] placeholder is a key
        # only once uncommented, so it must not appear in the parsed TOML.)
        self.assertNotIn("mcp_servers", data)

    def test_reviser_shell_inherits_parent_session(self):
        data = load("reviser")
        self.assertNotIn("sandbox_mode", data)
        self.assertNotIn("model", data)


if __name__ == "__main__":
    unittest.main()
