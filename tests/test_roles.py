"""The two-stage role setting (Critic / Reviser) cannot drift.

The spec casts one auditor as two relayed personas sharing one ethical standard.
These tests pin that: both roles exist, the sequence is critic -> reviser, the
personas match the required wording, the shared standard has exactly the four
named criteria, and the Reviser's allowed actions stay tied to the pipeline's
revision vocabulary.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import (  # noqa: E402
    CRITIC,
    ETHICAL_IDS,
    ETHICAL_STANDARD,
    REVISER,
    ROLE_ASSIGNMENT,
    ROLE_SEQUENCE,
    ROLES,
)
from rva.roles import describe, format_text, validate_revision  # noqa: E402


class TestRoleSequence(unittest.TestCase):
    def test_two_relayed_roles_in_order(self):
        self.assertEqual(ROLE_SEQUENCE, ("critic", "reviser"))
        self.assertEqual(set(ROLES), {"critic", "reviser"})
        self.assertIs(ROLES["critic"], CRITIC)
        self.assertIs(ROLES["reviser"], REVISER)

    def test_assignment_names_both_roles(self):
        self.assertIn("批判性核验者", ROLE_ASSIGNMENT)
        self.assertIn("审慎修订者", ROLE_ASSIGNMENT)
        self.assertIn("伦理判准", ROLE_ASSIGNMENT)


class TestPersonas(unittest.TestCase):
    def test_critic_is_investigative_journalist(self):
        self.assertEqual(CRITIC.name, "Critic")
        self.assertEqual(CRITIC.title, "批判性核验者")
        self.assertIn("调查记者", CRITIC.persona)
        self.assertIn("拆解声明", CRITIC.mission)
        self.assertIn("给出判定", CRITIC.mission)

    def test_reviser_is_editor_with_minimal_edit(self):
        self.assertEqual(REVISER.name, "Reviser")
        self.assertEqual(REVISER.title, "审慎修订者")
        self.assertIn("编辑", REVISER.persona)
        self.assertIn("最小修订", REVISER.mission)
        self.assertIn("不引入新事实", REVISER.mission)
        self.assertIn("结构", REVISER.mission)
        self.assertIn("篇幅", REVISER.mission)

    def test_role_prompts_carry_the_shared_standard(self):
        for role in (CRITIC, REVISER):
            prompt = role.prompt
            self.assertIn(role.persona, prompt)
            for label in ("能力要可证", "路径要可审", "态度要诚实", "结果要有据"):
                self.assertIn(label, prompt, msg=f"{role.name} prompt missing {label}")


class TestEthicalStandard(unittest.TestCase):
    def test_exactly_four_named_criteria(self):
        self.assertEqual(
            ETHICAL_IDS,
            ("ability_evidenced", "path_auditable", "attitude_honest", "outcome_evidenced"),
        )
        self.assertEqual(
            [c.label for c in ETHICAL_STANDARD],
            ["能力要可证", "路径要可审", "态度要诚实", "结果要有据"],
        )

    def test_each_criterion_names_a_mechanism(self):
        for c in ETHICAL_STANDARD:
            self.assertTrue(c.mechanism, msg=f"{c.id} has no mechanism")
            self.assertTrue(c.principle, msg=f"{c.id} has no principle")

    def test_standard_is_shared_not_per_role(self):
        # Both role prompts are generated from the same ETHICAL_STANDARD.
        for role in (CRITIC, REVISER):
            for c in ETHICAL_STANDARD:
                self.assertIn(c.label, role.prompt)


class TestReviserContract(unittest.TestCase):
    def test_only_known_revision_actions_allowed(self):
        self.assertTrue(validate_revision(None))
        self.assertTrue(validate_revision("keep"))
        for action in ("annotate", "qualify", "drop"):
            self.assertTrue(validate_revision(action))
        self.assertFalse(validate_revision("rewrite_from_scratch"))
        self.assertFalse(validate_revision("invent"))


class TestRendering(unittest.TestCase):
    def test_describe_is_serializable_and_complete(self):
        d = describe()
        self.assertEqual(d["sequence"], ["critic", "reviser"])
        self.assertEqual([r["id"] for r in d["roles"]], ["critic", "reviser"])
        self.assertEqual(len(d["ethics"]), 4)

    def test_format_text_lists_both_roles(self):
        text = format_text()
        self.assertIn("[Critic]", text)
        self.assertIn("[Reviser]", text)
        self.assertIn("调查记者", text)
        self.assertIn("编辑", text)
        self.assertIn("共享伦理判准", text)


if __name__ == "__main__":
    unittest.main()
