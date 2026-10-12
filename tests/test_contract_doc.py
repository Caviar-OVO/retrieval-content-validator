"""Cross-validate docs/output-contract.md against the implementation.

The contract document carries machine-readable index fences:

    ```json
    model:  <module>.<ClassName>      -> the table below lists the fields
    fields: <module>.<ClassName>      -> the inline code list is the field set
    vocabulary: verdicts|honesty_codes|revision_actions
    input: <module>.<function>        -> the table's first column is the parameter set
    ```

These tests parse the fences, import the referenced symbols, and assert the
documented field/enum/parameter sets match the code exactly. Drift in either
direction (doc updated, code not - or the reverse) fails the suite.
"""

from __future__ import annotations

import importlib
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOC = ROOT / "docs" / "output-contract.md"

# A fence body: a single "key: value" line of an index directive.
_FENCE = re.compile(r"```json\s*\n(?:model|fields|vocabulary|input):[^\n]*\n```")


def _table_names(block: str) -> list[str]:
    """First-column names from the leading markdown table in `block`."""
    rows = []
    for ln in block.split("\n"):
        if not ln.startswith("|"):
            break
        rows.append(ln)
    names: list[str] = []
    for idx, ln in enumerate(rows):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if not cells:
            continue
        head = cells[0]
        if set(head) <= set("-: "):
            continue  # separator row
        if idx == 0:
            continue  # header row (by position)
        names.append(head.replace("`", "").strip())
    return names


def _entries():
    """Yield (kind, value, names) for every index fence in the contract doc.

    * ``fields:`` fences are followed by an inline ``a`` · ``b`` list.
    * ``model:`` / ``input:`` / ``vocabulary:`` fences are followed by a table
      whose first column holds the field / parameter / code names.
    """
    text = DOC.read_text(encoding="utf-8")
    out = []
    for m in _FENCE.finditer(text):
        body = m.group(0)
        line = [ln for ln in body.split("\n") if ln.strip() and not ln.startswith("```")][0]
        kind, _, value = line.partition(":")
        kind, value = kind.strip(), value.strip()
        rest = text[m.end():].lstrip("\n")

        if kind == "fields":
            paragraph = rest.split("\n\n", 1)[0]
            names = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", paragraph)
            out.append((kind, value, names))
            continue

        first_pipe = rest.find("|")
        if first_pipe == -1:
            out.append((kind, value, None))
            continue
        out.append((kind, value, _table_names(rest[first_pipe:])))
    return out


def _resolve(dotted: str):
    module_name, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class TestContractDocSync(unittest.TestCase):
    def test_doc_exists(self):
        self.assertTrue(DOC.exists(), "docs/output-contract.md is missing")

    def test_index_fences_present(self):
        kinds = [k for k, _v, _f in _entries()]
        for required in ("vocabulary", "model", "fields", "input"):
            self.assertIn(required, kinds, msg=f"no `{required}:` index fence in the contract doc")

    def test_model_and_field_sets_match_code(self):
        checked = 0
        for kind, value, fields in _entries():
            if kind == "model":
                self.assertIsNotNone(fields, msg=f"{value} has no table")
                cls = _resolve(value)
                expected = set(getattr(cls, "model_fields").keys())
                self.assertEqual(set(fields), expected, msg=f"{value} field drift in contract doc")
                checked += 1
            elif kind == "fields":
                self.assertIsNotNone(fields, msg=f"{value} has no inline field list")
                self.assertTrue(fields, msg=f"{value} lists no fields")
                cls = _resolve(value)
                expected = set(getattr(cls, "model_fields").keys())
                self.assertEqual(set(fields), expected, msg=f"{value} field drift in contract doc")
                checked += 1
        self.assertGreater(checked, 8, msg="too few model/fields fences were checked")

    def test_vocabularies_match_code(self):
        seen = set()
        for kind, value, fields in _entries():
            if kind != "vocabulary":
                continue
            seen.add(value)
            doc_codes = set(fields or [])
            if value == "verdicts":
                from rva.models import Verdict
                self.assertEqual(doc_codes, {v.value for v in Verdict})
            elif value == "honesty_codes":
                from rva.honesty import HONESTY_CODES
                self.assertEqual(doc_codes, set(HONESTY_CODES))
            elif value == "revision_actions":
                from rva.revise import REVISION_ACTIONS
                self.assertEqual(doc_codes, set(REVISION_ACTIONS))
            elif value == "authority_grades":
                from rva.models import AuthorityGrade
                self.assertEqual(doc_codes, {g.value for g in AuthorityGrade})
            else:
                self.fail(f"unknown vocabulary: {value}")
        self.assertEqual(
            seen,
            {"verdicts", "honesty_codes", "revision_actions", "authority_grades"},
        )

    def test_input_parameters_match_signature(self):
        import inspect
        for kind, value, fields in _entries():
            if kind != "input":
                continue
            self.assertEqual(value, "rva.validate")
            func = _resolve(value)
            params = {
                p.name
                for p in inspect.signature(func).parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            }
            self.assertTrue(set(fields) <= params, msg=f"documented inputs not in signature: {set(fields) - params}")

    def test_grounding_does_not_borrow_verdict_vocabulary(self):
        # The "two-column" promise, checked against the doc's own models.
        from rva.models import ClaimGrounding, GroundingResult
        for cls in (ClaimGrounding, GroundingResult):
            self.assertNotIn("verdict", getattr(cls, "model_fields"))


class TestGroundingAndFactualitySamples(unittest.TestCase):
    """Behavioural spot-checks that the documented shape is what validate() returns."""

    def setUp(self):
        from rva.models import Source
        self.source = Source(id="A", url="https://www.who.int/fact",
                             text="室外空气污染导致全球每年约420万人过早死亡。")
        self.question = "空气污染每年导致多少人过早死亡？"

    def test_default_run_matches_documented_top_level(self):
        from rva import validate
        out = validate(question=self.question, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[self.source])
        doc_fields = next(f for k, v, f in _entries()
                          if k == "model" and v == "rva.models.ValidationResult")
        self.assertEqual(set(out.model_dump().keys()), set(doc_fields))
        self.assertIsNone(out.grounding)   # documented: null unless a judge is chosen
        self.assertIsNone(out.revision)    # documented: null unless revise=True

    def test_revised_text_marks_changed_claims(self):
        from rva import validate
        out = validate(question=self.question,
                       answer="空气污染每年导致约420万人过早死亡。我认为这最糟糕。",
                       sources=[self.source], revise=True)
        self.assertIsNotNone(out.revision)
        # documented promise: changed claims carry [C#]
        changed = [i for i in out.revision.revision_log if i.action != "keep"]
        self.assertTrue(changed)
        for item in changed:
            self.assertIn(f"[{item.claim_id}]", out.revision.revised_text)


if __name__ == "__main__":
    unittest.main()
