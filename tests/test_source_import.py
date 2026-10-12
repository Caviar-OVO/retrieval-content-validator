"""Tests for the source-import helper (app/import_sources.py).

The console lets users paste or upload retrieved sources in whatever shape
their retrieval tool produced; `parse_sources` normalizes the common ones into
the `Source` schema. These tests pin that normalization and the validation
boundary (invalid records are surfaced, never silently kept).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from import_sources import parse_sources  # noqa: E402
from rva.models import Source  # noqa: E402


class TestJsonShapes(unittest.TestCase):
    def test_json_list_with_aliases(self):
        r = parse_sources('[{"id":"S1","link":"https://a.org/x","snippet":"alpha","score":0.82}]')
        self.assertEqual(r.format, "json")
        self.assertEqual(len(r.records), 1)
        rec = r.records[0]
        self.assertEqual(rec["url"], "https://a.org/x")
        self.assertEqual(rec["text"], "alpha")
        self.assertEqual(rec["traffic"], 0.82)
        self.assertEqual(rec["tag"], "a.org")

    def test_json_wrapped_and_nested_metadata(self):
        r = parse_sources('{"results":[{"href":"https://b.org/y","title":"Bit",'
                          '"metadata":{"content":"gamma","domain":"b.org"}}]}')
        self.assertEqual(len(r.records), 1)
        rec = r.records[0]
        self.assertEqual(rec["url"], "https://b.org/y")
        self.assertEqual(rec["text"], "gamma")
        self.assertEqual(rec["title"], "Bit")
        self.assertEqual(rec["tag"], "b.org")

    def test_jsonl(self):
        r = parse_sources('{"url":"https://c.org/1","content":"one"}\n'
                          '{"url":"https://d.org/2","content":"two"}', "x.jsonl")
        self.assertEqual(r.format, "jsonl")
        self.assertEqual([x["id"] for x in r.records], ["S1", "S2"])

    def test_score_0_to_100_is_rescaled(self):
        r = parse_sources('[{"url":"https://k.org","text":"x","score":73}]')
        self.assertEqual(r.records[0]["traffic"], 0.73)


class TestDelimitedAndPlainText(unittest.TestCase):
    def test_csv(self):
        r = parse_sources("id,title,url,text,traffic\nS1,T,https://e.org,hello,0.5")
        self.assertEqual(r.format, "csv/tsv")
        rec = r.records[0]
        self.assertEqual(rec["title"], "T")
        self.assertEqual(rec["text"], "hello")
        self.assertEqual(rec["traffic"], 0.5)

    def test_tsv(self):
        r = parse_sources("url\ttitle\tsnippet\nhttps://f.org\tF\thello", "x.tsv")
        self.assertEqual(r.records[0]["title"], "F")
        self.assertEqual(r.records[0]["text"], "hello")

    def test_urls_one_per_line(self):
        r = parse_sources("https://g.org/a\nhttps://h.org/b")
        self.assertEqual(r.format, "url per line")
        self.assertEqual([x["url"] for x in r.records], ["https://g.org/a", "https://h.org/b"])

    def test_markdown_links(self):
        r = parse_sources("- [Reuters study](https://i.org/r)\n- [WHO](https://j.org/w)")
        self.assertEqual([x["title"] for x in r.records], ["Reuters study", "WHO"])

    def test_text_blocks_title_url_snippet(self):
        raw = ("WHO: air quality\nhttps://www.who.int/x\n室外空气污染每年约420万人死亡。\n\n"
               "某自媒体说法\nhttps://blog.example.com/y\n这被夸大了。")
        r = parse_sources(raw)
        self.assertEqual(r.format, "text blocks")
        self.assertEqual(len(r.records), 2)
        self.assertEqual(r.records[0]["title"], "WHO: air quality")
        self.assertEqual(r.records[0]["url"], "https://www.who.int/x")
        self.assertIn("420万", r.records[0]["text"])


class TestNormalizationAndValidation(unittest.TestCase):
    def test_ids_are_filled_and_unique(self):
        r = parse_sources('[{"url":"https://a.org/1"},{"url":"https://a.org/2"},'
                          '{"id":"S1","url":"https://a.org/3"}]')
        ids = [x["id"] for x in r.records]
        self.assertEqual(len(set(ids)), len(ids))

    def test_duplicate_urls_dropped_with_warning(self):
        r = parse_sources('[{"url":"https://a.org/x","text":"a"},{"url":"https://a.org/x","text":"b"}]')
        self.assertEqual(len(r.records), 1)
        self.assertTrue(any("重复" in w for w in r.warnings))

    def test_all_records_pass_the_source_model(self):
        r = parse_sources('[{"url":"https://a.org/x","snippet":"a","score":0.9},'
                          '{"url":"https://b.org/y","content":"b"}]')
        for rec in r.records:
            Source(**rec)  # raises if the importer produced a bad shape

    def test_bad_traffic_is_dropped_not_crashing(self):
        r = parse_sources('[{"url":"https://a.org/x","text":"a","traffic":"not a number"}]')
        self.assertNotIn("traffic", r.records[0])

    def test_empty_input_is_reported(self):
        r = parse_sources("   ")
        self.assertEqual(r.records, [])
        self.assertTrue(r.warnings)


if __name__ == "__main__":
    unittest.main()
