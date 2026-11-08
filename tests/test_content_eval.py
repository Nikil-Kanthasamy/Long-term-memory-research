import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from src.ltm.content_eval import load_content_cases, run_content_evaluation


class ContentEvaluationTests(unittest.TestCase):
    def test_load_cases_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError): load_content_cases(path)
            path.write_text(json.dumps([{"id": "missing-topic"}]), encoding="utf-8")
            with self.assertRaises(ValueError): load_content_cases(path)

    def test_three_conditions_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = Mock(side_effect=["No-memory draft", "Memory draft", "Review-condition draft",
                '{"clarity":"ok","structure":"ok","audience_fit":"ok","tone_consistency":"ok","factual_risk":"check","improvements":[]}'])
            retriever = Mock(return_value=[{"id": 8, "content": "Context"}])
            summary = run_content_evaluation(object(), "user-a", [{"id":"brief-a","topic":"AI memory"}],
                tmp, generator=generator, retriever=retriever)
            self.assertEqual(summary["drafts"], 3)
            self.assertEqual(retriever.call_count, 2)
            self.assertTrue((Path(tmp) / "drafts/case-001-no_memory.md").exists())
            with (Path(tmp) / "ratings_template.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            records = [json.loads(line) for line in (Path(tmp) / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual([r["condition"] for r in records], ["no_memory", "memory", "memory_plus_review"])
            self.assertIn("editorial_review", records[-1])

    def test_review_condition_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = Mock(side_effect=["No-memory draft", "Memory draft"])
            summary = run_content_evaluation(object(), "user-a", [{"topic":"A"}], tmp,
                include_review=False, generator=generator, retriever=Mock(return_value=[]))
            self.assertEqual(summary["conditions"], ["no_memory", "memory"])
            self.assertEqual(generator.call_count, 2)


if __name__ == "__main__":
    unittest.main()
