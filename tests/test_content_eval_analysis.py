import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_content_eval import analyze_ratings


class ContentRatingAnalysisTests(unittest.TestCase):
    def test_condition_summaries_paired_differences_and_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = [
                {"blind_draft_id":"D1","case_id":"case-a","condition":"no_memory"},
                {"blind_draft_id":"D2","case_id":"case-a","condition":"memory"},
                {"blind_draft_id":"D3","case_id":"case-a","condition":"memory_plus_review"},
                {"blind_draft_id":"D4","case_id":"case-b","condition":"no_memory"},
                {"blind_draft_id":"D5","case_id":"case-b","condition":"memory"},
                {"blind_draft_id":"D6","case_id":"case-b","condition":"memory_plus_review"},
            ]
            (root / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
            )
            fields = [
                "reviewer_id","blind_draft_id","relevance","personalization",
                "factual_consistency","style_consistency","completeness","readability",
                "unsupported_claims","reviewer_notes",
            ]
            ratings = [
                ["r1","D1",3,2,4,3,3,3,2,""],
                ["r2","D1",3,3,4,3,3,3,2,""],
                ["r1","D2",4,4,4,4,4,4,1,""],
                ["r2","D2",5,4,4,4,4,4,1,""],
                ["r1","D3",4,4,4,4,4,4,1,""],
                ["r1","D4",2,2,3,3,2,3,3,""],
                ["r1","D5",4,4,4,4,4,4,1,""],
                ["r1","D6",4,4,4,4,4,4,1,""],
            ]
            with (root / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(ratings)
            report = analyze_ratings(root / "ratings.csv", root / "manifest.jsonl")
            self.assertEqual(report["rated_drafts"], 6)
            self.assertEqual(report["reviewers_identified"], 2)
            self.assertAlmostEqual(
                report["paired_comparisons"]["memory_minus_no_memory"]["relevance"]["mean_difference"],
                1.5,
            )
            self.assertIsNotNone(report["inter_rater_agreement"]["relevance"]["exact_agreement"])

    def test_invalid_scale_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.jsonl").write_text(
                json.dumps({"blind_draft_id":"D1","case_id":"a","condition":"memory"}) + "\n",
                encoding="utf-8",
            )
            with (root / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["blind_draft_id","relevance"])
                writer.writerow(["D1",6])
            with self.assertRaises(ValueError):
                analyze_ratings(root / "ratings.csv", root / "manifest.jsonl")


if __name__ == "__main__":
    unittest.main()
