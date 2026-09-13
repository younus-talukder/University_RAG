from __future__ import annotations

import os
import unittest

from src.reranker import get_reranker_model, rerank_candidate_pool


@unittest.skipUnless(
    os.environ.get("RUN_REAL_RERANKER_TEST") == "1",
    "Set RUN_REAL_RERANKER_TEST=1 to run the explicit 2+ GB model smoke test.",
)
class RealRerankerIntegrationTests(unittest.TestCase):
    def test_multilingual_pairs_score_and_provenance_survives(self) -> None:
        scorer = get_reranker_model()
        cases = (
            "What is the prerequisite?",
            "পূর্বশর্ত কী?",
            "prerequisite ki?",
        )
        for query in cases:
            candidates = [
                {"chunk_id": "relevant", "text": "Prerequisite: ABC 101", "source": "a.pdf", "page": 3, "final_rank": 2},
                {"chunk_id": "other", "text": "Course title: History", "source": "b.pdf", "page": 8, "final_rank": 1},
            ]
            ranked = rerank_candidate_pool(query, candidates, scorer, 2)
            self.assertEqual(len(ranked), 2)
            self.assertTrue(all("reranker_score" in item for item in ranked))
            self.assertTrue(all(item.get("source") and item.get("page") for item in ranked))


if __name__ == "__main__":
    unittest.main()
