from __future__ import annotations

import unittest

from src.retriever import (
    build_query_profile,
    detect_retrieval_course_code,
    normalize_query_for_retrieval,
    rerank_candidates,
    score_candidate_components,
    select_retrieval_results,
    section_tags,
)


def candidate(text: str, source: str = "curricula_BSc-Curriculum-New.pdf", page: int = 1, score: float = 1.0) -> dict:
    return {
        "text": text,
        "source": source,
        "page": page,
        "chunk_id": f"{source}-p{page}",
        "score": score,
    }


class RetrieverRerankTests(unittest.TestCase):
    def test_requested_clo_number_reranks_exact_chunk_first(self) -> None:
        rows = [
            candidate("Course Learning Outcomes CLO 1 Describe hardware.", page=2, score=1.8),
            candidate("Course Learning Outcomes CLO 3 Design flowcharts.", page=2, score=1.1),
        ]
        ranked = rerank_candidates("What is CLO 3 of CSE 101?", rows, debug=True)
        self.assertIn("CLO 3", ranked[0]["text"])
        self.assertGreater(ranked[0]["rerank_debug"]["field_boost"], 0)

    def test_requested_week_reranks_exact_week_first(self) -> None:
        rows = [
            candidate("Week 2 Keyboard and Mouse", page=3, score=1.7),
            candidate("Week 5 Number System", page=5, score=1.0),
        ]
        ranked = rerank_candidates("CSE 101 week 5 e ki porano hoy?", rows, debug=True)
        self.assertEqual(ranked[0]["page"], 5)
        self.assertGreater(ranked[0]["rerank_debug"]["field_boost"], 0)

    def test_course_mismatch_penalty(self) -> None:
        profile = build_query_profile("What is CLO 2 of CSE 101?")
        right = score_candidate_components(profile, candidate("CSE 101 CLO 2 Number systems"))
        wrong = score_candidate_components(profile, candidate("EEE 221 CLO 2 Integrals", "old-document.pdf"))
        self.assertEqual(right["penalty"], 0.0)
        self.assertLess(wrong["penalty"], 0.0)
        self.assertGreater(right["final_score"], wrong["final_score"])

    def test_intent_specific_boost_and_section_tags(self) -> None:
        profile = build_query_profile("What are the objectives of MTH 101?")
        item = candidate("11. Course Objectives and Course Summary: Define limits.")
        components = score_candidate_components(profile, item)
        self.assertGreater(components["intent_boost"], 0)
        self.assertIn("course_objective", section_tags(item["text"]))

    def test_banglish_normalization_is_conservative(self) -> None:
        normalized = normalize_query_for_retrieval("Week 5 e final exam e koto percent boraddo?")
        self.assertIn("week 5", normalized)
        self.assertIn("final exam", normalized)
        self.assertIn("allocated", normalized)

    def test_bangla_numeric_normalization(self) -> None:
        profile = build_query_profile("৫ম সপ্তাহে কী পড়ানো হয়?")
        self.assertEqual(profile.week_number, 5)

    def test_topic_vocabulary_does_not_invent_course_entities(self) -> None:
        cases = [
            ("What reading techniques are included in the course?", ""),
            ("Computer software-er dui-ti type ki?", ""),
            ("What topics are included under Functions?", ""),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(detect_retrieval_course_code(question), expected)

    def test_score_component_debug_shape(self) -> None:
        profile = build_query_profile("What percentage is allocated to the Final Exam in CSE 101?")
        components = score_candidate_components(
            profile,
            candidate("Weighting Assessment Type Final Exam (50%) Written Exam", page=9),
        )
        self.assertEqual(
            set(components),
            {
                "semantic_score",
                "course_boost",
                "intent_boost",
                "field_boost",
                "lexical_boost",
                "section_boost",
                "topic_plan_boost",
                "penalty",
                "metadata_penalty",
                "final_score",
            },
        )
        self.assertGreater(components["field_boost"], 0)

    def test_topic_subject_boost_prefers_matching_content_heading(self) -> None:
        rows = [
            candidate("Course Title: Computer Fundamentals and Programming Credit Value: 3.0", page=1, score=1.4),
            candidate("Course Content: Computer Software: application software, system software", page=3, score=1.0),
        ]
        ranked = rerank_candidates("Computer software-er under-e kon topic ache?", rows, debug=True)
        self.assertEqual(ranked[0]["page"], 3)
        self.assertGreater(ranked[0]["rerank_debug"]["field_boost"], 0)

    def test_metadata_page_penalized_for_specific_topic_query(self) -> None:
        profile = build_query_profile("What topics are covered in CSE 101?")
        components = score_candidate_components(
            profile,
            candidate("Course No. / Course Code: CSE 101 Course Title: Computer Fundamentals and Programming", page=1),
        )
        self.assertLess(components["metadata_penalty"], 0)

    def test_topic_plan_section_boosts_syllabus_pages(self) -> None:
        profile = build_query_profile("What topics are included under repetition and loop statements?")
        components = score_candidate_components(
            profile,
            candidate(
                "15. Alignment of topics of the courses with CLOs: Repetition and Loop Statements CLO4 "
                "16. Class Schedule/Lesson Plan/Weekly plan: Week 10",
                page=3,
            ),
        )
        self.assertGreater(components["topic_plan_boost"], 0)

    def test_selection_returns_first_k_from_final_ranking(self) -> None:
        ranked = [
            {"source": "curricula_BSc-Curriculum-New.pdf", "chunk_id": "a", "score": 9.0, "_semantic_score": 0.5},
            {"source": "curricula_BSc-Curriculum-New.pdf", "chunk_id": "b", "score": 8.0, "_semantic_score": 0.3},
            {"source": "curricula_BSc-Curriculum-New.pdf", "chunk_id": "c", "score": 7.0, "_semantic_score": 0.9},
            {"source": "curricula_BSc-Curriculum-New.pdf", "chunk_id": "d", "score": 6.0, "_semantic_score": 0.8},
        ]
        selected = select_retrieval_results(ranked, top_k=3)
        self.assertEqual([item["chunk_id"] for item in selected], ["a", "b", "c"])

    def test_top_k_ordering_stability_uses_semantic_tiebreak(self) -> None:
        rows = [
            candidate("Course Content: Number Systems", page=2, score=1.0),
            candidate("Course Content: Number Systems", page=3, score=1.2),
        ]
        ranked = rerank_candidates("What topics are covered in CSE 101?", rows, debug=True)
        self.assertEqual(ranked[0]["page"], 3)


if __name__ == "__main__":
    unittest.main()
