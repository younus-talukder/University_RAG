from __future__ import annotations

from enum import StrEnum


class SemanticErrorCategory(StrEnum):
    CORRECT = "CORRECT"
    WRONG_NEGATION = "WRONG_NEGATION"
    WRONG_POLARITY = "WRONG_POLARITY"
    WRONG_RELATION = "WRONG_RELATION"
    UNSUPPORTED_FACT = "UNSUPPORTED_FACT"
    MISSING_REQUIRED_FACT = "MISSING_REQUIRED_FACT"
    SEMANTIC_NONSENSE = "SEMANTIC_NONSENSE"
    LANGUAGE_MISMATCH = "LANGUAGE_MISMATCH"
    PURE_ENGLISH_BANGLISH = "PURE_ENGLISH_BANGLISH"
    BANGLA_GRAMMAR_FAILURE = "BANGLA_GRAMMAR_FAILURE"
    BANGLISH_GRAMMAR_FAILURE = "BANGLISH_GRAMMAR_FAILURE"
    TRUNCATED = "TRUNCATED"
    OVERBROAD = "OVERBROAD"
    IRRELEVANT = "IRRELEVANT"
    SAFE_ABSTENTION = "SAFE_ABSTENTION"


ERROR_TAXONOMY = {
    SemanticErrorCategory.CORRECT: "Meaning and required facts are correct and evidence-bound.",
    SemanticErrorCategory.WRONG_NEGATION: "A source fact was incorrectly negated or a source negation was removed.",
    SemanticErrorCategory.WRONG_POLARITY: "The answer reverses an affirmative/negative, yes/no, existence, permission, or requirement claim.",
    SemanticErrorCategory.WRONG_RELATION: "The subject, relation, authority, or object/value is attached incorrectly.",
    SemanticErrorCategory.UNSUPPORTED_FACT: "The answer adds a factual claim not supported by verified evidence.",
    SemanticErrorCategory.MISSING_REQUIRED_FACT: "A fact needed to answer the question is absent.",
    SemanticErrorCategory.SEMANTIC_NONSENSE: "Words or clauses do not form a coherent interpretable claim.",
    SemanticErrorCategory.LANGUAGE_MISMATCH: "The response is not in the requested language or script.",
    SemanticErrorCategory.PURE_ENGLISH_BANGLISH: "A Banglish response is English prose without meaningful Latin-script Bangla structure.",
    SemanticErrorCategory.BANGLA_GRAMMAR_FAILURE: "Bangla grammar is broken enough to damage clarity or meaning.",
    SemanticErrorCategory.BANGLISH_GRAMMAR_FAILURE: "Latin-script Bangla grammar is broken enough to damage clarity or meaning.",
    SemanticErrorCategory.TRUNCATED: "The answer ends before a required claim is complete.",
    SemanticErrorCategory.OVERBROAD: "The answer generalizes beyond the scope supported by evidence.",
    SemanticErrorCategory.IRRELEVANT: "The answer does not address the requested entity, relation, or field.",
    SemanticErrorCategory.SAFE_ABSTENTION: "The system safely declines because semantic integrity or evidence is insufficient.",
}


__all__ = ["ERROR_TAXONOMY", "SemanticErrorCategory"]
