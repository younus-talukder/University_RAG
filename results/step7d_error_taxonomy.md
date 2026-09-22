# Step 7D Multilingual Semantic-Error Taxonomy

This taxonomy is generic and independent of question IDs, pages, PDF names, course lists, reference answers, and dataset size.

| Category | Meaning |
|---|---|
| `CORRECT` | Meaning and required facts are correct and evidence-bound. |
| `WRONG_NEGATION` | A source fact was incorrectly negated or a source negation was removed. |
| `WRONG_POLARITY` | Affirmative/negative, yes/no, existence, permission, or requirement meaning is reversed. |
| `WRONG_RELATION` | A subject, relation, authority, or object/value is attached incorrectly. |
| `UNSUPPORTED_FACT` | A factual claim is absent from verified evidence. |
| `MISSING_REQUIRED_FACT` | A fact needed to answer the question is omitted. |
| `SEMANTIC_NONSENSE` | The text does not form a coherent interpretable claim. |
| `LANGUAGE_MISMATCH` | The response uses the wrong language or script. |
| `PURE_ENGLISH_BANGLISH` | Banglish is English prose without meaningful Latin-script Bangla structure. |
| `BANGLA_GRAMMAR_FAILURE` | Broken Bangla grammar damages clarity or meaning. |
| `BANGLISH_GRAMMAR_FAILURE` | Broken Latin-script Bangla grammar damages clarity or meaning. |
| `TRUNCATED` | A required claim ends incomplete. |
| `OVERBROAD` | The answer generalizes beyond evidence. |
| `IRRELEVANT` | The answer does not address the requested entity/relation/field. |
| `SAFE_ABSTENTION` | The system safely declines because semantic integrity or evidence is insufficient. |

Human reviewers may select one primary category and explain secondary issues in `notes`. Automated code does not populate human ratings or taxonomy labels.
