# Step 8D — conflict scope and equivalence stabilization

## 1. STEP 8D GOAL

Stop false document-conflict claims while preserving genuine same-relation contradictions. Step 9 was not started.

## 2. FILES CHANGED

`src/evidence.py`, `tests/test_step8d_conflict_scope.py`, `scripts/evaluate_step8d.py`, `scripts/audit_step8d_credit_exceptions.py`, `scripts/report_step8d.py`, and the new Step 8D output builder. Prior Step-8 artifacts were preserved; no commit was made.

## 3. Q018 ROOT CAUSE

The former field-only comparator treated course/lab credit values as possible contradictions of the theoretical-course credit-assignment rule. Frozen Bangla conflict candidates: page 17 theory rule (no numeric field value), page 81 lab value 1.50, page 63 course value 3.00. Frozen Banglish candidates: page 17 theory, page 63 lab/course value 1.5, page 81 course value 1.50. The numeric values belong to different relations. Current complete top-three retrieval and verified-evidence trace:

- english: field `credits`; public relation label `none`; conflict identity `theoretical_course_credit_rule`; retrieved p17 doc-4d373684ce4879ed066003f5-p000017-b0004-c0001, p18 doc-4d373684ce4879ed066003f5-p000018-b0005-c0001, p19 doc-4d373684ce4879ed066003f5-p000019-b0004-c0001; verified p17 doc-4d373684ce4879ed066003f5-p000017-b0004-c0001; normalized conflict values [].
- bangla: field `credits`; public relation label `none`; conflict identity `theoretical_course_credit_rule`; retrieved p17 doc-4d373684ce4879ed066003f5-p000017-b0004-c0001, p81 doc-4d373684ce4879ed066003f5-p000081-b0001-c0001, p63 doc-4d373684ce4879ed066003f5-p000063-b0005-c0001; verified p17 doc-4d373684ce4879ed066003f5-p000017-b0004-c0001; normalized conflict values [].
- banglish: field `credits`; public relation label `none`; conflict identity `theoretical_course_credit_rule`; retrieved p17 doc-4d373684ce4879ed066003f5-p000017-b0004-c0001, p63 doc-4d373684ce4879ed066003f5-p000063-b0004-c0001, p81 doc-4d373684ce4879ed066003f5-p000081-b0003-c0001; verified p17 doc-4d373684ce4879ed066003f5-p000017-b0004-c0001; normalized conflict values [].

The full passage text and metadata are retained in `results/step8d_targeted_checkpoint_v3.jsonl`.

## 4. CONFLICT-SCOPE MODEL

A reusable conflict identity carries entity, field, relation, scope, material qualifiers, and document context. Values are compared only under a matching semantic key. Document context is retained for review, not used for automatic recency arbitration.

## 5. RELATION / QUALIFIER COMPATIBILITY

Theoretical credit assignment, lab assignment, course credit value, and semester total are distinct. Attendance, seat reservation, marks, and admission percentages are distinct domains. Explicit category, program level, every-course, and per-semester qualifiers remain in the comparison identity.

## 6. Q018 BEFORE / AFTER

- english: SUPPORTED → SUPPORTED; verified page(s) 17; answer: One lecture per week per semester is equivalent to one credit for theoretical courses.
- bangla: CONFLICTING_EVIDENCE → GENERATION_REJECTED; verified page(s) 17; answer: প্রাসঙ্গিক তথ্য পাওয়া গেছে, কিন্তু নির্ভরযোগ্য উত্তর তৈরি করা যায়নি।
- banglish: CONFLICTING_EVIDENCE → SUPPORTED; verified page(s) 17; answer: Theoretical course-er one credit assign koreche holo one lecture per week per semester.

## 7. Q088 COMPLETE EVIDENCE AUDIT

- Evidence A: `curricula_BSc-Curriculum-New.pdf`, PDF page 51, `doc-4d373684ce4879ed066003f5-p000051-b0009-c0001`; `Nil` → `NO_PREREQUISITE`; MTH 101 Math I: Basic Calcu lus, Coordinate Geometry 3.00 Nil Total 19.00…
- Evidence B: `curricula_BSc-Curriculum-New.pdf`, PDF page 63, `doc-4d373684ce4879ed066003f5-p000063-b0005-c0001`; `N/A` → `NO_PREREQUISITE`; Course Code: MTH 101 Course Title: Math I Credits: 3.00 Prerequisite: N/A Three dimensional geometry: Co-ordinates in three dimensions, direction cosines and direction ratios, planes, sphere, straight line and conicoids (basic definition and properties only). Concept of functions…

Both contributing passages are in `results/step8d_q088_evidence_audit.csv` with complete raw text, parent block, source structure, scope, and context. A retrieved MTH 201 passage was excluded because MTH 101 is mentioned only as its prerequisite, not as the subject course.

## 8. Q088 ADJUDICATION

Equivalent: YES, for the verified MTH 101 prerequisite relation in the same BSc curriculum PDF. Genuine conflict: NO. Ambiguous: NO. Page 51 has `Nil` in the MTH 101 Pre-Requisite table column; page 63 explicitly has `Prerequisite: N/A` under MTH 101. No distinct edition or supersession metadata was present in these chunks.

## 9. NO-PREREQUISITE NORMALIZATION

`Nil` and `None` retain existing equivalence. `N/A` maps to `NO_PREREQUISITE` only after it is extracted from a verified prerequisite label/row; an unrelated N/A is not globally rewritten.

## 10. TRUE CONFLICT REGRESSION

Same-course credit 3.00 versus 4.00: CONFLICTING. Same-policy/same-category attendance requirement 70% versus 75%: CONFLICTING. Different document editions are retained for review; newer is not assumed to win.

## 11. FALSE-CONFLICT REGRESSION

Theory versus lab rule: no conflict. Course credit versus semester total: no conflict. Nil/None/N/A in a verified prerequisite field: equivalent. Explicitly different policy categories: no conflict.

## 12. TARGETED CASE RESULTS

- Q014: english=SUPPORTED, bangla=SUPPORTED, banglish=SUPPORTED
- Q016: english=GENERATION_REJECTED, bangla=RETRIEVAL_UNCERTAIN, banglish=GENERATION_REJECTED
- Q018: english=SUPPORTED, bangla=GENERATION_REJECTED, banglish=SUPPORTED
- Q020: english=SUPPORTED, bangla=SUPPORTED, banglish=SUPPORTED
- Q031: english=GENERATION_REJECTED, bangla=GENERATION_REJECTED, banglish=GENERATION_REJECTED
- Q081: english=SUPPORTED, bangla=SUPPORTED, banglish=SUPPORTED
- Q088: english=SUPPORTED, bangla=SUPPORTED, banglish=SUPPORTED
- Q095 collateral equivalence: all three languages supported.
- Q034 HSS 101: all three safely report a genuine page-51 3.00 versus page-61 1.50 credit conflict.
- Q065 CSE 304: all three safely report a genuine page-53 0.75 versus page-78 3.00 credit conflict. Neither source value is silently selected.

## 13. MISLEADING-STATUS AUDIT

Targeted false conflict: 0. Targeted false ambiguity: 0. Other: 6 genuine newly surfaced conflict variants (Q034/Q065), 6 safe coverage-gap variants (Q016/Q031), and 0 unadjudicated possible insufficient-evidence variants. The CSV retains prior/current status per case.

## 14. UNSAFE RETURNED ANSWERS

Targeted count: 0. Target: 0. This is not a new full-300 audit.

## 15. TEST RESULTS

Previous: 232; new: 6; total: 238; failures: 0; skipped: 1.

## 16. FULL 300 RERUN

Performed: NO. The prior full run had eight conflict-status rows, all in Q018, Q088, or Q095; all were targeted. A read-only scan of all 491 indexed chunks covered 49 credit-question course entities and 20 prerequisite-question entities. It found exactly two further contradictory credit entities (HSS 101, CSE 304) and no divergent prerequisites; all six affected language variants were then run live. This bounds the detected status changes without a memory-heavy 300-answer run. The prior 258/300 safe-answer coverage is historical, not a current Step-8D score; no new coverage or accuracy score is claimed.

## 17. MEMORY

Sequential target peak RSS 5.37 GiB, minimum sampled available RAM 237 MiB, peak sampled system pagefile use 3.10 GiB. No generator settings or parallel workers changed.

## 18. FINAL MANUAL REVIEW FILE

`results/step8d_final_manual_review.csv`: 30 fresh current-pipeline rows, 10 per language, including Q018, Q088, Q081, and both newly surfaced credit conflicts. Human review fields are blank.

## 19. STEP 1–8C COMPATIBILITY

Q014 emails, Q016 safe duration handling, Q020 three-language distribution, Q031 non-ambiguity, Q081 None/Nil, Q018 non-conflict, Q088/Q095 contextual equivalence, synthetic true conflicts, and the complete unit suite passed. Frozen Step-7H was not rerun in this narrow phase.

## 20. 70-PDF READINESS

Architectural only. Cross-document identity and version metadata are designed conservatively; 70-PDF ingestion and memory/quality at that scale were not measured.

## 21. 6000-QUESTION INDEPENDENCE

Production rules use question and evidence text, not dataset IDs or reference answers. This targeted development gate is not an independent 6000-question accuracy evaluation.

## 22. GIT DIFF SUMMARY

No commit. Existing dirty Step-7/8 work was preserved. This phase changes only conflict handling, synthetic tests, and new Step-8D evaluation/report outputs.

## 23. REMAINING RISKS

Q016 and Q031 remain safe answer-coverage gaps; Q018 Bangla also safely rejects generation after the false conflict is removed. HSS 101 and CSE 304 have genuine conflicting values inside the source PDF; the system now abstains instead of choosing one. Some Bangla/Banglish wording remains awkward. The 8-GB machine has little RAM headroom. Source edition metadata is incomplete; genuine cross-version discrepancies require explicit review, never automatic newer-wins arbitration.

## 24. STEP 8 ENGINEERING-COMPLETE?

YES for the specified development trust and compatibility gates, subject to human review of the new 30-row sample. This is not a thesis accuracy or scale claim.

## 25. READY FOR STEP 9?

YES for a separately authorized phase. Step 9 has not begun.

## 26. HUMAN INPUT NEEDED?

No Q088 adjudication is needed: both complete source contexts establish the same MTH 101 prerequisite field. HSS 101 and CSE 304 source values genuinely disagree; an authoritative correction would require a document owner, but the system safely surfaces the conflicts without that decision. Human review of the new 30-row sample remains a separate quality step.

## 27. NEXT STEP

Review the new sample and freeze Step 8 only after the reviewer accepts the conflict/status changes. Plan Step 9 separately with explicit authorization and memory constraints.
