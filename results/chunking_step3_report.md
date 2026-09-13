# Step 3 Structure-Aware Chunking Report

This is a document-representation development diagnostic, not a final thesis accuracy result.

## Before vs after

| Measure | Before: page-dominated | After: structure-aware |
|---|---:|---:|
| Documents | 1 | 1 |
| Extractable physical pages | 95 | 95 |
| Pages with retained evidence chunks | 95 | 92 |
| Blocks | 95 implicit page blocks | 475 semantic blocks |
| Retrieval chunks | 95 | 490 |
| Average words/chunk | 291.95 | 57.70 |
| Median words/chunk | 337 | 21.5 |
| Minimum words/chunk | 3 | 3 |
| Maximum words/chunk | 558 | 260 |
| Mixed-entity risk chunks | 50 | 16 |
| Over-limit chunks under Step 3 policy | 46 | 0 |

Mixed-entity diagnostics use the same generic course-code detector and flag more than three distinct codes in an unstructured chunk, or an unusually broad structured course record. Short structured table rows remain valid evidence even when below the generic small-chunk warning threshold.

## After: chunk types

| Type | Chunks |
|---|---:|
| Course records | 81 |
| Table rows | 117 |
| Table-like context/header blocks | 8 |
| Policy sections | 25 |
| Other detected sections | 250 |
| Generic fallback | 9 |
| Entity/course chunks (all types) | 198 |

## Representative pages

| Physical PDF page | Before | After |
|---|---|---|
| 20 | One 558-word page chunk containing attendance, absence, progress, discipline, and examination rules | Six coherent sections: Attendance; Absence during Semester; Performance Evaluation; Conduct and Discipline; Examination Rules; Rules for Repeat Examination |
| 51 | One page chunk containing the curriculum introduction and many first-semester course rows | One table-context block plus seven independently identified course rows |
| 52 | One page chunk containing two semesters and sixteen course rows | One table-context block plus sixteen course-row chunks, including independent CSE 205 and EEE 221 rows |
| 61 | One page chunk containing bibliography text and three course records | Three small reference/section blocks plus independent CSE 102, HSS 101, and HSS 111 course records |
| 66 | One page chunk containing CSE 103, CSE 104, and CSE 105 | Three independent course-record chunks |
| 68 | One page chunk containing EEE 221 and CSE 205 | Two independent course-record chunks |

## Cross-page representation

Chunks always retain their physical page. Deterministic `previous_chunk_id` and `next_chunk_id` links are recorded within a document. A conservative `continuation_of` link is added only when the next page begins with an unheaded fallback block and the prior chunk appears incomplete. Five such continuation links were detected in the current PDF. No text from unrelated neighboring sections is merged across pages.

## Quality diagnostics

- Empty chunks: 0
- Very long chunks: 0
- Missing provenance: 0
- Duplicate chunk IDs: 0
- Very short warnings: 76, primarily meaningful compact table rows/headings
- Duplicate-text warnings: 2; retained for inspection because automatic deletion is unsafe
- Mixed-entity warnings: 16; retained for inspection

## Performance

- PDF parse/extraction: approximately 3.08 seconds
- Block construction: approximately 0.25 seconds
- Complete structure-aware chunk construction after parsing: approximately 0.50 seconds
- Blocks: 475
- Chunks: 490
- BGE-M3 staged rebuild: approximately 3 minutes 43 seconds on CPU

## Retrieval-only development diagnostic

The same BGE-M3 model, FAISS type, retrieval logic, and reranking weights were used before and after.

| Language | Representation | Page Hit@1 | Page Hit@3 | MRR@3 |
|---|---|---:|---:|---:|
| English | Old | 56.00% | 83.00% | 0.6783 |
| English | New | 23.00% | 80.00% | 0.4733 |
| Bangla | Old | 50.00% | 81.00% | 0.6383 |
| Bangla | New | 23.00% | 64.00% | 0.4117 |
| Banglish | Old | 52.00% | 86.00% | 0.6717 |
| Banglish | New | 21.00% | 59.00% | 0.3833 |

Overall entity retrieval changed from 98.56%/99.52% Hit@1/Hit@3 to 96.63%/99.52%. Overall evidence-bound field retrieval changed from 85.66%/88.97% to 76.84%/92.28%.

Page metrics are not directly equivalent after one page becomes many candidates, but the regression is real under the unchanged top-3 selection policy and must be addressed in a later retrieval-focused step. No reranking or retrieval tuning was performed here.
