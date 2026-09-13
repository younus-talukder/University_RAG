# University Multilingual RAG Chatbot

This project is a thesis prototype for a multilingual university RAG chatbot grounded only in the official PDFs in `data/documents/`.

## Current Project State

- Knowledge base: one active curriculum PDF.
- Dataset: 100 base questions with English, Bangla, and Banglish variants.
- Embedding model: `BAAI/bge-m3`.
- Retrieval: pinned BGE-M3 + FAISS `IndexFlatIP`, local BM25, metadata candidates, and reciprocal-rank fusion (RRF).
- Generator: local Qwen2.5 GGUF through `llama-cpp-python`.
- UI: Streamlit.
- Current vector index: 490 structure-aware chunks from 92 evidence-bearing pages.

## Knowledge And Evaluation Data

- Knowledge source: `data/documents/curricula_BSc-Curriculum-New.pdf`
- Evaluation dataset: `data/questions/questions.csv`
- Source question workbook: `data/questions/uap_cse_multilingual_rag_dataset.xlsx`
- Dataset columns: `question_id`, `english_question`, `bangla_question`, `banglish_question`, language-specific ground-truth answers, `expected_source`, `expected_page`

The question-answer CSV is used for testing and evaluation only. It is not used as retrieval or generation context.

## Pipeline

```text
Student Question -> Language Detection -> BGE-M3/FAISS + BM25 + Metadata Candidates
-> Rank-Based RRF -> Top-K Relevant Chunks -> Step-1 Evidence Validation
-> Fast Extractive Answer or Qwen2.5 GGUF -> Language Validation
-> Final Answer
```

## Repository Structure

```text
app.py                    Streamlit application
run_app.py                Streamlit launcher for local environments
data/documents/           Current source PDF documents
data/questions/           Canonical dataset and source workbook
src/                      Runtime RAG modules
scripts/                  Ingestion, indexing, evaluation, and utilities
tests/                    Current regression and behavior tests
vector_db/                Active FAISS index, metadata, and manifest
docs/                     Thesis process documentation
archive/                  Legacy scripts, reports, tests, and old results
```

## Setup

Install dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

The required local models are:

- `BAAI/bge-m3`
- `Qwen/Qwen2.5-1.5B-Instruct`

## Commands

Inspect PDF extraction and chunking:

```bash
python scripts/ingest.py
```

Import the question workbook into the canonical CSV:

```bash
python scripts/import_question_workbook.py data/questions/uap_cse_multilingual_rag_dataset.xlsx
```

Build the mutually compatible dense and sparse retrieval artifacts:

```bash
python scripts/build_index.py
```

Force a rebuild even if the document manifest is fresh:

```bash
python scripts/build_index.py --force
```

## Reproducible BGE-M3 embeddings

Semantic indexing and queries use the same pinned configuration:

- Model: `BAAI/bge-m3`
- Immutable revision: `5617a9f61b028005a4858fdac845db406aefb181`
- Dimension: 1024 float32 values
- Similarity contract: normalized vectors with FAISS `IndexFlatIP`
- Default loading policy: local-only/offline
- Cached weight format for the pinned complete snapshot: `pytorch_model.bin`

The loader validates the snapshot, tokenizer, configuration, sentence-transformer files, and weight policy before use. It never silently substitutes another embedding model. Set `EMBEDDING_LOCAL_ONLY=false` only when intentionally allowing Hugging Face to resolve missing files. A changed model revision, normalization policy, dimension, weight format, chunker configuration, or corpus makes the existing index incompatible and requires rebuilding it.

The most recent build status and timings are written to `results/index_build_report.json`. Exact embedding provenance and installed library versions are stored in `vector_db/index_manifest.json`. These files do not contain machine-specific model-cache paths.

## Hybrid retrieval

The retrieval unit remains each Step-3 structured chunk. At query time the system independently retrieves up to 30 dense FAISS candidates, up to 30 positive-score BM25 candidates, and up to 20 exact structured-metadata candidates. It merges duplicate `chunk_id` values and ranks the union with weighted reciprocal-rank fusion:

```text
dense contribution    = 1.0 / (60 + dense rank)
sparse contribution   = 1.0 / (60 + sparse rank)
metadata contribution = 0.8 / (60 + metadata rank)
```

Raw cosine and BM25 scores are retained for debugging but are never added together. The final top three are the first three items in the fused ordering. If BM25 has no positive lexical matches, dense candidates keep their ordering. Dense-search operational failures remain visible instead of being silently relabeled as semantic success.

`vector_db/sparse_index.pkl` is published atomically with FAISS, metadata, and the manifest. Corpus changes, structured-chunk changes, embedding incompatibility, or sparse tokenizer/BM25 configuration changes make the artifact set stale. The build command reuses an unchanged compatible set and can attach a newly required sparse index to a verified dense index without re-embedding.

Run the 300-query development comparison (answer bank and generation are not used):

```bash
python scripts/evaluate_step5_hybrid.py
```

This writes `results/step5_dense_baseline.json`, `results/step5_hybrid_results.json`, and `results/step5_hybrid_report.md`. These are development/regression diagnostics, not final thesis results.

## Multilingual query normalization

Runtime query handling preserves three bounded representations: the original user query, a conservative normalized query, and an original-plus-normalized sparse retrieval query. The shared layer handles Unicode NFKC, Bengali digits, course-code formatting, a small documented Banglish variant set, and selected Bengali/mixed terminology without using translation APIs or dataset-specific mappings.

The original query remains the BGE-M3 input because the Step 6 development comparison found that this best protects field and supportability top-three performance. BM25 receives the bounded original-plus-normalized representation, while entity and field understanding use the normalized form. Debug output exposes these representations and the language-detection reason.

Optional second-stage reranking uses the locally cached, immutable `BAAI/bge-reranker-v2-m3` revision `b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1`. Runtime resolution is local-only, validates the complete snapshot, and raises a visible error instead of silently falling back. RRF creates a bounded candidate pool and the neural logits determine ordering only; raw RRF and reranker scores are never added.

`RERANKER_ENABLED=false` remains the default. The 300-query CPU development evaluation rejected default enablement: the least harmful tested pool (`K=10`) changed Supportable@1 from 88.00% to 86.33%, Page Hit@1 from 57.00% to 28.67%, and added 4.64 seconds median reranker latency. The implementation remains available for explicit experiments without changing the approved normalization plus hybrid baseline.

Run the normalization-only development comparison:

```bash
python scripts/evaluate_step6_normalization.py
```

The Step 5 reports are not overwritten. Step 6 results are written separately under `results/step6_*`.

Run the explicit reranker comparison (100 English, 100 Bangla, and 100 Banglish queries; answer bank and generation off):

```bash
python scripts/evaluate_step6_reranker.py
```

The expensive real-model smoke test is opt-in with `RUN_REAL_RERANKER_TEST=1`; normal unit tests use mocks and do not load the model.

Run a test query:

```bash
python scripts/test_query.py
```

Run regression tests:

```bash
python -m unittest discover -s tests -v
```

Run controlled fast evaluation:

```bash
python scripts/evaluate.py --mode fast
```

Run controlled local GGUF evaluation:

```bash
python scripts/evaluate.py --mode gguf
```

Controlled evaluation disables the curated answer bank by default. Add `--answer-bank` only when intentionally measuring answer-bank-assisted behavior.

Run the UI:

```bash
streamlit run app.py
```

If the Windows `streamlit.exe` launcher fails, use the project launcher instead:

```bash
python run_app.py 8510
```

Then open the port you passed, for example `http://localhost:8510`. With no argument, `run_app.py` uses `http://localhost:8511`.

## Retrieval Metrics

The current CSV includes expected source and page labels for the curriculum PDF. Retrieval evaluation can therefore compute source/page ranking metrics, while final factual quality and groundedness still require manual thesis review.
