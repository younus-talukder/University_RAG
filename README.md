# University Multilingual RAG Chatbot

This project is a thesis prototype for a multilingual university RAG chatbot grounded only in the official PDFs in `data/documents/`.

## Current Project State

- Knowledge base: one active curriculum PDF.
- Dataset: 100 base questions with English, Bangla, and Banglish variants.
- Embedding model: `BAAI/bge-m3`.
- Vector store: FAISS `IndexFlatIP`.
- Generator: local Qwen2.5 GGUF through `llama-cpp-python`.
- UI: Streamlit.
- Current vector index: 95 chunks.

## Knowledge And Evaluation Data

- Knowledge source: `data/documents/curricula_BSc-Curriculum-New.pdf`
- Evaluation dataset: `data/questions/questions.csv`
- Source question workbook: `data/questions/uap_cse_multilingual_rag_dataset.xlsx`
- Dataset columns: `question_id`, `english_question`, `bangla_question`, `banglish_question`, language-specific ground-truth answers, `expected_source`, `expected_page`

The question-answer CSV is used for testing and evaluation only. It is not used as retrieval or generation context.

## Pipeline

```text
Student Question -> Language Detection -> BGE-M3 Embedding -> FAISS Search
-> Top-K Relevant Chunks -> Fast Extractive Answer or Qwen2.5 GGUF -> Language Validation
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

Build the FAISS index:

```bash
python scripts/build_index.py
```

Force a rebuild even if the document manifest is fresh:

```bash
python scripts/build_index.py --force
```

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
