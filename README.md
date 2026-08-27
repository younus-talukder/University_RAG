# University Multilingual RAG Chatbot

This project is a thesis prototype for a multilingual university RAG chatbot grounded only in the official PDFs in `data/documents/`.

## Knowledge And Evaluation Data

- Knowledge source: `data/documents/document1.pdf`, `document2.pdf`, `document3.pdf`
- Evaluation dataset: `data/questions/questions.csv`
- Dataset columns: `ID`, `English Query`, `Bengali Query`, `Banglish Query`, `Expected Answer (Ground Truth)`

The question-answer CSV is used for testing and evaluation only. It is not used as retrieval or generation context.

## Pipeline

```text
Student Question -> Language Detection -> BGE-M3 Embedding -> FAISS Search
-> Top-K Relevant Chunks -> Qwen2.5-1.5B-Instruct -> Language Validation
-> Final Answer
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

Build the FAISS index:

```bash
python scripts/build_index.py
```

Run a test query:

```bash
python scripts/test_query.py
```

Run evaluation:

```bash
python scripts/evaluate.py
```

Run the UI:

```bash
streamlit run app.py
```

If the Windows `streamlit.exe` launcher fails, use the project launcher instead:

```bash
python run_app.py 8510
```

Then open `http://localhost:8511` when using the default port.

Diagnose local runtime issues:

```bash
python diagnose_runtime.py
```

## Retrieval Metrics

The current CSV does not include ground-truth source/page annotations. Proper Hit@K, Recall@K, and MRR require those labels, so this prototype records retrieved source/page/score for qualitative inspection until annotations are added.
