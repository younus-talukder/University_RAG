import os
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / 'debug_phasec.txt'
lines = []
lines.append('START')
lines.append(f'PYTHON={sys.executable}')
lines.append(f'VERSION={sys.version}')
lines.append(f'CWD={os.getcwd()}')
lines.append(f'ROOT_EXISTS={root.exists()}')
for name in ['sentence_transformers', 'faiss', 'transformers', 'torch', 'pypdf']:
    try:
        __import__(name)
        lines.append(f'{name}=OK')
    except Exception as exc:
        lines.append(f'{name}=ERROR:{type(exc).__name__}:{exc}')

try:
    from src.pdf_loader import load_pdf_documents
    pages, errors = load_pdf_documents(root / 'data/documents')
    lines.append(f'PAGES={len(pages)} ERRORS={len(errors)}')
    if pages:
        lines.append(f'FIRST_SOURCE={pages[0].get("source")} FIRST_PAGE={pages[0].get("page")}')
except Exception:
    lines.append('PDF_LOAD_ERROR')
    lines.extend(traceback.format_exc().splitlines())

try:
    from src.embeddings import EmbeddingModel
    model = EmbeddingModel()
    vec = model.embed('test question')
    lines.append(f'EMBED_DIM={len(vec)}')
except Exception:
    lines.append('EMBED_ERROR')
    lines.extend(traceback.format_exc().splitlines())

out.write_text('\n'.join(lines), encoding='utf-8')
print('WROTE', out)
