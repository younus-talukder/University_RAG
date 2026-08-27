import importlib
import os
import sys
from pathlib import Path

root = Path(r'c:\Code\university_rag')
out = root / 'import_check.txt'
mods = ['sentence_transformers', 'faiss', 'transformers', 'torch', 'pypdf', 'streamlit', 'pandas']
lines = []
lines.append('PYTHON=' + sys.executable)
lines.append('VERSION=' + sys.version)
lines.append('CWD=' + os.getcwd())
for name in mods:
    try:
        importlib.import_module(name)
        lines.append(f'{name}=OK')
    except Exception as exc:
        lines.append(f'{name}=ERROR:{type(exc).__name__}:{exc}')
out.write_text('\n'.join(lines), encoding='utf-8')
print('WROTE=' + str(out))
