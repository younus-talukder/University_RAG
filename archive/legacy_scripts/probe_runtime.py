import os, sys
from pathlib import Path

root = Path(__file__).resolve().parent
out_path = root / 'runtime_probe.txt'
lines = []
lines.append('PYTHON=' + sys.executable)
lines.append('VERSION=' + sys.version)
lines.append('CWD=' + os.getcwd())
lines.append('ROOT_EXISTS=' + str(root.exists()))
for name in ['sentence_transformers', 'faiss', 'transformers', 'torch', 'pypdf']:
    try:
        __import__(name)
        lines.append(name + '=OK')
    except Exception as e:
        lines.append(name + '=ERROR=' + type(e).__name__ + ':' + str(e))
out_path.write_text('\n'.join(lines), encoding='utf-8')
print('WROTE=' + str(out_path))
