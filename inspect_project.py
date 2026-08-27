import csv
from pathlib import Path
from pypdf import PdfReader

root = Path(__file__).resolve().parent
report_lines = []
report_lines.append(f'ROOT={root}')
report_lines.append(f'CSV_FILE={root / "data/questions/questions.csv"}')
with open(root / 'data/questions/questions.csv', 'r', encoding='utf-8-sig', newline='') as f:
    rows = list(csv.reader(f))
report_lines.append('COLUMNS=' + str(rows[0]))
for row in rows[1:4]:
    report_lines.append('ROW=' + str(row))
report_lines.append('PDF_FILES=' + str(sorted([p.name for p in (root / 'data/documents').glob('*.pdf')])))
for p in sorted((root / 'data/documents').glob('*.pdf')):
    report_lines.append(f'FILE={p.name}')
    reader = PdfReader(str(p))
    report_lines.append(f'PAGE_COUNT={len(reader.pages)}')
    samples = []
    for i, page in enumerate(reader.pages[:3], start=1):
        text = ' '.join((page.extract_text() or '').split())
        if text:
            samples.append((i, text[:200]))
    report_lines.append('SAMPLES=' + str(samples[:3]))
out = root / 'inspection_report.txt'
out.write_text('\n'.join(report_lines), encoding='utf-8')
print(f'WROTE_REPORT={out}')
