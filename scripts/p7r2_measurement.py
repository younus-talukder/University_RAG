"""Shared P7 matcher: preserves literal NA values and applies Step 5A equivalences."""
from __future__ import annotations
import csv,re,unicodedata
from pathlib import Path
_DIGITS=str.maketrans('০১২৩৪৫৬৭৮৯','0123456789')
_ORDINALS={'first':'1st','second':'2nd','third':'3rd','fourth':'4th','fifth':'5th','sixth':'6th','seventh':'7th','eighth':'8th','ninth':'9th','tenth':'10th','eleventh':'11th','twelfth':'12th','thirteenth':'13th','fourteenth':'14th','fifteenth':'15th','sixteenth':'16th'}
def canonical(value):
 text=unicodedata.normalize('NFKC',str(value)).translate(_DIGITS)
 for word,number in _ORDINALS.items():text=re.sub(r'\b'+word+r'\b',number,text,flags=re.I)
 return text
def value_score(reference,candidate,mapping,measure):
 return measure.value_score(canonical(reference),canonical(candidate),mapping)
def csv_rows(path):
 # csv.DictReader does not interpret None, N/A, N/A. or Nil as missing values.
 return list(csv.DictReader(Path(path).open(encoding='utf-8',newline='')))

def load_5a_matcher(path):
    """Load only the read-only 5A matcher definitions; skip its report writer."""
    from types import SimpleNamespace
    source=Path(path).read_text(encoding='utf-8')
    definitions=source[:source.index('def measure(')]
    namespace={'__name__':'p7_read_only_5a_matcher'}
    exec(compile(definitions,str(path),'exec'),namespace)
    return SimpleNamespace(**namespace)
