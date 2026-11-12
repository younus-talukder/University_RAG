"""Find CO/CLO outcome links from layout column bands, then verify plain text.

Layout contributes coordinates only. Each accepted cell is located verbatim in
the same physical page's plain extraction before it becomes an atomic card.
"""
from __future__ import annotations
import re
from pathlib import Path
from .course_cards_tiered import compact, _new

HEADER = re.compile(r'corres\w*|\bPLOs?\b|\bPOs?\b', re.I)
BLOOM = re.compile(r'Bloom|taxonomy', re.I)
ROW_ID = re.compile(r'\b(CLO|CO)\s*(\d+)\b', re.I)
END = re.compile(r'\b(?:Part\s+[BCD]|Weighting\s+COs|Course\s+Content\s+Outline|14\s*\.\s*Course\s+Content|15\s*\.\s*Alignment)\b', re.I)
CELL = re.compile(r'\b(?:PLO|PO)\s*[-(]?\s*[a-z0-9]+\s*\)?|\b[a-z]\b|\b\d{1,2}\b', re.I)
LEVEL = re.compile(r'\b(?:\d+\s*/\s*(?:Apply|Analyze|Understand|Create|Evaluate|Remember|Valuing)|Cognitive\s*/)', re.I)

def _header(lines: list[str]):
    for i,line in enumerate(lines):
        if not HEADER.search(line):continue
        candidates=[(m.start(),m.group()) for m in HEADER.finditer(line) if m.group().lower().startswith('corres')]
        if not candidates:continue
        start=candidates[0][0]
        bloom=BLOOM.search(line,start+1)
        if bloom is None:
            for following in lines[i+1:i+4]:
                bloom=BLOOM.search(following,start+1)
                if bloom:break
        if bloom is None:continue
        end=bloom.start()
        head=' '.join(x[start:end].strip() for x in lines[i:i+7] if x[start:end].strip())
        family='PLO' if re.search(r'\bPLOs?\b',head,re.I) else 'PO' if re.search(r'\bPOs?\b',head,re.I) else None
        return i,start,end,head,family
    return None

def _row_anchors(lines: list[str],first: int,band_start: int):
    found=[]
    for i in range(first+1,len(lines)):
        line=lines[i]
        if END.search(line[:max(band_start,35)]):break
        left=line[:band_start]
        prefix=re.search(r'\b(CLO|CO)\b',left,re.I)
        if not prefix:continue
        number=re.search(r'\b(\d{1,2})\b',left[prefix.end():])
        if number is None:
            for following in lines[i+1:i+3]:
                number=re.search(r'\b(\d{1,2})\b',following[:band_start])
                if number:break
        if number:
            ident=f'{prefix.group().upper()} {int(number.group(1))}'
            if not found or found[-1][1]!=ident:found.append((i,ident))
    return found

def _plain_cell(raw: str,ident: str,layout_cell: str,used: set[int]):
    flat=compact(raw)
    family,number=ident.split()
    anchors=list(re.finditer(r'\b'+family+r'\s*'+number+r'\b',flat,re.I))
    for anchor in anchors:
        if anchor.start() in used:continue
        following=ROW_ID.search(flat,anchor.end())
        stop=following.start() if following else len(flat)
        end=END.search(flat,anchor.end(),stop)
        if end:stop=end.start()
        region=flat[anchor.end():stop]
        bloom=LEVEL.search(region)
        if bloom:region=region[:bloom.start()]
        cells=list(CELL.finditer(region))
        desired=re.sub(r'[^a-z0-9]','',layout_cell.casefold()).lstrip('0')
        matching=[]
        for cell in cells:
            value=cell.group().strip()
            canonical=re.sub(r'[^a-z0-9]','',value.casefold()).lstrip('0')
            if canonical==desired:matching.append((cell.start(),value))
        if matching:
            # In a statement row the outcome cell is the last matching token
            # before the Bloom column, not an earlier token in the statement.
            at,value=matching[-1]
            used.add(anchor.start())
            return value
    return None

def mapping_cards(pdf: str, pages: list[str], layouts: list[str]) -> tuple[list[dict],list[dict]]:
    cards=[];diagnostics=[];active=None
    for page,(raw,layout) in enumerate(zip(pages,layouts),1):
        lines=layout.splitlines()
        header=_header(lines)
        if header is None:
            if active is None or END.search(' '.join(lines[:8])):
                active=None
                diagnostics.append(dict(pdf=pdf,page=page,reason='header_not_found'))
                continue
            _,start,end,head,head_family=active
            first=-1
        else:
            active=header
            first,start,end,head,head_family=header
        anchors=_row_anchors(lines,first,start)
        used=set()
        if not anchors:
            diagnostics.append(dict(pdf=pdf,page=page,reason='band_error'))
            continue
        for i,(line_index,ident) in enumerate(anchors):
            row_end=anchors[i+1][0] if i+1<len(anchors) else len(lines)
            segments=[]
            for line in lines[line_index:row_end]:
                if END.search(line[:max(start,35)]):break
                part=line[start:end].strip()
                if part:segments.append(part)
            cell=' '.join(segments).strip()
            # Extra whitespace marks the next visual column when pypdf's
            # layout grid is one character wider than the header band.
            cell=re.split(r'\s{2,}',cell,maxsplit=1)[0].strip()
            first_cell=re.match(r'(?:(?:PLO|PO)\s*[-(]?\s*[a-z0-9]+\s*\)?|[a-z]|\d{1,2})(?:\s*,\s*(?:(?:PLO|PO)\s*[-(]?\s*[a-z0-9]+\s*\)?|[a-z]|\d{1,2}))*',cell,re.I)
            if first_cell:cell=first_cell.group().strip()
            if not cell and line_index>first+1:
                cell=lines[line_index-1][start:end].strip()
            cell=re.sub(r'\b(PLO|PO)\s+(\d+)\b',r'\1\2',cell,flags=re.I)
            cell=re.sub(r'\s*,\s*',', ',cell)
            if not cell or not re.search(r'(?:PLO|PO)\s*[-(]?\s*[A-Za-z0-9]|\b[a-z]\b|\b\d+\b',cell,re.I):
                diagnostics.append(dict(pdf=pdf,page=page,reason='cell_empty_in_pdf',row_key=ident))
                continue
            plain=_plain_cell(raw,ident,cell,used)
            if plain is None:
                diagnostics.append(dict(pdf=pdf,page=page,reason='band_error',row_key=ident,layout_cell=cell))
                continue
            family=head_family or ('PLO' if ident.startswith('CLO') else 'PO')
            display=plain if re.match(r'^(?:PLO|PO)',plain,re.I) else (
                f'{family} ({plain})' if re.fullmatch(r'[A-Za-z]',plain) else f'{family} {plain}')
            field='clo_plo_link' if ident.startswith('CLO') else 'co_po_link'
            extra={'plo_display':display} if family=='PLO' else {'po_display':display}
            cards.append(_new(pdf,page,field,plain,row_key=ident,po_cell_verbatim=plain,
                              header_text=head,column_hint=[start,end],**extra))
        if any(END.search(line) for line in lines[anchors[-1][0]+1:]):active=None
    return cards,diagnostics
