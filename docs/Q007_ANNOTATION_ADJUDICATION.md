# Q007 development annotation audit

Scope: evaluation metadata only. No production RAG exception or answer rule is authorized.

The source PDF makes two inconsistent claims about UAP's initial bachelor programs:

- Physical PDF page 4: Computer Science & Engineering and Business Administration.
- Physical PDF page 8: Computer Science & Technology and Business Administration.

The active development dataset imported the page-4 Engineering reference in English, Bangla and Banglish, with `expected_page=4`. An earlier direct human choice selected page 4, and the user explicitly reconfirmed **“page 4 — Engineering”** for Step 10 on 2026-09-30. The Step 10 attachment's claim that page 8 Technology had been approved is superseded by this direct confirmation.

Adjudication version: `page4-engineering-confirmed-2026-09-30`. The dataset's Q007 row is therefore retained unchanged. Page 8 remains an acknowledged conflicting source passage, not a hidden typo. Any future reversal requires a new documented annotation version and dataset hash. Historical Step 9 result artifacts are preserved as originally measured.
