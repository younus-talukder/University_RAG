"""Opt-in Stage A repair switches. Every flag defaults to OFF.

Each flag is read from the environment at call time (``RAG_REPAIR_<NAME>``),
so an evaluation process can enable one repair without touching the others.
No flag relaxes a validation gate; each one only changes which passages the
unchanged gates are asked to judge, or how a passage's provenance is recorded.
"""

from __future__ import annotations

import os

FLAGS = (
    "FIX_COURSE_CODE_EXTRACTOR",        # R0
    "SAME_PAGE_LABELED_FIELD_JOIN",     # R5a
    "TABLE_HEADER_CONTEXT",             # R3
    "COURSE_ANCHORED_RETRIEVAL",        # R1
    "DEEPER_GATE_CHECKED_SELECTION",    # R2
    "SINGLE_COURSE_DOCUMENT_OWNERSHIP", # R5b
)


def repair_enabled(name: str) -> bool:
    if name not in FLAGS:
        raise KeyError(f"Unknown repair flag: {name}")
    raw = os.environ.get(f"RAG_REPAIR_{name}", "0").strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"RAG_REPAIR_{name} must be one of: true, false, 1, 0, yes, no, on, off.")


def enabled_repairs() -> tuple[str, ...]:
    return tuple(name for name in FLAGS if repair_enabled(name))


__all__ = ["FLAGS", "enabled_repairs", "repair_enabled"]
