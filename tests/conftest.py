"""Protect the frozen Stage A index during the test suite.

Tests that publish an index must pass a temporary target directory.  This
guard also checks the four real index files byte for byte after the suite.
"""
from __future__ import annotations

import builtins
import hashlib
import io
import os
from pathlib import Path

import pytest


REAL_INDEX = Path(__file__).resolve().parents[1] / "vector_db"
INDEX_FILES = ("index.faiss", "index_manifest.json", "metadata.pkl", "sparse_index.pkl")
_originals: dict[str, object] = {}
_before: dict[str, str] = {}


def _protected(path: object) -> bool:
    try:
        candidate = Path(path).resolve()
    except (TypeError, ValueError):
        return False
    return candidate == REAL_INDEX or REAL_INDEX in candidate.parents


def _check(path: object) -> None:
    if _protected(path):
        raise AssertionError(f"tests must write indexes to a temporary directory: {path}")


def _hashes() -> dict[str, str]:
    return {name: hashlib.sha256((REAL_INDEX / name).read_bytes()).hexdigest()
            for name in INDEX_FILES}


def pytest_sessionstart(session: pytest.Session) -> None:
    global _before
    _before = _hashes()
    _originals.update(open=builtins.open, io_open=io.open,
                      os_open=os.open, replace=os.replace, rename=os.rename)

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            _check(file)
        return _originals["open"](file, mode, *args, **kwargs)

    def guarded_io_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            _check(file)
        return _originals["io_open"](file, mode, *args, **kwargs)

    def guarded_os_open(file, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            _check(file)
        return _originals["os_open"](file, flags, *args, **kwargs)

    def guarded_replace(src, dst, *args, **kwargs):
        _check(dst)
        return _originals["replace"](src, dst, *args, **kwargs)

    def guarded_rename(src, dst, *args, **kwargs):
        _check(dst)
        return _originals["rename"](src, dst, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    os.replace = guarded_replace
    os.rename = guarded_rename


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    for name, original in _originals.items():
        if name == "open": builtins.open = original
        elif name == "io_open": io.open = original
        elif name == "os_open": os.open = original
        elif name == "replace": os.replace = original
        elif name == "rename": os.rename = original
    after = _hashes()
    assert after == _before, "frozen vector_db hashes changed during the test suite"
