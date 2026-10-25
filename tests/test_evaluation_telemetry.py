from __future__ import annotations

import sys
from types import SimpleNamespace

from src.evaluation.runner import _telemetry


def test_telemetry_uses_proc_self_when_psutil_cannot_resolve_container_pid(monkeypatch):
    class BrokenProcess:
        def __init__(self):
            raise RuntimeError("PID is not visible in this namespace")

    fake_psutil = SimpleNamespace(
        Process=BrokenProcess,
        virtual_memory=lambda: SimpleNamespace(total=4096, available=2048),
        swap_memory=lambda: SimpleNamespace(used=0, total=0),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(
        "pathlib.Path.read_text",
        lambda *args, **kwargs: "Name:\tpython\nVmRSS:\t1234 kB\n",
    )

    result = _telemetry()

    assert result["rss_bytes"] == 1234 * 1024
    assert result["peak_process_bytes"] != "NOT_AVAILABLE"
    assert result["total_ram_bytes"] == 4096
    assert result["available_ram_bytes"] == 2048
