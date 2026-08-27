from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
import traceback

ERRORS: list[tuple[str, BaseException]] = []


def check_import(module_name: str) -> None:
    print(f"\n[{module_name}]")
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if version is None:
            distribution_name = "faiss-cpu" if module_name == "faiss" else module_name
            version = metadata.version(distribution_name)
        print(f"OK: {version}")
    except Exception as exc:
        ERRORS.append((module_name, exc))
        print(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def print_guidance() -> None:
    torch_dll_errors = [
        exc
        for module_name, exc in ERRORS
        if module_name in {"torch", "sentence_transformers"} and "c10.dll" in str(exc)
    ]
    if not torch_dll_errors:
        return

    print("\nTroubleshooting:")
    print("- PyTorch failed while loading c10.dll, so sentence-transformers cannot start.")
    print("- On Windows this is usually an environment/dependency issue, not a RAG pipeline code error.")
    print("- Reinstall the CPU PyTorch wheel in this virtual environment, then rerun this diagnostic.")
    print("  Example:")
    print("  python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")

    for package_name in ["torch", "transformers", "sentence_transformers", "faiss", "streamlit", "pypdf"]:
        check_import(package_name)
    print_guidance()


if __name__ == "__main__":
    main()
