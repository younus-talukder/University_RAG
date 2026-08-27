from __future__ import annotations

import sys

from streamlit.web import cli


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "8511"
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        port,
    ]
    cli.main()


if __name__ == "__main__":
    main()
