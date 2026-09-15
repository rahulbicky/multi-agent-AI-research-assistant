"""
run.py
======
Convenience launcher for the Multi-Agent AI Research Chatbot.

Equivalent to running:
    streamlit run frontend/app.py

Usage:
    python run.py [any extra streamlit args, e.g. --server.port=8502]
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli

APP_PATH = Path(__file__).resolve().parent / "frontend" / "app.py"


def main() -> None:
    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
