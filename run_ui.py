# Run the RVA Streamlit console.
#   python run_ui.py        (uses the project .venv if present)
#
# Equivalent to:  streamlit run app/streamlit_app.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "streamlit_app.py"

venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
python = str(venv_python) if venv_python.exists() else sys.executable

# Any extra args are forwarded to `streamlit run`, e.g.:
#   python run_ui.py --server.address 0.0.0.0   # share on the local network
#   python run_ui.py --server.port 8600
cmd = [python, "-m", "streamlit", "run", str(APP), *sys.argv[1:]]
raise SystemExit(subprocess.call(cmd))
