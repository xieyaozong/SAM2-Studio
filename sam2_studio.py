from __future__ import annotations

from pathlib import Path

from utils.bootstrap import bootstrap_venv_python


bootstrap_venv_python(Path(__file__).resolve().parent)

from utils.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
