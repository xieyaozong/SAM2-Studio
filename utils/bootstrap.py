from __future__ import annotations

from pathlib import Path

import os
import subprocess
import sys


def same_python(left: Path, right: Path) -> bool:
    try:
        return str(left.resolve()).casefold() == str(right.resolve()).casefold()
    except OSError:
        return str(left).casefold() == str(right).casefold()


def bootstrap_venv_python(app_dir: Path) -> None:
    if (
        getattr(sys, "frozen", False)
        or os.environ.get("SAM2_USE_CURRENT_PYTHON") == "1"
        or os.environ.get("SAM2_VENV_BOOTSTRAPPED") == "1"
    ):
        return

    venv_python = (
        app_dir / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else app_dir / ".venv" / "bin" / "python"
    )
    if not venv_python.exists() or same_python(Path(sys.executable), venv_python):
        return

    os.environ["SAM2_VENV_BOOTSTRAPPED"] = "1"
    command = [str(venv_python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    if os.name == "nt":
        raise SystemExit(subprocess.call(command))
    os.execv(str(venv_python), command)
