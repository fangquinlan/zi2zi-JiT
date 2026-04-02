#!/usr/bin/env python3
"""Helpers for repo-local Python launchers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def maybe_reexec_with_repo_python(script_path: str | Path, repo_root: str | Path) -> None:
    script_path = Path(script_path).resolve()
    repo_root = Path(repo_root).resolve()

    if os.name == "nt":
        return

    venv_python = repo_root / ".venv-5090" / "bin" / "python"
    if not venv_python.is_file():
        return

    current_python = Path(sys.executable).resolve()
    if current_python == venv_python.resolve():
        return

    if os.environ.get("ZI2ZI_SKIP_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["ZI2ZI_SKIP_REEXEC"] = "1"
    cmd = [str(venv_python), str(script_path), *sys.argv[1:]]
    result = subprocess.run(cmd, check=False, cwd=repo_root, env=env)
    raise SystemExit(result.returncode)
