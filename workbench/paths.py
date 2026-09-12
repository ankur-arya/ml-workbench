"""Local data directory. Override with $WORKBENCH_HOME."""

from __future__ import annotations

import os
from pathlib import Path

_STORE = None


def workbench_home() -> Path:
    raw = os.environ.get("WORKBENCH_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / ".workbench").resolve()


def get_store():
    """Return a process-wide Store bound to the current WORKBENCH_HOME."""
    global _STORE
    from workbench.store import Store

    home = workbench_home()
    if _STORE is None or _STORE.home != home:
        _STORE = Store(home)
    return _STORE


def reset_store() -> None:
    global _STORE
    _STORE = None
