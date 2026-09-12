from __future__ import annotations

from pathlib import Path

import pytest

from workbench.paths import reset_store
from workbench.store import Store


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Store:
    monkeypatch.setenv("WORKBENCH_HOME", str(tmp_path))
    reset_store()
    yield Store(tmp_path)
    reset_store()
