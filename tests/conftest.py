"""Default to the in-process tracker so tests do not need a ClearML Server."""

from __future__ import annotations

import os

os.environ.setdefault("WORKBENCH_TRACKER", "memory")
