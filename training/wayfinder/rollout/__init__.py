"""Rollout harness: runs scout's explore loop for one task on a real checkout (see docs/teacher_test.md).

Nothing of scout is copied: it is imported from ../voyage-scout/src, or from $SCOUT_SRC (on Modal it is mounted
at /scout_src)."""

import os
import sys
from pathlib import Path

for _p in (os.environ.get("SCOUT_SRC"), str(Path(__file__).resolve().parents[3] / "voyage-scout" / "src")):
    if _p and Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)
