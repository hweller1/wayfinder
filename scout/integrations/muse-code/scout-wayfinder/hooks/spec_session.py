#!/usr/bin/env python3
"""SessionStart hook of the retrieval spec: entry point for `spec_hooks.py session_start`.

Muse wants a distinct source file per hook, so each hook is a few lines that call the shared module.
"""

import importlib.util
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("spec_hooks", _here / "spec_hooks.py")
assert _spec is not None
assert _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["spec_hooks"] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    sys.exit(_mod.main(["spec_hooks", "session_start"]))
