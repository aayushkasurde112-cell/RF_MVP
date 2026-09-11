"""PS-26147 src package: Phases 4-7 (demod .. end-to-end)."""
import sys as _sys, pathlib as _pl

# Phase 1-3 modules live flat at the project root; expose them so the src
# package works under any launch style (python main.py, uvicorn src.server:app).
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
