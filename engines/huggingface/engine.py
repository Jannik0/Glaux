"""Glaux HF worker entry point.

This is the spawn target for the Electron bridge (see ``engines/huggingface/engine.js``):
it's loaded via ``importlib.util.spec_from_file_location`` (module name ``"engine"``) and
every RPC method the bridge calls is invoked as ``getattr(m, name)(**args)`` on this module.
The actual implementation lives in the ``worker`` package next to this file; this module is
just a thin re-export so it keeps working as that spawn target.

Add this file's own directory to ``sys.path`` explicitly so ``import worker`` works when the
process cwd is not this directory.
"""

import os
import sys
from pathlib import Path

def _hide_gpus_before_torch() -> None:
    """Hide CUDA/HIP before ``worker`` imports torch.

    Windows CUDA PyTorch wheels initialize the driver at import time. Combined with
    ``device_map="cpu"`` that can native-crash (Win32 0xC0000005 / exit 3221225477).
    Node may drop empty env vars on Windows, so this is also applied in-process.
    """
    value = os.environ.get("GLAUX_FORCE_CPU", "").strip().lower()
    if value not in ("1", "true", "yes"):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""


_hide_gpus_before_torch()

_ENGINE_DIR = str(Path(__file__).resolve().parent)
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from worker import *  # noqa: E402,F401,F403
