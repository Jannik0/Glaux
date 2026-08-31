"""Glaux: expose the shared CUDA 13 runtime to Python extension modules.

Windows Python 3.8+ does not search PATH for dependent DLLs of extensions.
add_dll_directory must run before torch (or any CUDA extension) is imported.
"""
import os
import sys
from pathlib import Path


def _glaux_cuda_dir():
    env = os.environ.get("GLAUX_CUDA_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    exe = Path(sys.executable).resolve()
    python_root = exe.parent
    if python_root.name.lower() == "bin":
        python_root = python_root.parent
    candidate = python_root.parent / "cuda"
    if candidate.is_dir():
        return str(candidate)
    return None


_cuda = _glaux_cuda_dir()
if _cuda and sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_cuda)
