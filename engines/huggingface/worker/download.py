"""Model download/cache management and load/download progress plumbing."""

import importlib.util
import json
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

_MODEL_DOWNLOADER_PATH = Path(__file__).resolve().parent.parent / "model-downloader.py"
_spec = importlib.util.spec_from_file_location("glaux_model_downloader", _MODEL_DOWNLOADER_PATH)
model_downloader = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(model_downloader)

DownloadCancelledError = model_downloader.DownloadCancelledError

_HF_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_PIPELINE_TAG_RE = re.compile(r"^pipeline_tag:\s*(.+)\s*$", re.MULTILINE)
MODEL_CACHE_DIR: Path | None = None


def _assert_valid_model_id(model_id: str) -> str:
    model_id = model_id.strip()
    if not model_id or len(model_id) > 512 or not _HF_REPO_ID_RE.fullmatch(model_id):
        raise ValueError("Invalid model id (expected namespace/repo-name).")
    return model_id


def set_download_progress_callback(callback):
    model_downloader.set_download_progress_callback(callback)


def request_download_cancel(model_id: str) -> None:
    model_downloader.request_download_cancel(_assert_valid_model_id(model_id))


LoadProgressCallback = Callable[[dict], None] | None
_load_progress_callback: LoadProgressCallback = None


def set_load_progress_callback(callback: LoadProgressCallback) -> None:
    """Register a sink for in-memory weight-load progress (Node IPC layer)."""
    global _load_progress_callback
    _load_progress_callback = callback


def _emit_load_progress(event: dict) -> None:
    if _load_progress_callback is not None:
        _load_progress_callback(event)


def _make_load_progress_tqdm_class():
    """tqdm subclass that forwards transformers' "Loading weights" bar over IPC."""
    from tqdm.auto import tqdm as base_tqdm

    class _GlauxLoadProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            self._track_weights = False
            self._last_percent = -1
            bar_desc = kwargs.get("desc")
            # Keep IPC updates even when the console bar would be disabled.
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            desc = getattr(self, "desc", bar_desc) or ""
            self._track_weights = "Loading weights" in str(desc)
            # disable=True skips initializing counters; restore them for percent math.
            self.n = 0
            if self._track_weights:
                self._emit_percent()

        def _emit_percent(self) -> None:
            if not self._track_weights:
                return
            loaded = int(self.n)
            total = int(self.total) if self.total else 0
            percent = int(100 * loaded / total) if total > 0 else 0
            if percent == self._last_percent:
                return
            self._last_percent = percent
            _emit_load_progress(
                {
                    "status": "progress",
                    "loaded": loaded,
                    "total": max(total, loaded),
                    "percent": percent,
                }
            )

        def update(self, n=1):
            if n is None:
                n = 1
            self.n += n
            self._emit_percent()

        def __iter__(self):
            if not self._track_weights:
                yield from super().__iter__()
                return
            for item in self.iterable:
                self.n += 1
                self._emit_percent()
                yield item

        def close(self):
            if self._track_weights and self.total:
                self.n = int(self.total)
                self._emit_percent()
            super().close()

    return _GlauxLoadProgressTqdm


@contextmanager
def _load_progress_tqdm_hook():
    """Route transformers' logging.tqdm through our load-progress bar during pipeline()."""
    from transformers.utils import logging as transformers_logging

    tqdm_class = _make_load_progress_tqdm_class()

    def _hook(_factory, args, kwargs):
        return tqdm_class(*args, **kwargs)

    previous = transformers_logging.set_tqdm_hook(_hook)
    try:
        yield
    finally:
        transformers_logging.set_tqdm_hook(previous)


def models_cache_dir(path: str | None = None) -> Path:
    """Models cache root for flat Hub ids (`namespace/repo` folders).

    Pass *path* to configure (Node RPC). On read, uses the configured value or
    ``GLAUX_MODELS_CACHE_DIR`` from the environment.
    """
    global MODEL_CACHE_DIR
    if path is not None:
        MODEL_CACHE_DIR = Path(path).resolve()
        return MODEL_CACHE_DIR
    if MODEL_CACHE_DIR is None:
        env = os.environ.get("GLAUX_MODELS_CACHE_DIR")
        if env:
            MODEL_CACHE_DIR = Path(env).resolve()
    if MODEL_CACHE_DIR is None:
        raise RuntimeError("Models cache directory is not configured.")
    return MODEL_CACHE_DIR


def model_local_dir(model_id: str) -> Path:
    """Absolute path to a flat cached model: `{cache_root}/{namespace}/{repo}`."""
    model_id = _assert_valid_model_id(model_id)
    root = models_cache_dir()
    local = root.joinpath(*model_id.split("/")).resolve()
    try:
        local.relative_to(root)
    except ValueError as exc:
        raise ValueError("Invalid model folder path.") from exc
    return local


def read_model_pipeline_tag(model_id: str) -> str | None:
    """Read ``pipeline_tag`` from a cached model's README.md YAML frontmatter."""
    readme_path = model_local_dir(model_id) / "README.md"
    try:
        content = readme_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _PIPELINE_TAG_RE.search(content)
    if not match:
        return None
    value = match.group(1).strip()
    return value.strip("'\"")


def download_model(
    model_id: str,
    allow_patterns: list[str] | None = None,
    gguf_variant: str | None = None,
) -> str:
    model_id = _assert_valid_model_id(model_id)
    local_dir = model_local_dir(model_id)
    result = model_downloader.download_model(
        model_id,
        local_dir,
        allow_patterns=allow_patterns,
    )
    if gguf_variant:
        selection = {
            "variant": str(gguf_variant),
            "files": list(allow_patterns or []),
        }
        selection_path = local_dir / ".glaux-gguf-selection.json"
        selection_path.write_text(
            json.dumps(selection, indent=2),
            encoding="utf-8",
        )
    return result


def list_model_files(model_id: str) -> list[dict]:
    model_id = _assert_valid_model_id(model_id)
    return model_downloader.list_model_files(model_id)
