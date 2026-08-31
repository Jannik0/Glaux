"""Aggregate Hugging Face snapshot download progress and cancellation (used by engine.py)."""

import os
import threading
from pathlib import Path
from typing import Callable

# hf_xet is not shipped. huggingface_hub's Xet backend has known stalls with no timeout
# (e.g. huggingface/xet-core#800); the classic HTTP/LFS path has mature timeout + resume.
#
# engine.js sets these same variables on the child process's environment before Python even
# starts, since huggingface_hub freezes them into module-level constants as soon as it's
# imported (transformers imports it well before this module ever runs). The setdefault calls
# here are just a fallback for running this module directly/standalone.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Default request timeout is 10s (HF_HUB_DOWNLOAD_TIMEOUT); bump it a bit so slow-but-alive
# chunks aren't mistaken for a stall, while still bounding each retry attempt.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
from tqdm.auto import tqdm as base_tqdm

DownloadProgressEvent = dict
DownloadProgressCallback = Callable[[DownloadProgressEvent], None]

_download_progress_callback: DownloadProgressCallback | None = None
_active_download_cancels: dict[str, threading.Event] = {}


class DownloadCancelledError(Exception):
    """Raised when the user stops an in-progress model download."""


def set_download_progress_callback(callback: DownloadProgressCallback | None) -> None:
    """Register a sink for aggregate download progress (used by the Node IPC layer)."""
    global _download_progress_callback
    _download_progress_callback = callback


def _emit_download_progress(event: DownloadProgressEvent) -> None:
    if _download_progress_callback is not None:
        _download_progress_callback(event)


def begin_download(model_id: str) -> None:
    """Register a cancel event for an in-flight download."""
    _active_download_cancels[model_id] = threading.Event()


def end_download(model_id: str) -> None:
    """Clear cancel state after download finishes or fails."""
    _active_download_cancels.pop(model_id, None)


def raise_if_download_cancelled(model_id: str) -> None:
    ev = _active_download_cancels.get(model_id)
    if ev is not None and ev.is_set():
        raise DownloadCancelledError("Download cancelled.")


def request_download_cancel(model_id: str) -> None:
    model_id = model_id.strip()
    ev = _active_download_cancels.get(model_id)
    if ev is not None:
        ev.set()


def _is_snapshot_bytes_aggregate_bar(desc: str | None, unit: str | None, unit_scale: bool) -> bool:
    """True for huggingface_hub's shared byte counter (total grows as files are discovered)."""
    if not desc or not str(desc).startswith("Downloading"):
        return False
    if unit_scale:
        return True
    return unit in (None, "", "B")


_devnull_fp = None


def _devnull_file():
    """Writable sink so tqdm does not early-exit when stdout is not a TTY."""
    global _devnull_fp
    if _devnull_fp is None or _devnull_fp.closed:
        _devnull_fp = open(os.devnull, "w", encoding="utf-8")
    return _devnull_fp


class _GlauxAggregateDownloadTqdm(base_tqdm):
    """Forwards snapshot_download's shared bytes bar (all files combined, approximate total)."""

    _progress_file: str = "download"

    def __init__(self, *args, **kwargs):
        # tqdm may call refresh() during super().__init__(); set this first.
        self._aggregate = False
        # huggingface_hub passes `name`; vanilla tqdm rejects it.
        kwargs.pop("name", None)
        bar_desc = kwargs.get("desc")
        bar_unit = kwargs.get("unit")
        bar_unit_scale = kwargs.get("unit_scale", False)
        # tqdm skips setting `desc`/`n` when disable=True; we need updates for IPC progress.
        kwargs["disable"] = False
        kwargs.setdefault("file", _devnull_file())
        super().__init__(*args, **kwargs)
        desc = getattr(self, "desc", bar_desc)
        unit = getattr(self, "unit", bar_unit)
        unit_scale = getattr(self, "unit_scale", bar_unit_scale)
        self._aggregate = _is_snapshot_bytes_aggregate_bar(desc, unit, unit_scale)
        if self._aggregate:
            _emit_download_progress({"status": "initiate", "file": self._progress_file})

    def _emit_aggregate_progress(self) -> None:
        if not getattr(self, "_aggregate", False):
            return
        raise_if_download_cancelled(self._progress_file)
        loaded = int(self.n)
        total = int(self.total) if self.total else 0
        _emit_download_progress(
            {
                "status": "progress",
                "file": self._progress_file,
                "loaded": loaded,
                "total": max(total, loaded),
            }
        )

    def refresh(self, *args, **kwargs):
        result = super().refresh(*args, **kwargs)
        self._emit_aggregate_progress()
        return result

    def update(self, n=1):
        result = super().update(n)
        self._emit_aggregate_progress()
        return result

    def close(self):
        if getattr(self, "_aggregate", False):
            _emit_download_progress({"status": "done", "file": self._progress_file})
        super().close()


def aggregate_tqdm_class_for(model_id: str) -> type:
    """``snapshot_download`` constructs tqdm itself; bind ``model_id`` via a per-call subclass."""
    safe_name = model_id.replace("/", "_").replace("\\", "_")
    return type(
        f"_GlauxAggregateDownloadTqdm_{safe_name}",
        (_GlauxAggregateDownloadTqdm,),
        {"_progress_file": model_id},
    )


def enable_download_progress_bars() -> None:
    enable_progress_bars()


# Each file downloads over a single HTTP connection on the classic (non-Xet) path, so
# fetching more files concurrently is the main lever left for aggregate throughput on
# multi-file repos (sharded checkpoints, tokenizer/config files, etc.).
DOWNLOAD_MAX_WORKERS = int(os.environ.get("GLAUX_DOWNLOAD_MAX_WORKERS", "16"))


def download_model(
    model_id: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
) -> str:
    """Download a Hub model into *local_dir* (flat layout).

    When *allow_patterns* is set (GGUF variant downloads), only matching files
    are fetched via huggingface_hub's snapshot_download filters.
    """
    model_id = model_id.strip()
    begin_download(model_id)
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        enable_download_progress_bars()
        kwargs: dict = {
            "repo_id": model_id,
            "repo_type": "model",
            "local_dir": str(local_dir),
            "tqdm_class": aggregate_tqdm_class_for(model_id),
            "max_workers": DOWNLOAD_MAX_WORKERS,
        }
        if allow_patterns:
            kwargs["allow_patterns"] = list(allow_patterns)
        try:
            snapshot_download(**kwargs)
        except DownloadCancelledError:
            raise
        except Exception:
            raise_if_download_cancelled(model_id)
            raise
        raise_if_download_cancelled(model_id)
        return str(local_dir)
    finally:
        end_download(model_id)


def list_model_files(model_id: str) -> list[dict]:
    """Return Hub file metadata ``[{path, size}, ...]`` for a model repo."""
    from huggingface_hub import HfApi

    model_id = model_id.strip()
    api = HfApi()
    info = api.repo_info(repo_id=model_id, repo_type="model", files_metadata=True)
    out: list[dict] = []
    for sibling in info.siblings or []:
        out.append(
            {
                "path": sibling.rfilename,
                "size": int(sibling.size or 0),
            }
        )
    return out
