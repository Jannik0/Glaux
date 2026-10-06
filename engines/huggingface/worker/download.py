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
_FRONT_MATTER_RE = re.compile(r"^---\r?\n([\s\S]*?)\r?\n---")
_PIPELINE_TAG_RE = re.compile(r"^pipeline_tag:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
_TAGS_FLOW_RE = re.compile(r"^tags:[ \t]*\[([^\]]*)\]", re.MULTILINE)
_TAGS_BLOCK_RE = re.compile(r"^tags:[ \t]*\r?\n((?:[ \t]*-[^\n]*(?:\r?\n|$))*)", re.MULTILINE)
_TAGS_SCALAR_RE = re.compile(r"^tags:[ \t]*(\S.*?)[ \t]*$", re.MULTILINE)
_TAG_ITEM_RE = re.compile(r"^[ \t]*-[ \t]*(.*?)\s*$")
_TASK_TAGS = ("automatic-speech-recognition", "text-to-image", "image-to-image")
MODEL_CACHE_DIR: Path | None = None


def _assert_valid_model_id(model_id: str) -> str:
    model_id = model_id.strip()
    if not model_id or len(model_id) > 512 or not _HF_REPO_ID_RE.fullmatch(model_id):
        raise ValueError("Invalid model id (expected namespace/repo-name).")
    return model_id


def set_download_progress_callback(callback, progress_id=None):
    model_downloader.set_download_progress_callback(callback, progress_id)


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


def _unquote(value: str) -> str:
    return value.strip().strip("'\"").strip()


def _task_tag(tags) -> str | None:
    """First ASR, text-to-image, or image-to-image tag, in card order."""
    if not tags:
        return None
    for tag in tags:
        if not isinstance(tag, str):
            continue
        text = _unquote(tag)
        if text in _TASK_TAGS:
            return text
    return None


def _card_tags(card: str) -> list[str]:
    flow = _TAGS_FLOW_RE.search(card)
    if flow:
        return [text for text in (_unquote(part) for part in flow.group(1).split(",")) if text]
    block = _TAGS_BLOCK_RE.search(card)
    if block and block.group(1).strip():
        tags: list[str] = []
        for line in block.group(1).splitlines():
            item = _TAG_ITEM_RE.match(line)
            if not item:
                continue
            text = _unquote(item.group(1))
            if text:
                tags.append(text)
        return tags
    scalar = _TAGS_SCALAR_RE.search(card)
    if not scalar or scalar.group(1).startswith(("|", ">")):
        return []
    text = _unquote(scalar.group(1))
    return [text] if text else []


def pipeline_tag_from_card(content: str) -> str | None:
    """``pipeline_tag``, or an ASR / diffusion task from ``tags`` when it is absent."""
    front = _FRONT_MATTER_RE.search(content or "")
    card = front.group(1) if front else (content or "")
    declared = _PIPELINE_TAG_RE.search(card)
    if declared:
        value = _unquote(declared.group(1))
        if value:
            return value
    return _task_tag(_card_tags(card))


def read_model_pipeline_tag(model_id: str) -> str | None:
    """Task id from a cached model's README.md model card."""
    readme_path = model_local_dir(model_id) / "README.md"
    try:
        content = readme_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return pipeline_tag_from_card(content)


_BASE_MODEL_ID_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


def _base_model_id(value) -> str | None:
    """First ``org/name`` in a model-card ``base_model`` scalar or list."""
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _base_model_id(item)
            if found:
                return found
        return None
    if not isinstance(value, str):
        return None
    text = value.strip().strip("'\"")
    return text if _BASE_MODEL_ID_RE.fullmatch(text) else None


def read_hub_model_card(model_id: str) -> dict:
    """``pipeline_tag`` and ``base_model`` from the Hub model card, before any files are cached."""
    from huggingface_hub import HfApi

    model_id = _assert_valid_model_id(model_id)
    info = HfApi().model_info(repo_id=model_id)
    card = info.card_data
    raw = None
    if card is not None:
        raw = card.get("base_model") if isinstance(card, dict) else getattr(card, "base_model", None)
    tag = _unquote(info.pipeline_tag) if isinstance(info.pipeline_tag, str) else ""
    if not tag:
        tag = _task_tag(getattr(info, "tags", None)) or ""
    if not tag and card is not None:
        card_tags = card.get("tags") if isinstance(card, dict) else getattr(card, "tags", None)
        tag = _task_tag(card_tags) or ""
    return {
        "pipeline_tag": tag or None,
        "base_model": _base_model_id(raw),
    }


def download_model(
    model_id: str,
    allow_patterns: list[str] | None = None,
    gguf_variant: str | None = None,
    progress_id: str | None = None,
) -> str:
    model_id = _assert_valid_model_id(model_id)
    local_dir = model_local_dir(model_id)
    tracked = _assert_valid_model_id(progress_id) if progress_id else model_id
    result = model_downloader.download_model(
        model_id,
        local_dir,
        allow_patterns=allow_patterns,
        progress_id=tracked,
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
