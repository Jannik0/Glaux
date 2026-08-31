"""Glaux Hugging Face worker package.

Re-exports every RPC callable the Electron bridge (``engines/huggingface/engine.js``)
invokes by name on the spawned Python process, plus a few stable helpers kept for
backward compatibility with anything importing ``engine`` directly.

Import order matters here and mirrors the internal dependency graph:

1. ``compat``   -- applies torch/transformers/ffmpeg shims (must run before any pipeline use).
2. ``download`` -- model cache/download + load-progress callbacks (no worker deps).
3. ``thinking`` -- ``<think>`` tag parsing (no worker deps).
4. ``context``  -- conversation history + usage cache (depends on ``thinking``;
   reaches into ``chat`` lazily via local imports to avoid a circular import).
5. ``chat``     -- chatbot lifecycle + run_chat/chat_stream generation (depends on all of the
   above at module scope; reaches into ``asr`` lazily for the ASR streaming branch).
6. ``asr``      -- streaming ASR paths (depends on ``chat`` at module scope; safe since
   ``chat`` is already fully loaded by this point).
"""

from . import compat  # noqa: F401  (import-time side effect: applies compat shims)
from .download import (
    DownloadCancelledError,
    download_model,
    list_model_files,
    model_local_dir,
    models_cache_dir,
    read_model_pipeline_tag,
    request_download_cancel,
    set_download_progress_callback,
    set_load_progress_callback,
)
from .context import (
    context_clear,
    context_replace,
    context_snapshot,
    context_usage,
)
# Import the chat *module* first, then bind RPC callables. Prefer ``from .chat import …``
# (or importlib) when sibling modules need the submodule itself.
from . import chat as _chat_module  # noqa: F401
from .chat import (
    chat_generation_starts_in_thinking,
    chat_stop,
    chat_stream,
    chatbot_create,
    chatbot_destroy,
    chatbot_has_chat_template,
    chatbot_supports_thinking,
    run_chat,
)
from . import asr  # noqa: F401  (registers ASR streaming paths used by chat.chat_stream)

__all__ = [
    "DownloadCancelledError",
    "download_model",
    "list_model_files",
    "model_local_dir",
    "models_cache_dir",
    "read_model_pipeline_tag",
    "request_download_cancel",
    "set_download_progress_callback",
    "set_load_progress_callback",
    "context_clear",
    "context_replace",
    "context_snapshot",
    "context_usage",
    "run_chat",
    "chat_generation_starts_in_thinking",
    "chat_stop",
    "chat_stream",
    "chatbot_create",
    "chatbot_destroy",
    "chatbot_has_chat_template",
    "chatbot_supports_thinking",
]
