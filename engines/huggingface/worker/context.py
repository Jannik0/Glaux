"""Conversation context (history) state, normalization, and token-usage tracking.

Note: several functions here need the active chatbot pipeline (owned by ``worker.chat``)
to measure token usage. ``worker.chat`` imports this module at top level, so to avoid a
circular import at module-load time, any reference back into ``worker.chat`` is done via
a local import inside the function body (see ``worker/__init__.py`` load order: this
module loads before ``worker.chat``).

Import the chat *submodule* via importlib so callers always get ``worker.chat``
(the module), not a package-level name collision.
"""

import copy
import importlib
import re

from .thinking import _strip_thinking_from_messages

_EXTRACTED_TEXT_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Ephemeral working buffer for the current inference/usage call.
# Canonical conversation history lives in engines/contextManager.js.
CONTEXT: list[dict] = []
_CONTEXT_USAGE_CACHE: dict = {
    "used": None,
    "resubmit": True,
    "valid": False,
}


def _chat_module():
    """Return the ``worker.chat`` submodule."""
    return importlib.import_module('.chat', __package__)


def _sanitize_extracted_text(text: str) -> str:
    """Strip NUL bytes, other control characters, and unpaired UTF-16 surrogates that
    sometimes appear in user/PDF text. Such characters reach the Rust tokenizer as
    invalid input and raise ``TypeError: TextEncodeInput must be Union[...]`` instead of
    a helpful message, so every user-message text is run through this before being
    tokenized (keeps tab/newline/CR intact).
    """
    if not text:
        return text
    cleaned = _EXTRACTED_TEXT_CONTROL_CHARS_RE.sub("", text)
    return cleaned.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")


def _normalize_media_paths(paths: list[str] | str | None) -> list[str] | None:
    """Ensure media path args are a list of strings (not a bare path string to iterate char-wise)."""
    if paths is None:
        return None
    if isinstance(paths, str):
        return [paths] if paths else []
    return [path for path in paths if isinstance(path, str) and path]


def _normalize_user_content_parts(content: list) -> list[dict]:
    """Reorder stored user content for Gemma 4: image/video, then text, then audio."""
    images: list[dict] = []
    videos: list[dict] = []
    texts: list[dict] = []
    audios: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        path = part.get("path")
        if part_type == "image" and isinstance(path, str) and path:
            images.append({"type": "image", "path": path})
        elif part_type == "video" and isinstance(path, str) and path:
            videos.append({"type": "video", "path": path})
        elif part_type == "text":
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                texts.append({"type": "text", "text": text})
        elif part_type == "audio" and isinstance(path, str) and path:
            audios.append({"type": "audio", "path": path})
    return images + videos + texts + audios


_PIPELINE_CONTENT_TYPES: dict[str, frozenset[str]] = {
    "text-generation": frozenset({"text"}),
    "summarization": frozenset({"text"}),
    "image-text-to-text": frozenset({"text", "image"}),
    "automatic-speech-recognition": frozenset({"audio"}),
    "any-to-any": frozenset({"text", "image", "audio", "video"}),
}
_DEFAULT_PIPELINE_CONTENT_TYPES = frozenset({"text"})
_TEXT_ONLY_CONTENT_TYPES = frozenset({"text"})


def _pipeline_allowed_content_types(pipeline_tag: str | None) -> frozenset[str]:
    if pipeline_tag and pipeline_tag in _PIPELINE_CONTENT_TYPES:
        return _PIPELINE_CONTENT_TYPES[pipeline_tag]
    return _DEFAULT_PIPELINE_CONTENT_TYPES


def _filter_message_content(
    content,
    allowed_types: frozenset[str],
    *,
    role: str,
) -> list[dict]:
    """Return content parts the active pipeline can consume (empty list if none)."""
    if isinstance(content, str):
        text = content.strip()
        if role == "assistant":
            return [{"type": "text", "text": content}] if text else []
        if "text" in allowed_types and text:
            return [{"type": "text", "text": content}]
        return []

    if not isinstance(content, list):
        return []

    filtered: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            if role != "assistant" and "text" not in allowed_types:
                continue
            text = part.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            filtered.append({"type": "text", "text": text})
        elif part_type in ("image", "audio", "video") and part_type in allowed_types:
            path = part.get("path")
            if isinstance(path, str) and path:
                filtered.append({"type": part_type, "path": path})
    return filtered


def _normalize_pipeline_message_content(
    content: list[dict],
    allowed_types: frozenset[str],
) -> str | list[dict]:
    """Shape message content for the target pipeline (e.g. plain string for text-only turns)."""
    if (
        allowed_types == _TEXT_ONLY_CONTENT_TYPES
        and len(content) == 1
        and content[0].get("type") == "text"
    ):
        return content[0]["text"]
    return content


_MEDIA_INFERENCE_PART_TYPES = frozenset({"image", "audio", "video"})


def _strip_previous_media_from_context(messages: list[dict]) -> list[dict]:
    """Drop image/audio/video parts from every message except the last (inference-only).

    Avoids deep-copying the whole (potentially large) context: messages are only read,
    never mutated in place, and a replacement dict/content-list is built solely for the
    messages that actually contain media to strip. The untouched majority (including the
    last message, which always passes through unmodified) keeps sharing its original
    reference.
    """
    if not messages:
        return []
    result = list(messages)
    for i, msg in enumerate(result[:-1]):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        filtered = [
            part
            for part in content
            if not (
                isinstance(part, dict)
                and part.get("type") in _MEDIA_INFERENCE_PART_TYPES
            )
        ]
        if len(filtered) != len(content):
            result[i] = {**msg, "content": filtered}
    return result


def _context_for_inference(messages: list[dict], *, resubmit: bool) -> list[dict]:
    # Persisted history may include <think>…</think> for the UI; models must not see it.
    stripped = _strip_thinking_from_messages(messages)
    if resubmit:
        return stripped
    return _strip_previous_media_from_context(stripped)


def _filter_context_for_pipeline(
    messages: list[dict],
    pipeline_tag: str | None,
) -> list[dict]:
    """Project stored context to modalities the pipeline can handle (does not mutate ``messages``)."""
    allowed = _pipeline_allowed_content_types(pipeline_tag)
    filtered_messages: list[dict] = []
    # No deep copy needed: every part below is read-only (``.get()``) and the output is
    # built entirely from fresh dict/list literals, so nothing here can alias back into
    # (or mutate) the caller's ``messages``.
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _filter_message_content(msg.get("content"), allowed, role=role)
        if not content:
            continue
        filtered_messages.append(
            {
                "role": role,
                "content": _normalize_pipeline_message_content(content, allowed),
            }
        )
    return filtered_messages


def _build_user_message_content(
    message: str,
    image_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
) -> list[dict]:
    """Build multimodal user content in an order Gemma 4 expects: image/video, then text, then audio."""
    parts: list[dict] = []
    for image_path in _normalize_media_paths(image_paths) or []:
        parts.append({"type": "image", "path": image_path})
    for video_path in _normalize_media_paths(video_paths) or []:
        parts.append({"type": "video", "path": video_path})
    text = _sanitize_extracted_text(message) if isinstance(message, str) else ""
    if text.strip():
        parts.append({"type": "text", "text": text})
    for audio_path in _normalize_media_paths(audio_paths) or []:
        parts.append({"type": "audio", "path": audio_path})
    return parts


def _preview_user_message(
    message: str,
    image_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
) -> dict:
    return {
        "role": "user",
        "content": _build_user_message_content(message, image_paths, audio_paths, video_paths),
    }


def _invalidate_context_usage_cache() -> None:
    _CONTEXT_USAGE_CACHE["valid"] = False
    _CONTEXT_USAGE_CACHE["used"] = None
    _CONTEXT_USAGE_CACHE["resubmit"] = True


def _set_context_usage_from_preprocess(model_inputs: dict, *, resubmit: bool) -> None:
    used = _extract_prompt_token_count(model_inputs)
    _CONTEXT_USAGE_CACHE["used"] = used
    _CONTEXT_USAGE_CACHE["resubmit"] = resubmit
    _CONTEXT_USAGE_CACHE["valid"] = used is not None


def _add_assistant_tokens_to_context_usage(response: str) -> None:
    if not _CONTEXT_USAGE_CACHE["valid"] or not response:
        return
    used = _CONTEXT_USAGE_CACHE["used"]
    if used is None:
        return
    tokens = _count_text_tokens(response)
    if tokens > 0:
        _CONTEXT_USAGE_CACHE["used"] = used + tokens


def _count_text_tokens(text: str) -> int:
    chat = _chat_module()

    if chat.CHATBOT is None or not text:
        return 0
    tokenizer = chat.CHATBOT.tokenizer
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


def _extract_prompt_token_count(model_inputs) -> int | None:
    if model_inputs is None or not hasattr(model_inputs, "get"):
        return None
    for key in ("input_ids", "decoder_input_ids"):
        tensor = model_inputs.get(key)
        if tensor is None or not hasattr(tensor, "shape"):
            continue
        if len(tensor.shape) == 0:
            continue
        return int(tensor.shape[-1])
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None and hasattr(attention_mask, "sum"):
        try:
            return int(attention_mask.sum().item())
        except Exception:
            pass
    for key in ("input_features", "input_values"):
        tensor = model_inputs.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 2:
            return int(tensor.shape[-1])
    return None


def _context_add_user(
    message: str,
    image_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
) -> None:
    CONTEXT.append(
        {
            "role": "user",
            "content": _build_user_message_content(message, image_paths, audio_paths, video_paths),
        }
    )


def _context_add_assistant(response: str) -> None:
    CONTEXT.append({"role": "assistant", "content": [{"type": "text", "text": response}]})


def _context_rollback_last_user() -> None:
    if CONTEXT and CONTEXT[-1].get("role") == "user":
        CONTEXT.pop()


def context_clear():
    CONTEXT.clear()
    _invalidate_context_usage_cache()


def _validate_context_messages(messages: list) -> None:
    if not isinstance(messages, list):
        raise ValueError("Context messages must be a list.")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Context message at index {i} must be an object.")
        role = msg.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"Context message at index {i} has invalid role.")


def context_snapshot() -> list[dict]:
    return copy.deepcopy(CONTEXT)


def _pipeline_messages_for_usage(
    *, resubmit: bool, messages: list | None = None
) -> list[dict]:
    chat = _chat_module()

    source = (
        _normalize_context_messages(messages)
        if messages is not None
        else CONTEXT
    )
    if not source:
        return []
    inference_context = _context_for_inference(source, resubmit=resubmit)
    return _filter_context_for_pipeline(inference_context, chat._active_pipeline_tag())


def _preprocess_params_for_context_usage(pipeline_messages: list[dict]) -> dict:
    chat = _chat_module()

    params = dict(chat.CHATBOT._preprocess_params)
    if pipeline_messages and pipeline_messages[-1].get("role") == "assistant":
        params["continue_final_message"] = True
    else:
        params["continue_final_message"] = False
    return params


def _context_usage_result(*, used: int | None, resubmit: bool, stale: bool = False) -> dict:
    chat = _chat_module()

    total = chat._model_context_length()
    safe_used = 0 if used is None else used
    return {
        "used": safe_used,
        "total": total,
        "stale": stale,
        "supported": chat.chatbot_has_chat_template(),
    }


def _refresh_context_usage(
    *, resubmit: bool = True, messages: list | None = None
) -> dict:
    """Run pipeline preprocess only to measure the current context (no generation)."""
    chat = _chat_module()

    if chat.CHATBOT is None:
        _invalidate_context_usage_cache()
        return _context_usage_result(used=0, resubmit=resubmit)

    pipeline_messages = _pipeline_messages_for_usage(
        resubmit=resubmit, messages=messages
    )
    if not pipeline_messages or not chat.chatbot_has_chat_template():
        _CONTEXT_USAGE_CACHE["used"] = 0
        _CONTEXT_USAGE_CACHE["resubmit"] = resubmit
        _CONTEXT_USAGE_CACHE["valid"] = True
        return _context_usage_result(used=0, resubmit=resubmit)

    try:
        pipeline_input = chat._coerce_pipeline_chat_input(pipeline_messages)
        preprocess_params = _preprocess_params_for_context_usage(pipeline_messages)
        model_inputs = chat.CHATBOT.preprocess(pipeline_input, **preprocess_params)
        _set_context_usage_from_preprocess(model_inputs, resubmit=resubmit)
        used = _CONTEXT_USAGE_CACHE["used"]
        if used is None:
            _CONTEXT_USAGE_CACHE["used"] = 0
            _CONTEXT_USAGE_CACHE["valid"] = True
            used = 0
        return _context_usage_result(used=used, resubmit=resubmit)
    except Exception:
        _CONTEXT_USAGE_CACHE["used"] = 0
        _CONTEXT_USAGE_CACHE["resubmit"] = resubmit
        _CONTEXT_USAGE_CACHE["valid"] = True
        return _context_usage_result(used=0, resubmit=resubmit)


def context_usage(
    *,
    resubmit: bool = True,
    refresh: bool = False,
    messages: list | None = None,
) -> dict:
    """Return context usage, optionally refreshing via preprocess when requested."""
    chat = _chat_module()

    if refresh:
        with chat._chat_lock_guard():
            return _refresh_context_usage(resubmit=resubmit, messages=messages)
    # Best-effort locked read: a background generation thread mutates
    # ``_CONTEXT_USAGE_CACHE`` while holding ``_CHAT_LOCK``, so reading its keys
    # unlocked could observe a torn snapshot. But this is also polled live during
    # streaming (which can run far longer than a UI poll should ever block for), so
    # don't wait for the lock -- take it opportunistically and fall back to a direct
    # (possibly torn-by-one-field) read when a generation is actively holding it.
    acquired = chat._CHAT_LOCK.acquire(timeout=0)
    try:
        if not _CONTEXT_USAGE_CACHE["valid"]:
            return _context_usage_result(used=0, resubmit=resubmit)
        if _CONTEXT_USAGE_CACHE["resubmit"] != resubmit:
            return _context_usage_result(used=0, resubmit=resubmit, stale=True)
        return _context_usage_result(used=_CONTEXT_USAGE_CACHE["used"], resubmit=resubmit)
    finally:
        if acquired:
            chat._CHAT_LOCK.release()


def _normalize_context_messages(messages: list) -> list[dict]:
    normalized = copy.deepcopy(messages)
    for msg in normalized:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = _normalize_user_content_parts(content)
    return normalized


def context_replace(messages: list) -> None:
    chat = _chat_module()

    _validate_context_messages(messages)
    with chat._chat_lock_guard():
        CONTEXT.clear()
        CONTEXT.extend(_normalize_context_messages(messages))
        _invalidate_context_usage_cache()
