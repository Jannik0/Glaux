"""Chatbot pipeline lifecycle and run_chat/chat_stream generation.

Owns the model/context-usage "engine" state (``CHATBOT``, ``_CHAT_LOCK``,
``CHAT_GENERATION_STOP``) plus everything needed to drive HF ``pipeline()`` generation,
including the chat-template and direct (non-template) pipeline input paths.

``worker.context`` needs some of this state too (to measure token usage), but only
references it via local imports inside its own functions to avoid a circular import at
module-load time -- this module is safe to import ``worker.context`` at the top level.
``worker.asr`` (loaded after this module) imports this module directly at the top level.
"""

import gc
import os
import threading
import types
from collections.abc import Mapping
from contextlib import contextmanager

from transformers import TextIteratorStreamer, pipeline
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from . import context
from . import download as _download
from .download import _assert_valid_model_id, _load_progress_tqdm_hook, model_local_dir, read_model_pipeline_tag
from .thinking import (
    _iter_stripped_non_thinking_markup,
    _parse_tags_and_answer,
    _thinking_open_tag_from_generation_suffix,
)

# Generation stops at EOS; this only caps the upper bound (true "unlimited" is not supported).
CHAT_MAX_NEW_TOKENS_CEILING = 1_048_576

CHATBOT_MODEL_ID = None
CHATBOT_PIPELINE_TAG: str | None = None
CHATBOT = None
_CHAT_LOCK = threading.Lock()
# Bounds how long a new call waits to *acquire* the lock. Streaming itself is unbounded
# (governed by generation length, not this), but if a caller ever abandons a chat_stream()
# generator without draining or closing it, the lock would otherwise stay held until the
# generator is garbage collected -- silently hanging every other chat/context operation
# forever. Failing loudly after a generous wait is preferable to an indefinite, silent hang.
_CHAT_LOCK_ACQUIRE_TIMEOUT_SECONDS = 300
CHAT_GENERATION_STOP = False


def _force_cpu() -> bool:
    value = os.environ.get("GLAUX_FORCE_CPU", "").strip().lower()
    return value in ("1", "true", "yes")


def _pipeline_placement_kwargs() -> dict:
    """Pin the HF pipeline to CPU or GPU.

    CPU uses ``device="cpu"``, not ``device_map="cpu"``. Accelerate's device_map
    path still initializes CUDA on Windows CUDA wheels and can native-crash
    (Win32 0xC0000005 / exit 3221225477) when ``GLAUX_FORCE_CPU`` is set.
    """
    if _force_cpu():
        return {"device": "cpu"}
    try:
        import torch
    except ImportError:
        return {"device": "cpu"}
    if torch.cuda.is_available():
        return {"device_map": "cuda"}
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"device_map": "mps"}
    return {"device": "cpu"}


@contextmanager
def _chat_lock_guard():
    """Acquire ``_CHAT_LOCK`` with a bounded wait instead of blocking indefinitely."""
    if not _CHAT_LOCK.acquire(timeout=_CHAT_LOCK_ACQUIRE_TIMEOUT_SECONDS):
        raise RuntimeError(
            "Chat engine is busy (lock held longer than expected); refusing to block indefinitely."
        )
    try:
        yield
    finally:
        _CHAT_LOCK.release()


def _chat_template_backend():
    """Tokenizer for text-only templates; processor when the pipeline is multimodal."""
    if CHATBOT is None:
        return None
    processor = getattr(CHATBOT, "processor", None)
    if processor is not None:
        return processor
    return CHATBOT.tokenizer


def chatbot_has_chat_template() -> bool:
    """True when the loaded tokenizer/processor defines a usable chat template."""
    if CHATBOT is None:
        return False
    backend = _chat_template_backend()
    if backend is None or not hasattr(backend, "apply_chat_template"):
        return False
    template = getattr(backend, "chat_template", None)
    if isinstance(template, str):
        return bool(template.strip())
    if isinstance(template, dict):
        return bool(template)
    return bool(template)


DEFAULT_PIPELINE_TAG = "text-generation"


def chatbot_create(model_id: str):
    global CHATBOT, CHATBOT_MODEL_ID, CHATBOT_PIPELINE_TAG
    model_id = _assert_valid_model_id(model_id)
    CHATBOT_MODEL_ID = model_id
    model_path = str(model_local_dir(model_id))
    CHATBOT_PIPELINE_TAG = read_model_pipeline_tag(model_id) or DEFAULT_PIPELINE_TAG

    def _build_pipeline():
        return pipeline(
            CHATBOT_PIPELINE_TAG,
            model=model_path,
            trust_remote_code=False,
            **_pipeline_placement_kwargs(),
        )

    if _download._load_progress_callback is not None:
        with _load_progress_tqdm_hook():
            CHATBOT = _build_pipeline()
    else:
        CHATBOT = _build_pipeline()
    _configure_asr_generation_limits()

def _configure_asr_generation_limits() -> None:
    """Drop pipeline token caps so ASR runs until the utterance is fully decoded."""
    if CHATBOT is None or _active_pipeline_tag() != "automatic-speech-recognition":
        return
    generation_config = getattr(CHATBOT, "generation_config", None)
    if generation_config is not None:
        generation_config.max_new_tokens = None

def _release_torch_cache() -> None:
    """Return cached GPU/MPS allocations to the driver when torch is available."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _chatbot_destroy_unlocked():
    """Clear chatbot globals; caller must hold ``_CHAT_LOCK`` when used from chat paths."""
    global CHATBOT, CHATBOT_MODEL_ID, CHATBOT_PIPELINE_TAG
    chatbot = CHATBOT
    CHATBOT = None
    CHATBOT_MODEL_ID = None
    CHATBOT_PIPELINE_TAG = None
    context._invalidate_context_usage_cache()
    return chatbot


def chatbot_destroy():
    """Drop the in-process chat pipeline and encourage the OS to reclaim weight memory."""
    with _chat_lock_guard():
        chatbot = _chatbot_destroy_unlocked()
    if chatbot is not None:
        del chatbot
    gc.collect()
    _release_torch_cache()


_THINKING_TEMPLATE_PARAMS = ("enable_thinking",)


def _chat_template_thinking_kwarg_name() -> str | None:
    """Template kwarg that enables reasoning (first match in ``_THINKING_TEMPLATE_PARAMS``)."""
    if not chatbot_has_chat_template():
        return None
    template = getattr(_chat_template_backend(), "chat_template", None)
    if not isinstance(template, str):
        return None
    for name in _THINKING_TEMPLATE_PARAMS:
        if name in template:
            return name
    return None


def chatbot_supports_thinking() -> bool:
    """True when the loaded chat template accepts a thinking toggle kwarg."""
    return _chat_template_thinking_kwarg_name() is not None


def _thinking_template_attempts(base_kwargs: dict) -> list[dict]:
    param = _chat_template_thinking_kwarg_name()
    if param is None:
        return [base_kwargs]
    return [
        {**base_kwargs, param: True},
        {**base_kwargs, "tokenizer_encode_kwargs": {param: True}},
    ]


def _build_direct_pipeline_input(
    message: str,
    image_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
) -> str | list[str]:
    """Build native pipeline input when no chat template is available."""
    pipeline_tag = _active_pipeline_tag()
    audio_paths = context._normalize_media_paths(audio_paths) or []
    image_paths = context._normalize_media_paths(image_paths) or []
    video_paths = context._normalize_media_paths(video_paths) or []
    message = message.strip()

    if pipeline_tag == "automatic-speech-recognition":
        if not audio_paths:
            raise ValueError("Automatic speech recognition requires an audio attachment.")
        return audio_paths[0] if len(audio_paths) == 1 else audio_paths

    if pipeline_tag == "summarization":
        if not message:
            raise ValueError("Summarization requires text input.")
        return message

    if message:
        return message
    if audio_paths:
        return audio_paths[0] if len(audio_paths) == 1 else audio_paths
    if image_paths:
        return image_paths[0] if len(image_paths) == 1 else image_paths
    if video_paths:
        return video_paths[0] if len(video_paths) == 1 else video_paths
    raise ValueError("No input provided for pipeline.")


def _extract_pipeline_output_text(result) -> str:
    """Normalize pipeline return values to a single assistant text string."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "summary_text", "generated_text", "translation_text", "answer", "label"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    if isinstance(result, list) and result:
        return _extract_pipeline_output_text(result[0])
    raise ValueError("Could not extract text from pipeline output.")


_PIPELINES_WITH_NESTED_GENERATE_KWARGS = frozenset({"any-to-any", "image-text-to-text"})
_ASR_TYPES_WITHOUT_GENERATE_STREAM = frozenset({"ctc", "ctc_with_lm"})
_ASR_TYPE_TDT = "tdt"
_GENERATION_CONFIG_KWARG_KEYS = frozenset({
    "max_new_tokens",
    "max_length",
    "min_length",
    "min_new_tokens",
    "num_beams",
    "num_beam_groups",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "length_penalty",
    "return_dict_in_generate",
})
_ASR_GENERATION_CAP_KEYS = frozenset({"max_new_tokens", "max_length"})


def _active_pipeline_tag() -> str | None:
    if CHATBOT_PIPELINE_TAG:
        return CHATBOT_PIPELINE_TAG
    if not CHATBOT_MODEL_ID:
        return None
    return read_model_pipeline_tag(CHATBOT_MODEL_ID) or DEFAULT_PIPELINE_TAG


def _apply_chat_template_text(
    messages: list[dict], thinking: bool, *, add_generation_prompt: bool
) -> str | None:
    if CHATBOT is None:
        return None
    tokenizer = _chat_template_backend()
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return None
    base_kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    attempts: list[dict] = (
        _thinking_template_attempts(base_kwargs)
        if thinking and chatbot_supports_thinking()
        else [base_kwargs]
    )
    for kwargs in attempts:
        try:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
            if isinstance(rendered, str):
                return rendered
        except TypeError:
            continue
        except Exception:
            continue
    return None


def _generation_prompt_suffix(messages: list[dict], thinking: bool) -> str | None:
    """Text appended for the assistant turn (skipped by ``skip_prompt`` in the streamer)."""
    without = _apply_chat_template_text(
        messages, thinking, add_generation_prompt=False
    )
    with_prompt = _apply_chat_template_text(
        messages, thinking, add_generation_prompt=True
    )
    if not isinstance(with_prompt, str):
        return None
    if isinstance(without, str) and with_prompt.startswith(without):
        return with_prompt[len(without) :]
    return with_prompt


def _generation_starts_in_thinking(messages: list[dict], thinking: bool) -> bool:
    """True when streamed tokens should start in the thoughts panel.

    The generation-prompt suffix almost always begins with a template marker tag
    (e.g. ``<|im_start|>``). That tag is skipped by ``skip_prompt`` but must not
    affect parity. An odd number of further tags means the suffix ends with an
    opening marker whose body will arrive in the stream (not in the suffix).
    """
    suffix = _generation_prompt_suffix(messages, thinking)
    return _thinking_open_tag_from_generation_suffix(suffix) is not None


def chat_generation_starts_in_thinking(
    model_id: str,
    thinking: bool,
    message: str,
    image_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
    *,
    resubmit: bool = True,
    messages: list | None = None,
) -> bool:
    model_id = _assert_valid_model_id(model_id)
    with _chat_lock_guard():
        if CHATBOT_MODEL_ID != model_id or CHATBOT is None:
            chatbot_create(model_id)
        if not chatbot_has_chat_template():
            return False
        base = (
            context._normalize_context_messages(messages)
            if messages is not None
            else list(context.CONTEXT)
        )
        preview_context = context._context_for_inference(
            base
            + [
                context._preview_user_message(message, image_paths, audio_paths, video_paths),
            ],
            resubmit=resubmit,
        )
        preview_messages = context._filter_context_for_pipeline(
            preview_context,
            _active_pipeline_tag(),
        )
        return _generation_starts_in_thinking(preview_messages, thinking)


def _resolve_pipeline_call_params(pipeline_kwargs: dict) -> tuple[dict, dict, dict]:
    preprocess_params, forward_params, postprocess_params = CHATBOT._sanitize_parameters(
        **pipeline_kwargs
    )
    return (
        {**CHATBOT._preprocess_params, **preprocess_params},
        {**CHATBOT._forward_params, **forward_params},
        {**CHATBOT._postprocess_params, **postprocess_params},
    )


def _forward_params_have_max_new_tokens(forward_params: dict) -> bool:
    if forward_params.get("max_new_tokens") is not None:
        return True
    generate_kwargs = forward_params.get("generate_kwargs")
    return isinstance(generate_kwargs, dict) and generate_kwargs.get("max_new_tokens") is not None


def _inject_max_new_tokens(forward_params: dict, max_new_tokens: int) -> None:
    if _active_pipeline_tag() in _PIPELINES_WITH_NESTED_GENERATE_KWARGS:
        generate_kwargs = forward_params.setdefault("generate_kwargs", {})
        generate_kwargs["max_new_tokens"] = max_new_tokens
    else:
        forward_params["max_new_tokens"] = max_new_tokens


def _forward_generate_kwargs_container(forward_params: dict) -> dict:
    nested = forward_params.get("generate_kwargs")
    if isinstance(nested, dict):
        return nested
    return forward_params


def _streaming_generate_kwargs(forward_params: dict) -> dict:
    """Build flat ``model.generate`` kwargs without a ``generation_config`` object.

    Passing generation settings only via explicit kwargs avoids transformers warnings
    when models ship with both ``max_length`` and derived ``max_new_tokens``, and when
    multimodal pipeline ``_forward`` would add ``return_dict_in_generate`` on top of
    ``generation_config``.
    """
    container = dict(_forward_generate_kwargs_container(forward_params))
    generate_kwargs: dict = {}
    overrides: dict = {}
    asr_pipeline = _active_pipeline_tag() == "automatic-speech-recognition"
    for key in _GENERATION_CONFIG_KWARG_KEYS:
        if key in container:
            if asr_pipeline and key in _ASR_GENERATION_CAP_KEYS:
                container.pop(key)
                continue
            overrides[key] = container.pop(key)
    config = container.pop("generation_config", None)
    if config is not None:
        for key in _GENERATION_CONFIG_KWARG_KEYS:
            if key not in overrides:
                if asr_pipeline and key in _ASR_GENERATION_CAP_KEYS:
                    continue
                value = getattr(config, key, None)
                if value is not None:
                    overrides[key] = value
    generate_kwargs.update(container)
    for key, value in overrides.items():
        if value is not None:
            if asr_pipeline and key in _ASR_GENERATION_CAP_KEYS:
                continue
            generate_kwargs[key] = value
    if asr_pipeline:
        for key in _ASR_GENERATION_CAP_KEYS:
            generate_kwargs.pop(key, None)
    else:
        if generate_kwargs.get("max_new_tokens") is None and CHATBOT is not None:
            pipeline_max_new_tokens = getattr(CHATBOT.generation_config, "max_new_tokens", None)
            if pipeline_max_new_tokens is not None:
                generate_kwargs["max_new_tokens"] = pipeline_max_new_tokens
        if generate_kwargs.get("max_new_tokens") is not None:
            generate_kwargs["max_length"] = None
    if _active_pipeline_tag() in _PIPELINES_WITH_NESTED_GENERATE_KWARGS:
        generate_kwargs.setdefault("return_dict_in_generate", False)
    return generate_kwargs


def _generate_with_device_placement(inputs: Mapping, generate_kwargs: dict) -> None:
    """Run ``model.generate`` under the pipeline's device-placement/inference context.

    Shared by the direct-model and ``tdt`` streaming paths, which differ only in how they
    trim ``model_inputs`` down to what ``model.generate`` accepts.
    """
    with CHATBOT.device_placement():
        with CHATBOT.get_inference_context()():
            tensors = CHATBOT._ensure_tensor_on_device(dict(inputs), device=CHATBOT.device)
            CHATBOT.model.generate(**tensors, **generate_kwargs)


def _run_direct_model_generate(model_inputs: Mapping, generate_kwargs: dict) -> None:
    """Call ``model.generate`` for streaming without pipeline ``_forward`` duplication."""
    batch = dict(model_inputs)
    batch.pop("text", None)
    batch.pop("prompt_text", None)
    # ChunkPipeline ASR preprocess attaches bookkeeping keys the model does not consume.
    batch.pop("is_last", None)
    batch.pop("stride", None)
    batch.pop("num_frames", None)
    _generate_with_device_placement(batch, generate_kwargs)


def _iter_preprocess_batch_outputs(result):
    """Yield preprocess outputs ready for ``forward`` (dict-like batch or chunked batches)."""
    if isinstance(result, Mapping) and not isinstance(result, (str, bytes)):
        yield result
        return
    if isinstance(result, types.GeneratorType):
        yield from result
        return
    if isinstance(result, list):
        yield from result
        return
    raise TypeError(f"Unexpected preprocess return type: {type(result).__name__}")


def _iter_preprocess_batches(pipeline_input, preprocess_params: dict):
    """Yield preprocess outputs; ASR ChunkPipeline preprocess returns a generator."""
    result = CHATBOT.preprocess(pipeline_input, **preprocess_params)
    yield from _iter_preprocess_batch_outputs(result)


def _coerce_pipeline_chat_input(pipeline_input):
    """Match the chat wrapper each pipeline expects in ``preprocess``."""
    if not isinstance(pipeline_input, list) or not pipeline_input:
        return pipeline_input
    try:
        from transformers.utils.chat_template_utils import Chat as UtilsChat, is_valid_message
    except ImportError:
        return pipeline_input
    if not is_valid_message(pipeline_input[0]):
        return pipeline_input
    if isinstance(pipeline_input[0], (list, tuple)):
        if not pipeline_input[0] or not is_valid_message(pipeline_input[0][0]):
            return pipeline_input
        if _active_pipeline_tag() == "any-to-any":
            from transformers.pipelines import any_to_any

            return [any_to_any.Chat(chat) for chat in pipeline_input]
        return [UtilsChat(chat) for chat in pipeline_input]
    if _active_pipeline_tag() == "any-to-any":
        from transformers.pipelines import any_to_any

        return any_to_any.Chat(pipeline_input)
    return UtilsChat(pipeline_input)


def _run_pipeline_preprocess_and_forward(
    pipeline_input,
    pipeline_kwargs: dict,
    *,
    resubmit: bool = True,
    derive_max_new_tokens: bool = False,
) -> None:
    """Mirror ``Pipeline.run_single`` preprocess + forward without postprocess (streaming)."""
    pipeline_input = _coerce_pipeline_chat_input(pipeline_input)
    preprocess_params, forward_params, _postprocess_params = _resolve_pipeline_call_params(
        pipeline_kwargs
    )
    prepared_forward = False
    generate_kwargs: dict | None = None
    for model_inputs in _iter_preprocess_batches(pipeline_input, preprocess_params):
        if not prepared_forward:
            prompt_len = context._extract_prompt_token_count(model_inputs)
            context._set_context_usage_from_preprocess(model_inputs, resubmit=resubmit)
            if derive_max_new_tokens and not _forward_params_have_max_new_tokens(forward_params):
                _inject_max_new_tokens(
                    forward_params,
                    _max_new_tokens_for_prompt_len(prompt_len),
                )
            generate_kwargs = _streaming_generate_kwargs(forward_params)
            prepared_forward = True
        _run_direct_model_generate(model_inputs, generate_kwargs)


def _model_context_length() -> int | None:
    if CHATBOT is None:
        return None
    config = CHATBOT.model.config
    for attr in ("max_position_embeddings", "max_seq_len", "n_positions", "seq_length"):
        val = getattr(config, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attr in ("max_position_embeddings", "max_seq_len"):
            val = getattr(text_config, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    model_max = getattr(CHATBOT.tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max < 10**9:
        return model_max
    return None


def _max_new_tokens_for_prompt_len(prompt_len: int | None) -> int:
    context_len = _model_context_length()
    if context_len is not None and prompt_len is not None:
        return max(1, min(CHAT_MAX_NEW_TOKENS_CEILING, context_len - prompt_len))
    if context_len is not None:
        return min(CHAT_MAX_NEW_TOKENS_CEILING, context_len)
    return CHAT_MAX_NEW_TOKENS_CEILING


class _ChatStopCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return CHAT_GENERATION_STOP

def chat_stop() -> None:
    global CHAT_GENERATION_STOP
    CHAT_GENERATION_STOP = True


def _pipeline_asr_type() -> str | None:
    if _active_pipeline_tag() != "automatic-speech-recognition":
        return None
    return getattr(CHATBOT, "type", None)


def _direct_pipeline_supports_progressive_stream() -> bool:
    """True when the loaded direct pipeline can stream decoded tokens during generation."""
    if CHATBOT is None or CHATBOT.tokenizer is None:
        return False
    if not getattr(CHATBOT, "_pipeline_calls_generate", False):
        return False
    model = getattr(CHATBOT, "model", None)
    if model is not None and hasattr(model, "can_generate") and not model.can_generate():
        return False
    if _active_pipeline_tag() == "automatic-speech-recognition":
        if getattr(CHATBOT, "type", None) in _ASR_TYPES_WITHOUT_GENERATE_STREAM:
            return False
    return True


def _build_streaming_pipeline_kwargs(
    streamer: TextIteratorStreamer,
    stop_criteria: StoppingCriteriaList,
    thinking: bool,
    *,
    max_new_tokens: int | None = None,
) -> dict:
    """Build pipeline call kwargs for streaming generation.

    ``text-generation`` expects ``streamer`` / ``stopping_criteria`` at the top level;
    multimodal pipelines nest them under ``generate_kwargs``.

    When *max_new_tokens* is omitted, the pipeline keeps its configured generation limits
    (required for Whisper and other task-specific pipelines with small decoder windows).
    """
    generation_kwargs = {
        "streamer": streamer,
        "stopping_criteria": stop_criteria,
        "num_beams": 1,
    }
    pipeline_kwargs: dict = {}
    if max_new_tokens is not None:
        pipeline_kwargs["max_new_tokens"] = max_new_tokens
    pipeline_tag = _active_pipeline_tag()

    if pipeline_tag in _PIPELINES_WITH_NESTED_GENERATE_KWARGS:
        pipeline_kwargs["generate_kwargs"] = generation_kwargs
        if chatbot_supports_thinking() and thinking:
            param = _chat_template_thinking_kwarg_name()
            if param:
                pipeline_kwargs[param] = True
    else:
        pipeline_kwargs.update(generation_kwargs)
        if chatbot_supports_thinking() and thinking:
            param = _chat_template_thinking_kwarg_name()
            if param:
                pipeline_kwargs["tokenizer_encode_kwargs"] = {param: True}

    return pipeline_kwargs


def _iter_progressive_pipeline_stream(
    pipeline_input,
    *,
    thinking: bool,
    skip_special_tokens: bool,
    max_new_tokens: int | None = None,
    resubmit: bool = True,
    derive_max_new_tokens: bool = False,
):
    """Run CHATBOT on native pipeline input and yield decoded tokens as they are generated."""
    global CHAT_GENERATION_STOP
    CHAT_GENERATION_STOP = False
    stop_criteria = StoppingCriteriaList([_ChatStopCriteria()])
    streamer = TextIteratorStreamer(
        CHATBOT.tokenizer,
        skip_prompt=True,
        skip_special_tokens=skip_special_tokens,
    )
    thread_exc: list[Exception] = []

    pipeline_kwargs = _build_streaming_pipeline_kwargs(
        streamer,
        stop_criteria,
        thinking,
        max_new_tokens=max_new_tokens,
    )

    def _run_generate() -> None:
        try:
            _run_pipeline_preprocess_and_forward(
                pipeline_input,
                pipeline_kwargs,
                resubmit=resubmit,
                derive_max_new_tokens=derive_max_new_tokens,
            )
        except Exception as exc:
            thread_exc.append(exc)
        finally:
            streamer.end()

    gen_thread = threading.Thread(target=_run_generate)
    gen_thread.start()
    try:
        for chunk in streamer:
            yield chunk
    finally:
        gen_thread.join()
    if thread_exc:
        raise thread_exc[0]


def _iter_chat_template_stream(
    thinking: bool,
    image_paths: list[str] | None,
    audio_paths: list[str] | None,
    video_paths: list[str] | None,
    *,
    resubmit: bool = True,
):
    inference_context = context._context_for_inference(context.CONTEXT, resubmit=resubmit)
    pipeline_messages = context._filter_context_for_pipeline(
        inference_context, _active_pipeline_tag()
    )
    suffix = _generation_prompt_suffix(pipeline_messages, thinking)
    prefix = _thinking_open_tag_from_generation_suffix(suffix) or ""
    yield from _iter_stripped_non_thinking_markup(
        _iter_progressive_pipeline_stream(
            pipeline_messages,
            thinking=thinking,
            skip_special_tokens=False,
            resubmit=resubmit,
            derive_max_new_tokens=True,
        ),
        prefix=prefix,
    )


def _iter_direct_pipeline_batch(pipeline_input):
    """Return the full pipeline result in one chunk when token streaming is unavailable.

    Runs synchronously: unlike the progressive/TDT streamers, there is no producer/consumer
    pair here (nothing else runs concurrently while we wait), so a worker thread would only
    add overhead without any concurrency benefit.
    """
    result = CHATBOT(pipeline_input)
    yield _extract_pipeline_output_text(result)


def _iter_direct_pipeline_stream(
    message: str,
    image_paths: list[str] | None,
    audio_paths: list[str] | None,
    video_paths: list[str] | None,
):
    # Local import: worker.asr imports worker.chat at module scope, so importing it back
    # here at module scope would be circular. By the time this generator actually runs
    # (well after package init), worker.asr is fully loaded.
    from .asr import _iter_tdt_asr_stream

    pipeline_input = _build_direct_pipeline_input(
        message, image_paths=image_paths, audio_paths=audio_paths, video_paths=video_paths
    )
    if _pipeline_asr_type() == _ASR_TYPE_TDT:
        streamed_any = False
        for chunk in _iter_tdt_asr_stream(pipeline_input):
            streamed_any = True
            yield chunk
        if not streamed_any:
            yield from _iter_direct_pipeline_batch(pipeline_input)
        return
    if _direct_pipeline_supports_progressive_stream():
        yield from _iter_progressive_pipeline_stream(
            pipeline_input,
            thinking=False,
            skip_special_tokens=True,
        )
        return
    yield from _iter_direct_pipeline_batch(pipeline_input)


def chat_stream(
    model_id: str,
    thinking: bool,
    message: str,
    image_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
    *,
    resubmit: bool = True,
    messages: list | None = None,
):
    """Stream a chat response.

    When ``messages`` is provided (canonical history from contextManager, already
    including the new user turn), it becomes the working CONTEXT for this call and
    assistant text is *not* appended here — the Node contextManager owns commits.
    """
    global CHAT_GENERATION_STOP
    model_id = _assert_valid_model_id(model_id)
    owns_history = messages is None
    with _chat_lock_guard():
        if CHATBOT_MODEL_ID != model_id:
            old = _chatbot_destroy_unlocked()
        else:
            old = None
        if CHATBOT is None:
            chatbot_create(model_id)
        if messages is not None:
            context._validate_context_messages(messages)
            context.CONTEXT.clear()
            context.CONTEXT.extend(context._normalize_context_messages(messages))
        else:
            context._context_add_user(
                message,
                image_paths=context._normalize_media_paths(image_paths),
                audio_paths=context._normalize_media_paths(audio_paths),
                video_paths=context._normalize_media_paths(video_paths),
            )
        context_committed = False
        parts: list[str] = []
        used_chat_template = chatbot_has_chat_template()
        try:
            stream = (
                _iter_chat_template_stream(
                    thinking,
                    image_paths,
                    audio_paths,
                    video_paths,
                    resubmit=resubmit,
                )
                if used_chat_template
                else _iter_direct_pipeline_stream(message, image_paths, audio_paths, video_paths)
            )
            for chunk in stream:
                parts.append(chunk)
                yield chunk
            stopped = CHAT_GENERATION_STOP
            CHAT_GENERATION_STOP = False
            if not stopped:
                if used_chat_template:
                    _, answer = _parse_tags_and_answer("".join(parts))
                    if owns_history:
                        context._context_add_assistant(answer)
                    context._add_assistant_tokens_to_context_usage(answer)
                else:
                    response = "".join(parts)
                    if owns_history:
                        context._context_add_assistant(response)
                    context._add_assistant_tokens_to_context_usage(response)
                context_committed = True
        finally:
            if not context_committed:
                if owns_history:
                    context._context_rollback_last_user()
                context._invalidate_context_usage_cache()
            # Always release the outgoing model's memory, whether the stream
            # finished cleanly, was stopped, or raised partway through.
            if old is not None:
                del old
                gc.collect()
                _release_torch_cache()


def run_chat(
    model_id: str,
    thinking: bool,
    message: str,
    image_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
    *,
    resubmit: bool = True,
    messages: list | None = None,
) -> str:
    return "".join(
        chat_stream(
            model_id,
            thinking,
            message,
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_paths=video_paths,
            resubmit=resubmit,
            messages=messages,
        )
    )
