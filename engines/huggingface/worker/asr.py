"""Streaming ASR paths: offline TDT (Parakeet) and Nemotron cache-aware streaming.

Loaded after ``worker.chat`` (which owns the chatbot/model state), so this module can
import it at the top level without creating a circular import.
"""

import threading

from transformers import TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteriaList

# Use importlib so we get the real submodule (named ``chat``) unambiguously.
import importlib

chat = importlib.import_module('.chat', __package__)
from .chat import (  # noqa: E402
    _ASR_GENERATION_CAP_KEYS,
    _ChatStopCriteria,
    _generate_with_device_placement,
    _iter_preprocess_batch_outputs,
    _streaming_generate_kwargs,
)
from .compat import FfmpegPcmReader, _malformed_audio_error
from .context import _CONTEXT_USAGE_CACHE, _set_context_usage_from_preprocess
from .download import model_local_dir

_NEMOTRON_CACHE_AWARE_ASR_MODEL_TYPES = frozenset({
    "nemotron_asr_streaming",
    "nemotron3_5_asr",
})


def _tdt_model_type() -> str | None:
    if chat.CHATBOT is None:
        return None
    model = getattr(chat.CHATBOT, "model", None)
    if model is None:
        return None
    return getattr(model.config, "model_type", None)


def _supports_nemotron_cache_aware_asr_stream() -> bool:
    model_type = _tdt_model_type()
    return model_type in _NEMOTRON_CACHE_AWARE_ASR_MODEL_TYPES


def _normalize_tdt_audio_path(pipeline_input) -> str:
    if isinstance(pipeline_input, list):
        if not pipeline_input:
            raise ValueError("Automatic speech recognition requires an audio attachment.")
        pipeline_input = pipeline_input[0]
    if not isinstance(pipeline_input, str) or not pipeline_input.strip():
        raise ValueError("Automatic speech recognition requires an audio attachment.")
    return pipeline_input


def _nemotron_streaming_processor():
    """Return the Nemotron streaming processor (cached on the active pipeline)."""
    processor = getattr(chat.CHATBOT, "_glaux_nemotron_processor", None)
    if processor is not None:
        return processor
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_local_dir(chat.CHATBOT_MODEL_ID)))
    processor.set_num_lookahead_tokens(processor.default_num_lookahead_tokens)
    chat.CHATBOT._glaux_nemotron_processor = processor
    return processor


def _nemotron_processor_call(processor, audio, *, sampling_rate: int, is_first_audio_chunk: bool):
    kwargs = {
        "sampling_rate": sampling_rate,
        "is_streaming": True,
        "is_first_audio_chunk": is_first_audio_chunk,
        "return_tensors": "pt",
    }
    if _tdt_model_type() == "nemotron3_5_asr":
        kwargs["language"] = "auto"
    return processor(audio, **kwargs)


def _nemotron_tensor_device_dtype():
    model = chat.CHATBOT.model
    dtype = getattr(model, "dtype", None)
    if dtype is None:
        dtype = next(model.parameters()).dtype
    return model.device, dtype


def _nemotron_move_batch(batch, device, dtype):
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            if getattr(value, "is_floating_point", lambda: False)():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _nemotron_decode_stream_text(sequences) -> str:
    processor = _nemotron_streaming_processor()
    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    if hasattr(sequences, "detach"):
        sequences = sequences[0]
    return processor.decode(
        sequences,
        skip_special_tokens=True,
        group_tokens=False,
    )


def _iter_asr_preprocess_batches(pipeline_input):
    preprocess_params = dict(chat.CHATBOT._preprocess_params)
    batches = chat.CHATBOT.preprocess(pipeline_input, **preprocess_params)
    yield from _iter_preprocess_batch_outputs(batches)


def _build_tdt_generate_inputs(model_inputs: dict) -> dict:
    """Mirror the Hugging Face ASR pipeline ``tdt`` forward input shaping."""
    model_inputs = dict(model_inputs)
    model_inputs.pop("is_last", None)
    model_inputs.pop("stride", None)
    model_inputs.pop("num_frames", None)
    main_input_name = chat.CHATBOT.model.main_input_name
    inputs = {main_input_name: model_inputs.pop(main_input_name)}
    if "attention_mask" in model_inputs:
        inputs["attention_mask"] = model_inputs.pop("attention_mask")
    return inputs


def _run_tdt_generate(model_inputs: dict, **generate_kwargs) -> None:
    _generate_with_device_placement(_build_tdt_generate_inputs(model_inputs), generate_kwargs)


def _iter_tdt_offline_asr_stream(pipeline_input):
    """Offline full-utterance TDT ASR with token streaming via ``TextIteratorStreamer``."""
    chat.CHAT_GENERATION_STOP = False
    stop_criteria = StoppingCriteriaList([_ChatStopCriteria()])
    streamer = TextIteratorStreamer(
        chat.CHATBOT.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        group_tokens=False,
    )
    thread_exc: list[Exception] = []
    generate_kwargs = _streaming_generate_kwargs({
        "streamer": streamer,
        "stopping_criteria": stop_criteria,
        "num_beams": 1,
    })

    def _run() -> None:
        try:
            for model_inputs in _iter_asr_preprocess_batches(pipeline_input):
                if not _CONTEXT_USAGE_CACHE["valid"]:
                    _set_context_usage_from_preprocess(model_inputs, resubmit=True)
                _run_tdt_generate(model_inputs, **generate_kwargs)
        except Exception as exc:
            thread_exc.append(exc)
        finally:
            streamer.end()

    worker = threading.Thread(target=_run)
    worker.start()
    try:
        for chunk in streamer:
            yield chunk
    finally:
        worker.join()
    if thread_exc:
        raise thread_exc[0]


def _iter_nemotron_cache_aware_asr_stream(audio_path: str):
    """Progressive file transcription for Nemotron cache-aware streaming ASR models."""
    chat.CHAT_GENERATION_STOP = False

    processor = _nemotron_streaming_processor()
    sampling_rate = processor.feature_extractor.sampling_rate
    pcm = FfmpegPcmReader(audio_path, sampling_rate)

    try:
        device, dtype = _nemotron_tensor_device_dtype()
        first_len = processor.num_samples_first_audio_chunk
        first_pcm, _ = pcm.read(0, first_len)
        if first_pcm is None:
            raise _malformed_audio_error()

        first_chunk_inputs = _nemotron_processor_call(
            processor,
            first_pcm,
            sampling_rate=sampling_rate,
            is_first_audio_chunk=True,
        )
        first_chunk_inputs = _nemotron_move_batch(first_chunk_inputs, device, dtype)
        if not _CONTEXT_USAGE_CACHE["valid"]:
            _set_context_usage_from_preprocess(first_chunk_inputs, resubmit=True)

        def input_features_generator():
            yield first_chunk_inputs["input_features"][:, : processor.num_mel_frames_first_audio_chunk, :]

            mel_frame_idx = processor.num_mel_frames_first_audio_chunk
            hop_length = processor.feature_extractor.hop_length
            n_fft = processor.feature_extractor.n_fft
            start_idx = mel_frame_idx * hop_length - n_fft // 2
            required_samples = processor.num_samples_per_audio_chunk

            while True:
                pcm.discard_before(max(0, start_idx))
                window, is_last = pcm.read(start_idx, required_samples)
                if window is None:
                    break
                chunk_inputs = _nemotron_processor_call(
                    processor,
                    window,
                    sampling_rate=sampling_rate,
                    is_first_audio_chunk=False,
                )
                chunk_inputs = _nemotron_move_batch(chunk_inputs, device, dtype)
                yield chunk_inputs["input_features"]
                if is_last:
                    break

                mel_frame_idx += processor.num_mel_frames_per_audio_chunk
                start_idx = mel_frame_idx * hop_length - n_fft // 2

        stop_criteria = StoppingCriteriaList([_ChatStopCriteria()])
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_special_tokens=True,
            group_tokens=False,
        )
        thread_exc: list[Exception] = []
        generate_result: list = []
        generate_kwargs = {
            key: value
            for key, value in first_chunk_inputs.items()
            if key not in ("input_features", "num_lookahead_tokens")
        }
        generate_kwargs.update({
            "input_features": input_features_generator(),
            "streamer": streamer,
            "stopping_criteria": stop_criteria,
            "num_beams": 1,
            "num_lookahead_tokens": processor.default_num_lookahead_tokens,
        })
        for key in _ASR_GENERATION_CAP_KEYS:
            generate_kwargs.pop(key, None)
        if "prompt_ids" in generate_kwargs:
            generate_kwargs["prompt_ids"] = generate_kwargs["prompt_ids"].long()

        def _run() -> None:
            try:
                with chat.CHATBOT.device_placement():
                    with chat.CHATBOT.get_inference_context()():
                        generate_result.append(
                            chat.CHATBOT.model.generate(**generate_kwargs)
                        )
            except Exception as exc:
                thread_exc.append(exc)
            finally:
                streamer.end()
                pcm.close()

        worker = threading.Thread(target=_run)
        worker.start()
        streamed_parts: list[str] = []
        try:
            for chunk in streamer:
                streamed_parts.append(chunk)
                yield chunk
        finally:
            worker.join()
            pcm.close()
        if thread_exc:
            raise thread_exc[0]

        streamed_text = "".join(streamed_parts).strip()
        if streamed_text:
            return

        if generate_result:
            decoded = _nemotron_decode_stream_text(generate_result[0]).strip()
            if decoded:
                yield decoded
                return

        yield from _iter_tdt_offline_asr_stream(audio_path)
    except Exception:
        pcm.close()
        raise


def _iter_tdt_asr_stream(pipeline_input):
    """Stream Parakeet/Nemotron transducer ASR models.

    Nemotron cache-aware checkpoints feed mel chunks incrementally; other ``tdt`` models
    preprocess the full utterance first. The Hugging Face ASR pipeline ``tdt`` forward path
    calls ``model.generate`` without forwarding ``generate_kwargs``, so the generic pipeline
    streamer integration stays empty for the offline path.
    """
    if _supports_nemotron_cache_aware_asr_stream():
        yield from _iter_nemotron_cache_aware_asr_stream(_normalize_tdt_audio_path(pipeline_input))
        return

    yield from _iter_tdt_offline_asr_stream(pipeline_input)
