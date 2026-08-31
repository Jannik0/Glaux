"""torch/transformers media I/O compatibility shims.

Imported first by ``worker/__init__.py`` so the shims are applied before any other
worker module touches ``transformers`` pipelines. Decode goes through the vendored
ffmpeg/ffprobe CLIs (``GLAUX_FFMPEG`` / ``GLAUX_FFPROBE``), not PyAV.
"""

from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
import threading
import wave

import numpy as np


def _apply_transformers_compat_shims() -> None:
    """Restore helpers removed in transformers 5.x for trust_remote_code Hub models."""
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):

        def is_torch_fx_available() -> bool:
            return import_utils.is_torch_available()

        import_utils.is_torch_fx_available = is_torch_fx_available

    for name in ("is_torchdynamo_available", "is_torch_compile_available"):
        if not hasattr(import_utils, name):
            setattr(import_utils, name, import_utils.is_torch_available)


def _apply_torchcodec_disable_shim() -> None:
    """Glaux uses vendored ffmpeg for media I/O; ignore a broken or partial torchcodec install."""

    def is_torchcodec_available() -> bool:
        return False

    try:
        import transformers.utils.import_utils as import_utils
    except ImportError:
        return

    if getattr(import_utils, "_glaux_torchcodec_disabled", False):
        return

    import_utils.is_torchcodec_available = is_torchcodec_available
    import_utils._glaux_torchcodec_disabled = True

    try:
        import transformers.utils as utils

        utils.is_torchcodec_available = is_torchcodec_available
    except ImportError:
        pass

    for module_name in (
        "transformers.pipelines.automatic_speech_recognition",
        "transformers.pipelines.audio_classification",
        "transformers.video_processing_utils",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "is_torchcodec_available"):
            module.is_torchcodec_available = is_torchcodec_available


def _resolve_tool(env_key: str) -> str:
    explicit = os.environ.get(env_key, "").strip()
    if explicit:
        return explicit
    raise FileNotFoundError(f"{env_key} is not set. Run npm run build:ffmpeg.")


def _ffmpeg_bin() -> str:
    return _resolve_tool("GLAUX_FFMPEG")


def _ffprobe_bin() -> str:
    return _resolve_tool("GLAUX_FFPROBE")


def _is_mpegts_path(path: str) -> bool:
    ext = os.path.splitext(str(path or "").lower())[1]
    return ext in {".mts", ".m2ts", ".ts"}


def _probe_args_prefix(path: str) -> list[str]:
    if not _is_mpegts_path(path):
        return []
    return ["-probesize", "100M", "-analyzeduration", "100M"]


def _subprocess_kwargs(**extra):
    """Windows: hide the ffmpeg console. Never use ``-nostdin`` with ``-i pipe:0``."""
    kwargs = dict(extra)
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kwargs


def _stderr_text(stderr: bytes | str | None) -> str:
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace").strip()
    return str(stderr).strip()


def _as_media_path(value) -> str:
    if isinstance(value, dict):
        for key in ("path", "url", "video", "audio"):
            item = value.get(key)
            if item:
                return os.fspath(item)
    return os.fspath(value)


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        piece = stream.read(remaining)
        if not piece:
            break
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _s16le_to_f32(raw: bytes) -> np.ndarray:
    leftover = len(raw) % 2
    if leftover:
        raw = raw[: len(raw) - leftover]
    if not raw:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.float32(32768.0)


def _wav_bytes_to_f32(data: bytes) -> np.ndarray:
    """Decode a PCM WAV blob (what our vendored ffmpeg can mux) to mono float32."""
    if not data:
        return np.empty(0, dtype=np.float32)
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except Exception as exc:
        raise _malformed_audio_error(str(exc)) from exc
    if width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.float32(32768.0)
    elif width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / np.float32(2147483648.0)
    elif width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / np.float32(128.0)
    else:
        raise _malformed_audio_error(f"unsupported WAV sample width {width}")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm


def _consume_wav_header(stream) -> int:
    """Advance *stream* to PCM samples. Returns sample width in bytes (expect 2)."""
    riff = _read_exact(stream, 12)
    if len(riff) < 12 or riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
        raise _malformed_audio_error("ffmpeg did not produce a WAV stream")
    sample_width = 2
    while True:
        chunk = _read_exact(stream, 8)
        if len(chunk) < 8:
            raise _malformed_audio_error("truncated WAV header")
        cid = chunk[0:4]
        size = int.from_bytes(chunk[4:8], "little")
        if cid == b"fmt ":
            fmt = _read_exact(stream, size)
            if len(fmt) < 16:
                raise _malformed_audio_error("truncated WAV fmt chunk")
            sample_width = max(1, int.from_bytes(fmt[14:16], "little") // 8)
            if size % 2:
                _read_exact(stream, 1)
        elif cid == b"data":
            return sample_width
        else:
            skipped = _read_exact(stream, size)
            if len(skipped) < size:
                raise _malformed_audio_error("truncated WAV chunk")
            if size % 2:
                _read_exact(stream, 1)
    raise _malformed_audio_error("WAV data chunk not found")


def _malformed_audio_error(detail: str | None = None) -> ValueError:
    message = (
        "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
        "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
        "URL, ensure that the URL is the full address to **download** the audio file."
    )
    extra = (detail or "").strip()
    if extra:
        message = f"{message} ffmpeg: {extra}"
    return ValueError(message)


def _ffmpeg_read_audio_bytes(bpayload: bytes, sampling_rate: int) -> np.ndarray:
    """Decode arbitrary audio bytes to mono float32 PCM at *sampling_rate* via ffmpeg.

    stdin is the bitstream (``-i pipe:0``). Do not pass ``-nostdin``: that redirects
    stdin from NUL, so ffmpeg sees an empty file and this returns no samples.

    Output is WAV / pcm_s16le because the vendored ffmpeg does not ship the ``f32le``
    muxer that transformers' default ``ffmpeg_read`` uses.
    """
    cmd = [
        _ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-probesize",
        "10M",
        "-analyzeduration",
        "10M",
        "-i",
        "pipe:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            cmd,
            input=bpayload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            **_subprocess_kwargs(),
        )
    except FileNotFoundError as exc:
        raise ValueError("ffmpeg was not found but is required to load audio files") from exc
    audio = _wav_bytes_to_f32(completed.stdout or b"")
    if audio.size == 0:
        raise _malformed_audio_error(_stderr_text(completed.stderr))
    return audio


class FfmpegPcmReader:
    """Decode mono float32 PCM on demand from an on-disk audio file via ffmpeg.

    A long-running ffmpeg process writes WAV / pcm_s16le to stdout. Samples are
    converted to float32 and pulled lazily so Nemotron streaming can start after
    the first chunk.
    """

    def __init__(self, path: str, sampling_rate: int):
        self._offset = 0
        self._eof = False
        self._closed = False
        self._buffer = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._wav_ready = False
        self._s16_carry = b""
        cmd = [
            _ffmpeg_bin(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *_probe_args_prefix(path),
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sampling_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ]
        self._stderr = b""
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                **_subprocess_kwargs(),
            )
        except FileNotFoundError as exc:
            raise ValueError("ffmpeg was not found but is required to load audio files") from exc
        if self._proc.stdout is None:
            self.close()
            raise _malformed_audio_error()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            if self._proc.stderr is None:
                return
            self._stderr = self._proc.stderr.read() or b""
        except Exception:
            self._stderr = b""

    def _append(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        flat = arr.reshape(-1).astype(np.float32, copy=False)
        if self._buffer.size == 0:
            self._buffer = flat.copy()
        else:
            self._buffer = np.concatenate([self._buffer, flat])

    def _decode_more(self) -> bool:
        if self._eof or self._proc.stdout is None:
            return False
        if not self._wav_ready:
            try:
                width = _consume_wav_header(self._proc.stdout)
            except ValueError:
                self._eof = True
                raise
            if width != 2:
                self._eof = True
                raise _malformed_audio_error(f"expected 16-bit WAV, got sample width {width}")
            self._wav_ready = True
        raw = self._proc.stdout.read(16384)
        if not raw:
            self._eof = True
            return False
        raw = self._s16_carry + raw
        leftover = len(raw) % 2
        if leftover:
            self._s16_carry = raw[-leftover:]
            raw = raw[:-leftover]
        else:
            self._s16_carry = b""
        self._append(_s16le_to_f32(raw))
        return True

    def _ensure(self, absolute_end: int) -> None:
        while self._offset + self._buffer.size < absolute_end and not self._eof:
            if not self._decode_more():
                break

    def discard_before(self, absolute_index: int) -> None:
        with self._lock:
            if absolute_index <= self._offset:
                return
            drop = absolute_index - self._offset
            if drop >= self._buffer.size:
                self._buffer = np.empty(0, dtype=np.float32)
                self._offset = absolute_index
                return
            self._buffer = self._buffer[drop:].copy()
            self._offset = absolute_index

    def read(self, start: int, length: int) -> tuple[np.ndarray | None, bool]:
        if length <= 0:
            return None, True
        with self._lock:
            self._ensure(start + length)
            if start < self._offset:
                raise ValueError(
                    f"PCM read start {start} is before retained buffer offset {self._offset}."
                )
            local = start - self._offset
            available = self._buffer.size - local
            if available <= 0:
                return None, True
            if available >= length:
                return self._buffer[local : local + length].copy(), False
            if not self._eof:
                self._ensure(start + length)
                available = self._buffer.size - local
                if available >= length:
                    return self._buffer[local : local + length].copy(), False
            if available <= 0:
                return None, True
            return np.pad(self._buffer[local:], (0, length - available)), True

    def read_all(self) -> np.ndarray:
        with self._lock:
            while not self._eof:
                if not self._decode_more():
                    break
            if self._buffer.size == 0:
                raise _malformed_audio_error(_stderr_text(self._stderr))
            return self._buffer.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._eof = True
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        thread = getattr(self, "_stderr_thread", None)
        if thread is not None:
            try:
                thread.join(timeout=1)
            except Exception:
                pass


def _parse_rational(value: str) -> float:
    text = (value or "").strip()
    if not text or text.upper() == "N/A":
        return 0.0
    if "/" in text:
        num_s, den_s = text.split("/", 1)
        try:
            den = float(den_s)
            if den == 0:
                return 0.0
            return float(num_s) / den
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _probe_video(path: str):
    from transformers.video_utils import VideoMetadata

    cmd = [
        _ffprobe_bin(),
        "-v",
        "quiet",
        *_probe_args_prefix(path),
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
        "-select_streams",
        "v:0",
        "-of",
        "default=noprint_wrappers=1",
        path,
    ]
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            **_subprocess_kwargs(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe was not found but is required to load video files") from exc
    width = 0
    height = 0
    fps = 0.0
    avg_fps = 0.0
    n_frames = -1
    duration = -1.0
    for line in (completed.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if key == "width":
            width = int(val or 0)
        elif key == "height":
            height = int(val or 0)
        elif key == "r_frame_rate":
            fps = _parse_rational(val)
        elif key == "avg_frame_rate":
            avg_fps = _parse_rational(val)
        elif key == "nb_frames" and val.upper() != "N/A":
            try:
                n_frames = int(val)
            except ValueError:
                n_frames = -1
        elif key == "duration" and val.upper() != "N/A":
            try:
                duration = float(val)
            except ValueError:
                duration = -1.0
    if fps <= 0:
        fps = avg_fps
    if width <= 0 or height <= 0 or fps <= 0:
        detail = _stderr_text(completed.stderr)
        extra = f": {detail}" if detail else ""
        raise ValueError(f"Could not probe video metadata for {path}{extra}")
    if n_frames <= 0 and duration > 0:
        n_frames = int(duration * fps + 0.5)
    if n_frames <= 0:
        n_frames = 1
    if duration <= 0:
        duration = n_frames / fps
    return VideoMetadata(
        total_num_frames=int(n_frames),
        fps=float(fps),
        duration=float(duration),
        video_backend="ffmpeg",
        height=int(height),
        width=int(width),
    )


def _video_filters(path: str, width: int, height: int) -> str:
    parts = []
    if _is_mpegts_path(path):
        parts.append("yadif")
    parts.append(f"scale={int(width)}:{int(height)}:flags=fast_bilinear")
    return ",".join(parts)


def _read_video_ffmpeg(path: str, sample_indices_fn):
    path = _as_media_path(path)
    metadata = _probe_video(path)
    kwargs = {}
    indices = sample_indices_fn(metadata=metadata, **kwargs)
    if hasattr(indices, "detach"):
        indices = indices.detach().cpu().numpy()
    indices = np.asarray(indices).reshape(-1).astype(np.int64)
    if indices.size == 0:
        empty = np.zeros((0, metadata.height, metadata.width, 3), dtype=np.uint8)
        metadata.frames_indices = []
        return empty, metadata

    needed = {int(i) for i in indices.tolist() if int(i) >= 0}
    max_idx = max(needed) if needed else 0
    frame_size = int(metadata.width) * int(metadata.height) * 3
    cmd = [
        _ffmpeg_bin(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *_probe_args_prefix(path),
        "-i",
        path,
        "-an",
        "-vf",
        _video_filters(path, metadata.width, metadata.height),
        "-c:v",
        "rawvideo",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            **_subprocess_kwargs(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found but is required to load video files") from exc

    stderr_chunks: list[bytes] = []

    def _drain_stderr() -> None:
        try:
            if proc.stderr is not None:
                stderr_chunks.append(proc.stderr.read() or b"")
        except Exception:
            pass

    drain = threading.Thread(target=_drain_stderr, daemon=True)
    drain.start()
    kept: dict[int, np.ndarray] = {}
    decoded = 0
    try:
        while decoded <= max_idx:
            raw = _read_exact(proc.stdout, frame_size) if proc.stdout is not None else b""
            if len(raw) < frame_size:
                break
            if decoded in needed:
                kept[decoded] = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (int(metadata.height), int(metadata.width), 3)
                ).copy()
            decoded += 1
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        drain.join(timeout=2)

    ordered = [kept[int(i)] for i in indices.tolist() if int(i) in kept]
    if not ordered:
        detail = _stderr_text(b"".join(stderr_chunks))
        extra = f": {detail}" if detail else ""
        raise ValueError(f"ffmpeg produced no frames for {path}{extra}")
    video = np.stack(ordered, axis=0)
    metadata.frames_indices = [int(i) for i in indices.tolist() if int(i) in kept]
    metadata.total_num_frames = max(int(metadata.total_num_frames), decoded)
    return video, metadata


def _apply_video_decoder_compat_shims() -> None:
    """Route multimodal video decoding through vendored ffmpeg, never torchcodec/PyAV."""
    try:
        from transformers.video_processing_utils import BaseVideoProcessor
        from transformers.video_utils import default_sample_indices_fn
    except ImportError:
        return

    if getattr(BaseVideoProcessor, "_glaux_video_decoder_shim", False):
        return

    def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
        if isinstance(video_url_or_urls, list):
            return list(
                zip(*[self.fetch_videos(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls])
            )
        sampler = sample_indices_fn
        if sampler is None:
            sampler = default_sample_indices_fn
        return _read_video_ffmpeg(_as_media_path(video_url_or_urls), sampler)

    BaseVideoProcessor.fetch_videos = fetch_videos
    BaseVideoProcessor._glaux_video_decoder_shim = True


def _apply_audio_decoder_compat_shims() -> None:
    """Point transformers audio file loading at the vendored ffmpeg binary."""
    try:
        import transformers.pipelines.audio_utils as audio_utils
    except ImportError:
        return

    if getattr(audio_utils, "_glaux_audio_decoder_shim", False):
        return

    def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.ndarray:
        return _ffmpeg_read_audio_bytes(bpayload, sampling_rate)

    module_names = (
        "transformers.pipelines.audio_utils",
        "transformers.pipelines.automatic_speech_recognition",
        "transformers.pipelines.audio_classification",
        "transformers.pipelines.zero_shot_audio_classification",
    )
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
        if hasattr(module, "ffmpeg_read"):
            module.ffmpeg_read = ffmpeg_read

    audio_utils._glaux_audio_decoder_shim = True


_apply_transformers_compat_shims()
_apply_torchcodec_disable_shim()
_apply_video_decoder_compat_shims()
_apply_audio_decoder_compat_shims()
