"""Diffusers text-to-image and image-to-image loading and one-shot generation.

``worker.chat`` branches here for those pipeline tags. Inpaint and
unconditional pipelines are rejected. An image-to-image load may use an
img2img class. Callers pass only the current prompt and, for image-to-image,
an optional init image. Size, steps, seed, and CFG stay at the pipeline
defaults.

Stop uses the same ``CHAT_GENERATION_STOP`` flag as chat and ASR. Those paths
check it from a ``StoppingCriteria`` at each new token. Diffusion checks it at
each denoising step and aborts before the remaining steps and the VAE decode.
"""

import inspect
import json
import os
import shutil
import sys
from pathlib import Path

_REJECTED_CLASS_MARKERS = ("inpaint", "unconditional")
# A component this large is offloaded to disk instead of staged in RAM.
_DISK_OFFLOAD_MIN_BYTES = 1536 * 1024 * 1024
# Hidden folder under the model cache. Not the system temp dir: on Linux that
# is often a size-capped tmpfs, and a full tmpfs kills the worker with SIGBUS.
_OFFLOAD_DIRNAME = ".glaux-t2i-offload"
_RAM_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})
_active_offload_root: Path | None = None


def _class_name_rejected(class_name: str, *, allow_img2img: bool = False) -> bool:
    lowered = class_name.lower()
    if not allow_img2img and "img2img" in lowered:
        return True
    return any(marker in lowered for marker in _REJECTED_CLASS_MARKERS)


def assert_text_to_image_repo(model_path: str, *, allow_img2img: bool = False) -> str:
    """Read diffusers ``model_index.json`` and reject unsupported pipeline classes."""
    index_path = Path(model_path) / "model_index.json"
    if not index_path.is_file():
        raise RuntimeError(
            "Text-to-image models need a diffusers model_index.json in the model cache."
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read model_index.json: {exc}") from exc
    class_name = str(index.get("_class_name") or "")
    if class_name and _class_name_rejected(class_name, allow_img2img=allow_img2img):
        raise RuntimeError(
            f"This diffusion class is not supported (model class is {class_name})."
        )
    return class_name


def _safetensors_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*.safetensors"):
        if ".index." in path.name:
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _available_ram_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def _read_model_index(model_path: str) -> dict:
    index_path = Path(model_path) / "model_index.json"
    return json.loads(index_path.read_text(encoding="utf-8"))


def _import_component_class(library: str, class_name: str):
    if library == "transformers":
        import transformers

        return getattr(transformers, class_name)
    if library == "diffusers":
        import diffusers

        return getattr(diffusers, class_name)
    raise RuntimeError(f"Cannot load a {library}.{class_name} component.")


def _unescape_mount(value: str) -> str:
    """Undo the octal escapes ``/proc/mounts`` uses for spaces and other bytes."""
    chars: list[str] = []
    index = 0
    while index < len(value):
        escaped = value[index : index + 4]
        if (
            value[index] == "\\"
            and len(escaped) == 4
            and all(char in "01234567" for char in escaped[1:])
        ):
            chars.append(chr(int(escaped[1:], 8)))
            index += 4
            continue
        chars.append(value[index])
        index += 1
    return "".join(chars)


def _filesystem_type(path: Path) -> str | None:
    """Linux filesystem type for *path*, or None when ``/proc/mounts`` is absent."""
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return None
    best_type = None
    best_len = -1
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount = Path(_unescape_mount(parts[1]))
        try:
            if resolved != mount:
                resolved.relative_to(mount)
        except ValueError:
            continue
        length = len(mount.parts)
        if length > best_len:
            best_type = parts[2]
            best_len = length
    return best_type


def _is_ram_disk(path: Path) -> bool:
    """True when *path* is tmpfs, ramfs, or devtmpfs."""
    return _filesystem_type(path) in _RAM_FILESYSTEMS


def _existing_anchor(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return path
        current = parent
    return current


def _same_device(left: Path, right: Path) -> bool:
    try:
        return os.stat(_existing_anchor(left)).st_dev == os.stat(_existing_anchor(right)).st_dev
    except OSError:
        return False


def _offload_candidates(model_path: str) -> list[Path]:
    """Places that share a disk with the weights, then ``/var/tmp``.

    ``/var/tmp`` is the fallback when the model cache itself is a RAM disk.
    It is persistent temp space and, unlike ``/tmp``, is not tmpfs on systemd.
    """
    model_dir = Path(model_path).resolve()
    candidates: list[Path] = []
    try:
        from .download import models_cache_dir

        cache = models_cache_dir()
        if _same_device(cache, model_dir):
            candidates.append(cache)
    except (RuntimeError, OSError):
        pass
    parent = model_dir.parent
    if parent not in candidates:
        candidates.append(parent)
    var_tmp = Path("/var/tmp")
    if var_tmp.is_dir() and var_tmp not in candidates:
        candidates.append(var_tmp)
    return candidates


def _disk_offload_root(model_path: str) -> Path:
    """Folder for Accelerate weight files on a real disk.

    The system temp directory is often a size-capped tmpfs. Writing the
    offload files there still consumes RAM, and ``numpy.memmap`` raises
    SIGBUS when that tmpfs cannot allocate the pages.
    """
    for directory in _offload_candidates(model_path):
        if _is_ram_disk(directory):
            continue
        return directory / _OFFLOAD_DIRNAME
    raise RuntimeError(
        "Text-to-image disk offload needs a real disk. "
        "The model cache is on a RAM disk (tmpfs), which cannot hold the offloaded weights."
    )


def _require_offload_space(anchor: Path, nbytes: int) -> None:
    """Reject the offload before ``numpy.memmap`` can die with SIGBUS."""
    if nbytes <= 0:
        return
    try:
        free = shutil.disk_usage(_existing_anchor(anchor)).free
    except OSError:
        return
    if free >= nbytes:
        return
    raise RuntimeError(
        "Not enough free disk space to offload text-to-image weights "
        f"({nbytes / 1024**3:.1f} GiB needed, {free / 1024**3:.1f} GiB free on {anchor})."
    )


def release_disk_offload() -> None:
    """Delete offload files after the pipeline that reads them is gone."""
    global _active_offload_root
    root = _active_offload_root
    _active_offload_root = None
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)


def _prepare_disk_offload(model_path: str, nbytes: int) -> Path:
    """Create a fresh offload folder on a real disk with room for *nbytes*."""
    global _active_offload_root
    root = _disk_offload_root(model_path)
    # Drop a leftover copy first so its bytes count as free space. A live
    # pipeline is released before the next load, and an unlinked mmap stays
    # readable until that process drops it.
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    _require_offload_space(root.parent, nbytes)
    root.mkdir(parents=True, exist_ok=True)
    _active_offload_root = root
    return root


def _offload_disk_bytes(weight_bytes: int, dtype) -> int:
    """Size of the offload copy after the checkpoint is cast to *dtype*.

    Safetensors diffusion weights are stored at 2 bytes per value. A float32
    CPU load writes twice that.
    """
    itemsize = int(getattr(dtype, "itemsize", 2) or 2)
    return max(weight_bytes, weight_bytes * itemsize // 2)


def _weights_need_disk_offload(device: str, available: int | None, weights: int, dtype) -> bool:
    """True when keeping the cast weights resident would exceed about 55% of free RAM."""
    if device not in ("cuda", "cpu") or available is None:
        return False
    resident = _offload_disk_bytes(weights, dtype) if device == "cpu" else weights
    return resident > int(available * 0.55)


def _cuda_is_visible() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _disk_offload_budgets(large_count: int, device: str) -> dict:
    """Cap how much of each large component stays resident.

    The rest is written to disk and loaded for the forward pass. Budgets are
    split across every large component so they can be resident together. A CPU
    load leaves the GPU budget at zero even when CUDA is still visible, so a
    fallback after an out-of-memory error does not place weights on the GPU.
    """
    available = _available_ram_bytes() or (6 * 1024**3)
    cpu_bytes = max(1024**3, int(available * 0.28 / max(large_count, 1)))
    if device != "cuda":
        budgets: dict = {"cpu": cpu_bytes}
        if _cuda_is_visible():
            budgets[0] = 0
        return budgets
    gpu_bytes = 1024**3
    if _cuda_is_visible():
        import torch

        free_vram, _total = torch.cuda.mem_get_info()
        gpu_bytes = max(512 * 1024**2, min(2 * 1024**3, int(free_vram * 0.35)))
    return {0: gpu_bytes, "cpu": cpu_bytes}


def _place_unmapped_modules(pipe, device: str) -> None:
    """Move components that Accelerate did not assign.

    Disk-offloaded modules move themselves during the forward pass. On CUDA the
    VAE stays on the GPU because the latents it decodes are already there. On
    CPU every unmapped component stays on CPU.
    """
    import torch

    target = "cuda" if device == "cuda" else "cpu"
    for component in pipe.components.values():
        if not isinstance(component, torch.nn.Module):
            continue
        if getattr(component, "hf_device_map", None):
            continue
        component.to(target)


def _keep_sampling_off_meta(pipe) -> None:
    """Allocate latents on the CPU when disk offload reports the meta device.

    A component whose weights all live on disk keeps its parameters on the meta
    device. Diffusers reports that as the pipeline device, the denoising loop
    allocates latents there, and Accelerate cannot copy those latents onto the
    CPU for the forward pass.
    """
    if getattr(pipe.__class__, "_glaux_sampling_off_meta", False):
        return
    import torch

    base = pipe.__class__

    def device(self):
        found = super(pinned, self).device
        if getattr(found, "type", None) == "meta":
            return torch.device("cpu")
        return found

    def _execution_device(self):
        found = super(pinned, self)._execution_device
        if getattr(found, "type", None) == "meta":
            return torch.device("cpu")
        return found

    pinned = type(
        base.__name__,
        (base,),
        {
            "device": property(device),
            "_execution_device": property(_execution_device),
            "_glaux_sampling_off_meta": True,
        },
    )
    pipe.__class__ = pinned


def _enable_vae_tiling(pipe) -> bool:
    """Decode in tiles so one full-frame activation does not have to fit at once."""
    enable = getattr(pipe, "enable_vae_tiling", None)
    if not callable(enable):
        vae = getattr(pipe, "vae", None)
        enable = getattr(vae, "enable_tiling", None)
    if not callable(enable):
        return False
    enable()
    return True


def _large_component_load_kwargs(dtype, budgets: dict, offload_folder: str) -> dict:
    """Keyword arguments for one component loaded through Accelerate.

    ``offload_state_dict`` is omitted on purpose. Diffusers enables it only when
    the device map contains ``"disk"``, which is also when it copies the parked
    CPU weights back onto the module. Forcing it on for a component that fits in
    the resident budget leaves every parameter on the meta device, and the
    following ``model.to("cpu")`` raises "Cannot copy out of meta tensor; no data!".
    """
    return {
        "torch_dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "max_memory": budgets,
        "offload_folder": offload_folder,
    }


def _load_disk_offloaded(model_path: str, dtype, device: str):
    """Load components that do not fit in RAM with Accelerate disk offload."""
    from diffusers import DiffusionPipeline

    index = _read_model_index(model_path)
    root = Path(model_path)
    large = []
    for name, value in index.items():
        if name.startswith("_") or not isinstance(value, list) or len(value) != 2:
            continue
        if _safetensors_bytes(root / name) >= _DISK_OFFLOAD_MIN_BYTES:
            large.append(name)
    budgets = _disk_offload_budgets(len(large), device)
    offload_bytes = sum(
        _offload_disk_bytes(_safetensors_bytes(root / name), dtype) for name in large
    )
    offload_root = _prepare_disk_offload(model_path, offload_bytes)
    print(
        "Text-to-image weights are larger than available RAM; "
        f"offloading them to {offload_root}.",
        file=sys.stderr,
        flush=True,
    )
    preloaded = {}
    try:
        for name in large:
            library, class_name = index[name]
            component_cls = _import_component_class(library, class_name)
            preloaded[name] = component_cls.from_pretrained(
                root / name,
                **_large_component_load_kwargs(dtype, budgets, str(offload_root / name)),
            )
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            **preloaded,
        )
        _place_unmapped_modules(pipe, device)
        if device != "cuda":
            _keep_sampling_off_meta(pipe)
        return pipe
    except Exception:
        release_disk_offload()
        raise


def load_text_to_image_pipeline(model_path: str, device: str, *, allow_img2img: bool = False):
    """Load the diffusers pipeline named in ``model_index.json``.

    ``DiffusionPipeline`` follows ``_class_name``, so a class added after the
    last numbered diffusers release still loads. Inpaint classes are rejected
    by name. Img2img classes load only for image-to-image.

    Weights that fit in RAM stay there and stream to the GPU one layer at a
    time. Larger checkpoints, on CUDA or CPU, are offloaded to a real disk
    next to the model cache. The VAE decodes in tiles so the full frame does
    not have to fit in memory at once.
    """
    assert_text_to_image_repo(model_path, allow_img2img=allow_img2img)
    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "diffusers is not installed. Rebuild the Python runtime (npm run build:python)."
        ) from exc
    import torch

    # float32 on CPU. bfloat16 on CUDA when the GPU supports it; float16
    # otherwise. Steps, size, seed, and CFG stay at the pipeline defaults.
    if device == "cpu":
        dtype = torch.float32
    elif device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    available = _available_ram_bytes()
    weights = _safetensors_bytes(Path(model_path))
    offloaded = False
    if _weights_need_disk_offload(device, available, weights, dtype):
        pipe = _load_disk_offloaded(model_path, dtype, device)
        offloaded = True
    else:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if device == "cuda":
            # The text encoder alone is larger than laptop VRAM. Sequential
            # offload keeps the active layer on the GPU and the rest in RAM.
            offload = getattr(pipe, "enable_sequential_cpu_offload", None)
            if callable(offload):
                offload()
            else:
                pipe = pipe.to(device)
        else:
            pipe = pipe.to(device)
    loaded_name = type(pipe).__name__
    if _class_name_rejected(loaded_name, allow_img2img=allow_img2img):
        if offloaded:
            release_disk_offload()
        raise RuntimeError(f"This diffusion class is not supported (loaded {loaded_name}).")
    if _enable_vae_tiling(pipe):
        print("Text-to-image VAE tiling is enabled.", file=sys.stderr, flush=True)
    return pipe


class DiffusionStopped(Exception):
    """The user stopped image generation between denoising steps."""


def _generation_stop_requested() -> bool:
    """Read the flag ``chat_stop()`` sets for chat, ASR, and diffusion."""
    from .chat import CHAT_GENERATION_STOP

    return bool(CHAT_GENERATION_STOP)


def _request_pipeline_interrupt(pipe) -> None:
    """Ask the denoising loop to skip later steps if the callback raise is swallowed."""
    if hasattr(pipe, "_interrupt"):
        pipe._interrupt = True
        return
    try:
        pipe.interrupt = True
    except Exception:
        pass


def _release_pipeline_hooks(pipe) -> None:
    """Run the cleanup ``__call__`` would have reached after a full sample."""
    release = getattr(pipe, "maybe_free_model_hooks", None)
    if not callable(release):
        return
    try:
        release()
    except Exception:
        pass


def _on_diffusion_step_end(pipe, step_index, timestep, callback_kwargs):
    """Diffusion counterpart of ``_ChatStopCriteria``.

    Diffusers calls this at the end of a denoising step. Raising here leaves
    the current step finished and skips every later step, including the VAE
    decode that still runs when only ``pipe._interrupt`` is set.
    """
    if _generation_stop_requested():
        _request_pipeline_interrupt(pipe)
        raise DiffusionStopped("Generation canceled.")
    return callback_kwargs


def _declares_step_end_callback(pipe) -> bool:
    call = getattr(pipe, "__call__", None)
    if call is None:
        return False
    try:
        # Unwrap ``torch.no_grad`` so the real ``__call__`` parameters are visible.
        signature = inspect.signature(inspect.unwrap(call))
    except (TypeError, ValueError):
        return False
    return "callback_on_step_end" in signature.parameters


def _arm_scheduler_stop(pipe) -> None:
    """Watch ``scheduler.step`` when the pipeline has no step-end callback.

    The denoising loop calls ``scheduler.step`` once per step, which is the
    same boundary as ``callback_on_step_end``.
    """
    scheduler = getattr(pipe, "scheduler", None)
    step = getattr(scheduler, "step", None)
    if not callable(step) or getattr(step, "_glaux_stop_hook", False):
        return

    def hooked(*args, **kwargs):
        result = step(*args, **kwargs)
        if _generation_stop_requested():
            _request_pipeline_interrupt(pipe)
            raise DiffusionStopped("Generation canceled.")
        return result

    hooked._glaux_stop_hook = True
    scheduler.step = hooked


def generate_text_to_image(pipe, prompt: str, output_path: str, image_path: str | None = None) -> str:
    """Run the pipeline on ``prompt`` and write one image to ``output_path``.

    ``image_path`` is the optional init image for image-to-image. Strength
    and the other sampling settings stay at the pipeline defaults.

    Prompt and image are passed by name. Some pipelines, including Flux2,
    take the image as the first positional argument, so a positional prompt
    lands in that slot.

    ``chat_stop()`` ends the run at the next denoising step. The partial image
    is not saved.
    """
    text = prompt.strip()
    if not text:
        raise ValueError("Image generation requires a prompt.")
    if not output_path:
        raise ValueError("Image generation output path is not configured.")
    if _generation_stop_requested():
        raise DiffusionStopped("Generation canceled.")
    call = {"prompt": text}
    if image_path:
        from PIL import Image

        with Image.open(image_path) as opened:
            call["image"] = opened.convert("RGB")
    if _declares_step_end_callback(pipe):
        call["callback_on_step_end"] = _on_diffusion_step_end
    else:
        _arm_scheduler_stop(pipe)
    try:
        result = pipe(**call)
    except DiffusionStopped:
        _release_pipeline_hooks(pipe)
        raise
    if _generation_stop_requested():
        _release_pipeline_hooks(pipe)
        raise DiffusionStopped("Generation canceled.")
    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError("Text-to-image pipeline returned no images.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination)
    return str(destination)
