"""Diffusers text-to-image and image-to-image loading and one-shot generation.

``worker.chat`` branches here for those pipeline tags. Inpaint and
unconditional pipelines are rejected. An image-to-image load may use an
img2img class. Callers pass only the current prompt and, for image-to-image,
an optional init image. Size, steps, seed, and CFG stay at the pipeline
defaults.

Stop uses the same ``CHAT_GENERATION_STOP`` flag as chat and ASR. Those paths
check it from a ``StoppingCriteria`` at each new token. Diffusion checks it at
each denoising step and aborts before the remaining steps and the VAE decode.

CUDA placement is tiered. Weights that fit in free VRAM stay on the GPU.
Otherwise a whole component is swapped once per run, or groups are prefetched
from RAM. Disk is only for checkpoints that exceed the RAM budget, and that
path stays on Accelerate (see ``_load_disk_offloaded``).
"""

import gc
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

_REJECTED_CLASS_MARKERS = ("inpaint", "unconditional")
# A component this large is device-mapped on the Accelerate disk path.
# Smaller components load with the pipeline and are placed as a group.
_DISK_OFFLOAD_MIN_BYTES = 1536 * 1024 * 1024
# Hidden folder under the model cache. Not the system temp dir: on Linux that
# is often a size-capped tmpfs, and a full tmpfs kills the worker with SIGBUS.
# ``engines/engineManager.js`` deletes ``.glaux-t2i-offload/pid-<pid>`` after
# it kills the worker. Placement code must keep that relative layout.
_OFFLOAD_DIRNAME = ".glaux-t2i-offload"
# Left free for activations, the CUDA caching allocator, and the tiled VAE
# decode. A component counts as fitting in VRAM only when it fits in free VRAM
# minus this. 3 GiB covers a 1024px step on common shapes; a tighter card uses
# group offload instead of dying on the first step.
_ACTIVATION_HEADROOM_BYTES = 3 * 1024**3
# Cast weights may use this share of currently available RAM before disk.
# psutil's available figure already excludes memory the OS has given out.
# The previous 0.55 share sent models that still fit in RAM to disk.
_RAM_RESIDENT_FRACTION = 0.70
# Block groups keep one transformer block resident and prefetch the next.
# Below this much usable VRAM (free minus the activation headroom), leaf
# groups are smaller than a full block plus that prefetch.
_GROUP_BLOCK_MIN_VRAM_BYTES = 2 * 1024**3
# CPU staging floor for a large component once the RAM share is spoken for.
_CPU_BUDGET_FLOOR = 64 * 1024**2
# Spare GPU below this is too tight for a disk-offload shard plus activations.
_MIN_GPU_OFFLOAD_BYTES = 512 * 1024**2
_GPU_SLIVER_BYTES = 256 * 1024**2

TIER_CUDA = "cuda"
TIER_MODEL = "model_cpu_offload"
TIER_GROUP = "group_offload"
TIER_GROUP_LEAF = "group_offload_leaf"
TIER_DISK = "disk"
_CUDA_TIER_ORDER = (TIER_CUDA, TIER_MODEL, TIER_GROUP, TIER_GROUP_LEAF, TIER_DISK)
# After a tier runs out of memory, later loads of that model directory in this
# process start at the next tier down. A new worker starts with this empty.
_cuda_tier_limits: dict[str, str] = {}
_RAM_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})
_active_offload_root: Path | None = None
# Bumps every time this process creates an offload folder, so a failed tier
# can drop the folder it just created without deleting an older model's files.
_active_offload_generation = 0


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


def _offload_pid_dirname(pid: int | None = None) -> str:
    return f"pid-{os.getpid() if pid is None else pid}"


def _pid_from_offload_dirname(name: str) -> int | None:
    if not name.startswith("pid-") or not name[4:].isdigit():
        return None
    return int(name[4:])


def _pid_alive(pid: int) -> bool:
    """True when *pid* is this process or another process we must not delete.

    Without psutil, another pid is treated as live so one instance cannot
    remove an offload folder it cannot prove is stale.
    """
    if pid == os.getpid():
        return True
    try:
        import psutil
    except ImportError:
        return True
    return bool(psutil.pid_exists(pid))


def _clean_stale_offload_dirs(parent: Path) -> None:
    """Remove pid folders whose process is gone. Leave a live instance alone."""
    if not parent.is_dir():
        return
    for child in list(parent.iterdir()):
        pid = _pid_from_offload_dirname(child.name)
        if pid is None or _pid_alive(pid):
            continue
        shutil.rmtree(child, ignore_errors=True)


def release_disk_offload() -> None:
    """Delete this process's offload files after the pipeline that reads them is gone."""
    global _active_offload_root
    root = _active_offload_root
    _active_offload_root = None
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)


def _prepare_disk_offload(model_path: str, nbytes: int) -> Path:
    """Create a fresh per-process offload folder on a real disk with room for *nbytes*."""
    global _active_offload_root, _active_offload_generation
    parent = _disk_offload_root(model_path)
    # Another app instance keeps its own pid folder. A folder left by a dead
    # pid, including one whose unload timed out, is removed here.
    _clean_stale_offload_dirs(parent)
    root = parent / _offload_pid_dirname()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    _require_offload_space(parent, nbytes)
    root.mkdir(parents=True, exist_ok=True)
    _active_offload_generation += 1
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
    """True when the resident weights would exceed ``_RAM_RESIDENT_FRACTION`` of free RAM.

    CUDA compares checkpoint bytes (fp16 and bf16 files already match the
    runtime dtype). A float32 CPU load doubles a 2-byte checkpoint.
    """
    if device not in ("cuda", "cpu") or available is None:
        return False
    resident = _offload_disk_bytes(weights, dtype) if device == "cpu" else weights
    return resident > int(available * _RAM_RESIDENT_FRACTION)


def _cast_component_bytes(component_bytes: dict[str, int], dtype) -> dict[str, int]:
    return {name: _offload_disk_bytes(size, dtype) for name, size in component_bytes.items()}


def _component_weight_bytes(model_path: str) -> dict[str, int]:
    """Safetensors bytes for each ``model_index.json`` component folder."""
    index = _read_model_index(model_path)
    root = Path(model_path)
    sizes: dict[str, int] = {}
    for name, value in index.items():
        if name.startswith("_") or not isinstance(value, list) or len(value) != 2:
            continue
        nbytes = _safetensors_bytes(root / name)
        if nbytes > 0:
            sizes[name] = nbytes
    return sizes


def _unscoped_weight_bytes(model_path: str, components: dict[str, int]) -> int:
    """Checkpoint bytes that are not inside a component folder."""
    total = _safetensors_bytes(Path(model_path))
    return max(0, total - sum(components.values()))


def _is_denoiser_component(name: str) -> bool:
    return name == "unet" or name.startswith("transformer")


def _preferred_gpu_component(sizes: dict[str, int]) -> str | None:
    """The denoiser when one is present, otherwise the largest component."""
    if not sizes:
        return None
    denoisers = {name: size for name, size in sizes.items() if _is_denoiser_component(name)}
    pool = denoisers or sizes
    return max(pool, key=pool.get)


def _partition_components(sizes: dict[str, int]) -> tuple[dict[str, int], int]:
    """Large components (device-mapped) and the byte size of everything else."""
    large = {name: size for name, size in sizes.items() if size >= _DISK_OFFLOAD_MIN_BYTES}
    if not large and sum(sizes.values()) >= _DISK_OFFLOAD_MIN_BYTES:
        large = dict(sizes)
    small = sum(size for name, size in sizes.items() if name not in large)
    return large, small


def select_group_offload_type(free_vram: int) -> str:
    """``block_level`` when a prefetched block fits, otherwise ``leaf_level``."""
    usable = int(free_vram) - _ACTIVATION_HEADROOM_BYTES
    if usable < _GROUP_BLOCK_MIN_VRAM_BYTES:
        return "leaf_level"
    return "block_level"


def group_offload_pre_pin(total_cast: int, available_ram: int | None) -> bool:
    """True when a pinned duplicate of the weights still fits in the RAM budget.

    Streamed group offload keeps a host copy. Pinning that copy is skipped when
    the duplicate would pass ``_RAM_RESIDENT_FRACTION`` of available RAM.
    """
    if available_ram is None:
        return False
    return total_cast * 2 <= int(available_ram * _RAM_RESIDENT_FRACTION)


def select_cuda_tier(
    component_bytes: dict[str, int],
    free_vram: int,
    available_ram: int | None,
    dtype,
    extra_bytes: int = 0,
) -> str:
    """Pick a CUDA placement from cast component sizes and free memory.

    ``extra_bytes`` are checkpoint files that are not inside a component
    folder. They count toward the RAM and VRAM totals but not toward the
    largest component, because they are not a module we can offload alone.
    """
    cast = _cast_component_bytes(component_bytes, dtype)
    extra = _offload_disk_bytes(extra_bytes, dtype) if extra_bytes else 0
    total = sum(cast.values()) + extra
    largest = max(cast.values(), default=0)
    usable = int(free_vram) - _ACTIVATION_HEADROOM_BYTES
    exceeds_ram = (
        available_ram is not None and total > int(available_ram * _RAM_RESIDENT_FRACTION)
    )
    # from_pretrained stages weights in RAM even when the GPU could hold them.
    if total > 0 and exceeds_ram:
        return TIER_DISK
    if total <= 0 or (usable > 0 and total <= usable):
        return TIER_CUDA
    if usable > 0 and largest <= usable:
        return TIER_MODEL
    if select_group_offload_type(free_vram) == "leaf_level":
        return TIER_GROUP_LEAF
    return TIER_GROUP


def cuda_tiers_from(preferred: str, limit: str | None = None) -> list[str]:
    """Tiers to try, starting at *preferred* or at *limit* when that is lower."""
    start = preferred
    if limit is not None and _CUDA_TIER_ORDER.index(limit) > _CUDA_TIER_ORDER.index(preferred):
        start = limit
    return list(_CUDA_TIER_ORDER[_CUDA_TIER_ORDER.index(start) :])


def _next_cuda_tier(tier: str) -> str | None:
    order = cuda_tiers_from(tier)
    if len(order) < 2:
        return None
    return order[1]


def _note_tier_limit(model_path: str, failed_tier: str) -> str | None:
    """Remember the next lower tier for *model_path*. None at the disk tier."""
    nxt = _next_cuda_tier(failed_tier)
    if nxt is None:
        return None
    current = _cuda_tier_limits.get(model_path)
    if current is None or _CUDA_TIER_ORDER.index(nxt) > _CUDA_TIER_ORDER.index(current):
        _cuda_tier_limits[model_path] = nxt
    return nxt


def lower_cuda_tier_after_oom(pipe) -> bool:
    """Record a lower tier after *pipe* ran out of GPU memory. False at disk."""
    tier = getattr(pipe, "_glaux_placement_tier", None)
    model_path = getattr(pipe, "_glaux_model_path", None)
    if not isinstance(tier, str) or not model_path:
        return False
    return _note_tier_limit(str(model_path), tier) is not None


def _denoiser_gpu_budget(free_vram: int | None, small_bytes: int) -> int:
    """GPU bytes for the denoiser on the Accelerate disk path.

    Most of free VRAM, after the activation headroom and the small components
    that are moved onto the GPU whole. When that spare is below
    ``_MIN_GPU_OFFLOAD_BYTES``, use half of free VRAM so a small card still
    runs, leaving a sliver for activations.
    """
    if not free_vram or free_vram <= 0:
        return 0
    free = int(free_vram)
    small = max(0, int(small_bytes))
    spare = free - _ACTIVATION_HEADROOM_BYTES - small
    if spare >= _MIN_GPU_OFFLOAD_BYTES:
        return spare
    # The headroom does not fit. Keep the small components and a sliver, and
    # give the denoiser at most half of what is left so the first step can run.
    remain = free - small - _GPU_SLIVER_BYTES
    if remain < _MIN_GPU_OFFLOAD_BYTES:
        return max(0, remain)
    return min(int(remain * 0.5), remain)


def _split_cpu_budgets(sizes: dict[str, int], total_budget: int) -> dict[str, int]:
    """Give each large component a floor, then the denoiser, then the rest.

    Leftover RAM goes to the other large components (a text encoder stays in
    RAM when it fits) instead of being split evenly and pushing the denoiser
    to disk.
    """
    if not sizes:
        return {}
    floor = _CPU_BUDGET_FLOOR
    budgets = {name: min(size, floor) for name, size in sizes.items()}
    remaining = max(0, int(total_budget) - sum(budgets.values()))
    primary = _preferred_gpu_component(sizes)
    order = [primary] if primary in sizes else []
    order.extend(
        sorted((name for name in sizes if name != primary), key=lambda name: sizes[name], reverse=True)
    )
    for name in order:
        if remaining <= 0:
            break
        need = sizes[name] - budgets[name]
        if need <= 0:
            continue
        grant = min(need, remaining)
        budgets[name] += grant
        remaining -= grant
    return budgets


def _component_disk_budgets(
    component_bytes: dict[str, int],
    *,
    device: str,
    available_ram: int,
    free_vram: int | None = None,
    cuda_visible: bool = False,
) -> dict[str, dict]:
    """Per-component Accelerate ``max_memory`` for the disk path.

    The denoiser gets the GPU budget. Other large components get none, so a
    text encoder that runs once per prompt does not stay resident during
    denoising. A CPU load still publishes a zero GPU budget when CUDA is
    visible, so a fallback after an out-of-memory error does not place
    weights on the GPU.
    """
    large, small = _partition_components(component_bytes)
    if not large:
        return {}
    cpu_budgets = _split_cpu_budgets(large, int(available_ram * _RAM_RESIDENT_FRACTION))
    gpu_name = _preferred_gpu_component(large) if device == "cuda" else None
    gpu_budget = _denoiser_gpu_budget(free_vram, small) if device == "cuda" else 0
    budgets: dict[str, dict] = {}
    for name, cpu_bytes in cpu_budgets.items():
        entry: dict = {"cpu": cpu_bytes}
        if device == "cuda":
            entry[0] = gpu_budget if name == gpu_name else 0
        elif cuda_visible:
            entry[0] = 0
        budgets[name] = entry
    return budgets


class _PlacementUnsupported(Exception):
    """This pipeline cannot use the tier. Try the next one."""


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """True when *exc* is a CUDA out-of-memory error, including wrapped causes."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower().replace("_", "")
        text = str(current).lower()
        compact = text.replace(" ", "")
        if name == "outofmemoryerror" and ("cuda" in text or "out of memory" in text):
            return True
        if "cuda" in text and ("outofmemory" in compact or "withoom" in compact):
            return True
        current = current.__cause__ or current.__context__
    return False


def _should_drop_placement(exc: BaseException) -> bool:
    """True when the next, less eager CUDA tier should be tried."""
    if isinstance(exc, (_PlacementUnsupported, MemoryError)):
        return True
    return _is_cuda_oom_error(exc)


def _placement_failure_summary(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    if len(text) > 160:
        return text[:160] + "…"
    return text


def _format_gib(nbytes: int | None) -> str:
    if nbytes is None:
        return "unknown"
    return f"{nbytes / 1024**3:.2f}GiB"


def _dtype_name(dtype) -> str:
    text = str(dtype)
    prefix = "torch."
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _format_placement_log(
    tier: str,
    dtype_name: str,
    free_vram: int | None,
    available_ram: int | None,
    component_bytes: dict[str, int],
) -> str:
    parts = [f"{name}={_format_gib(size)}" for name, size in sorted(component_bytes.items())]
    comps = ", ".join(parts) if parts else "none"
    return (
        f"Text-to-image CUDA placement: {tier} "
        f"(dtype={dtype_name}, free_vram={_format_gib(free_vram)}, "
        f"available_ram={_format_gib(available_ram)}, components: {comps})."
    )


def _format_step_timing(steps: int, elapsed: float) -> str:
    average = elapsed / steps if steps else 0.0
    return (
        f"Text-to-image generation: {steps} steps, "
        f"average {average:.2f}s/step, total {elapsed:.1f}s."
    )


def _group_offload_kwargs(offload_type: str, *, pre_pin: bool) -> dict:
    """Keyword arguments for diffusers 0.41 ``apply_group_offloading``.

    Streams require ``num_blocks_per_group`` of 1; a larger group is forced
    back to 1 and disables the overlap the stream was opened for. Disk
    offload is intentionally absent: see ``_load_disk_offloaded``.
    """
    return {
        "offload_type": offload_type,
        "num_blocks_per_group": 1 if offload_type == "block_level" else None,
        "non_blocking": True,
        "use_stream": True,
        "record_stream": True,
        "low_cpu_mem_usage": not pre_pin,
    }


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _release_failed_placement(pipe) -> None:
    """Drop a pipeline that failed before it is published.

    The disk folder is not removed here. A failed non-disk tier must not delete
    an offload directory that still belongs to the previous model, and a failed
    disk load releases its own folder in ``_load_disk_offloaded``.
    """
    if pipe is not None:
        _release_pipeline_hooks(pipe)
    gc.collect()
    _empty_cuda_cache()


def _run_placement_attempts(tiers: list[str], attempt, on_drop=None):
    """Call *attempt(tier)* until one returns. Drop a tier on OOM or if it is unsupported."""
    last: BaseException | None = None
    for index, tier in enumerate(tiers):
        generation = _active_offload_generation
        try:
            return attempt(tier)
        except Exception as exc:
            last = exc
            # Drop a folder this attempt created. Leave an older model's folder
            # in place when the attempt failed before preparing one.
            if _active_offload_generation != generation:
                release_disk_offload()
            if index >= len(tiers) - 1 or not _should_drop_placement(exc):
                raise
            nxt = tiers[index + 1]
            if on_drop is not None:
                on_drop(tier, nxt)
            print(
                f"Text-to-image placement {tier} did not fit "
                f"({_placement_failure_summary(exc)}); trying {nxt}.",
                file=sys.stderr,
                flush=True,
            )
    if last is not None:
        raise last
    raise RuntimeError("No text-to-image placement was attempted.")


def _cuda_is_visible() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _free_vram_bytes() -> int | None:
    if not _cuda_is_visible():
        return None
    import torch

    free, _total = torch.cuda.mem_get_info()
    return int(free)


def _place_unmapped_modules(pipe, device: str) -> None:
    """Move components that Accelerate did not assign.

    Large components carry their own device map and move themselves. On CUDA
    whatever is left (a VAE under the large-component threshold, for example)
    is placed on the GPU. On CPU every unmapped component stays on CPU.
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


def _load_disk_offloaded(
    model_path: str,
    dtype,
    device: str,
    *,
    available_ram: int | None = None,
    free_vram: int | None = None,
):
    """Load components that do not fit in RAM with Accelerate disk offload.

    Diffusers 0.41.0 group offload accepts ``offload_to_disk_path`` and can
    prefetch that file with ``use_stream``, but ``ModuleGroup._offload_to_disk``
    then replaces every parameter with a full-size CPU ``empty_like``. It does
    this again after every offload, including every denoising step, so the
    weights stay in RAM. That cannot host a model larger than RAM, and it does
    not apply to every pipeline equally (a shared offload directory also
    collides when two components hash the same group id). Accelerate's
    ``device_map`` memmap does free the overflow. CPU loads keep this path:
    group-offload streams require a CUDA onload device.

    On CUDA the denoiser gets free VRAM minus the activation headroom. Other
    large components, including a text encoder that runs once per prompt, get
    no GPU budget so they do not stay resident during denoising.
    """
    from diffusers import DiffusionPipeline

    index = _read_model_index(model_path)
    root = Path(model_path)
    measured = _component_weight_bytes(model_path)
    cast = _cast_component_bytes(measured, dtype)
    large, _small = _partition_components(cast)
    if available_ram is None:
        available_ram = _available_ram_bytes() or (6 * 1024**3)
    if device == "cuda" and free_vram is None:
        free_vram = _free_vram_bytes() or 0
    budgets = _component_disk_budgets(
        cast,
        device=device,
        available_ram=available_ram,
        free_vram=free_vram,
        cuda_visible=device == "cuda" or _cuda_is_visible(),
    )
    offload_bytes = sum(large.values())
    offload_root = _prepare_disk_offload(model_path, offload_bytes)
    try:
        gpu_name = _preferred_gpu_component(large) if device == "cuda" else None
        gpu_budget = budgets.get(gpu_name or "", {}).get(0, 0)
        gpu_note = f" Denoiser GPU budget {_format_gib(gpu_budget)}." if device == "cuda" else ""
        print(
            "Text-to-image weights are larger than available RAM; "
            f"offloading them to {offload_root}.{gpu_note}",
            file=sys.stderr,
            flush=True,
        )
        preloaded = {}
        for name in large:
            if name not in index:
                continue
            library, class_name = index[name]
            component_cls = _import_component_class(library, class_name)
            preloaded[name] = component_cls.from_pretrained(
                root / name,
                **_large_component_load_kwargs(dtype, budgets[name], str(offload_root / name)),
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


def _apply_model_cpu_offload(pipe) -> None:
    """Swap whole components once per run so the denoiser stays on the GPU.

    Diffusers' ``model_cpu_offload_seq`` moves the text encoder on for the
    prompt and off again before the denoising loop. Pipelines with no sequence
    cannot do this; the caller tries group offload instead.
    """
    if not getattr(pipe, "model_cpu_offload_seq", None):
        raise _PlacementUnsupported("pipeline has no model_cpu_offload_seq")
    enable = getattr(pipe, "enable_model_cpu_offload", None)
    if not callable(enable):
        raise _PlacementUnsupported("pipeline has no enable_model_cpu_offload")
    enable()


def _apply_group_offload(pipe, offload_type: str, *, pre_pin: bool) -> None:
    """Prefetch groups onto CUDA. Every module is included, then offloaded.

    ``enable_group_offload(..., exclude_modules=...)`` moves excluded modules
    onto the onload device and leaves them there. A text encoder excluded that
    way would stay resident through denoising. Applying the hook to every
    ``nn.Module`` keeps it on the GPU only during its own forward.
    """
    try:
        from diffusers.hooks import apply_group_offloading
    except ImportError as exc:
        raise _PlacementUnsupported("diffusers group offload is unavailable") from exc
    import torch

    components = getattr(pipe, "components", None)
    if not isinstance(components, dict):
        raise _PlacementUnsupported("pipeline has no components")
    kwargs = _group_offload_kwargs(offload_type, pre_pin=pre_pin)
    applied = 0
    for name, component in components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        try:
            apply_group_offloading(
                component,
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                **kwargs,
            )
            applied += 1
        except Exception as exc:
            if _is_cuda_oom_error(exc) or isinstance(exc, MemoryError):
                raise
            if _is_denoiser_component(name):
                raise _PlacementUnsupported(
                    f"{name} group offload failed: {_placement_failure_summary(exc)}"
                ) from exc
            component.to("cpu")
            print(
                f"Text-to-image left {name} on CPU ({_placement_failure_summary(exc)}).",
                file=sys.stderr,
                flush=True,
            )
    if applied == 0:
        raise _PlacementUnsupported("pipeline has no modules to group-offload")


def _mark_cuda_pipe(pipe, tier: str, model_path: str) -> None:
    pipe._glaux_placement_tier = tier
    pipe._glaux_uses_cuda = True
    pipe._glaux_model_path = model_path


def _place_cuda_tier(
    model_path: str,
    dtype,
    tier: str,
    cast: dict[str, int],
    *,
    available_ram: int | None,
    free_vram: int | None,
):
    """Load *model_path* and apply one CUDA tier. The caller retries on failure."""
    from diffusers import DiffusionPipeline

    pipe = None
    try:
        if tier == TIER_DISK:
            return _load_disk_offloaded(
                model_path,
                dtype,
                "cuda",
                available_ram=available_ram,
                free_vram=free_vram,
            )
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if tier == TIER_CUDA:
            moved = pipe.to("cuda")
            if moved is not None:
                pipe = moved
        elif tier == TIER_MODEL:
            _apply_model_cpu_offload(pipe)
        elif tier in (TIER_GROUP, TIER_GROUP_LEAF):
            offload_type = "leaf_level" if tier == TIER_GROUP_LEAF else "block_level"
            pre_pin = group_offload_pre_pin(sum(cast.values()), available_ram)
            _apply_group_offload(pipe, offload_type, pre_pin=pre_pin)
        else:
            raise _PlacementUnsupported(f"unknown placement {tier}")
        return pipe
    except Exception:
        failed = pipe
        pipe = None
        _release_failed_placement(failed)
        del failed
        raise


def _load_cuda_pipeline(model_path: str, dtype, available_ram: int | None, free_vram: int | None):
    """Load a CUDA pipeline, dropping a tier when the choice does not fit."""
    resolved = str(Path(model_path).resolve())
    measured = _component_weight_bytes(model_path)
    extra = _unscoped_weight_bytes(model_path, measured)
    cast = _cast_component_bytes(measured, dtype)
    if extra:
        cast = {**cast, "other": _offload_disk_bytes(extra, dtype)}
    preferred = select_cuda_tier(
        measured,
        free_vram or 0,
        available_ram,
        dtype,
        extra_bytes=extra,
    )
    tiers = cuda_tiers_from(preferred, _cuda_tier_limits.get(resolved))
    dtype_name = _dtype_name(dtype)

    def attempt(tier: str):
        pipe = _place_cuda_tier(
            model_path,
            dtype,
            tier,
            cast,
            available_ram=available_ram,
            free_vram=free_vram,
        )
        _mark_cuda_pipe(pipe, tier, resolved)
        print(
            _format_placement_log(tier, dtype_name, free_vram, available_ram, cast),
            file=sys.stderr,
            flush=True,
        )
        return pipe

    def on_drop(failed: str, _nxt: str) -> None:
        _note_tier_limit(resolved, failed)

    return _run_placement_attempts(tiers, attempt, on_drop)


def load_text_to_image_pipeline(model_path: str, device: str, *, allow_img2img: bool = False):
    """Load the diffusers pipeline named in ``model_index.json``.

    ``DiffusionPipeline`` follows ``_class_name``, so a class added after the
    last numbered diffusers release still loads. Inpaint classes are rejected
    by name. Img2img classes load only for image-to-image.

    On CUDA the placement is the fastest tier the checkpoint fits: the whole
    pipeline on the GPU, whole components swapped once per run, streamed group
    offload from RAM, or Accelerate disk offload when the weights exceed the
    RAM budget. A tier that runs out of memory is retried one step down. CPU
    and MPS keep a single placement: resident weights, or the same Accelerate
    disk path when the RAM budget is exceeded. The VAE decodes in tiles so the
    full frame does not have to fit in memory at once.
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
    generation = _active_offload_generation
    if device == "cuda":
        pipe = _load_cuda_pipeline(model_path, dtype, available, _free_vram_bytes())
    elif _weights_need_disk_offload(device, available, _safetensors_bytes(Path(model_path)), dtype):
        pipe = _load_disk_offloaded(model_path, dtype, device, available_ram=available)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe = pipe.to(device)
    loaded_name = type(pipe).__name__
    if _class_name_rejected(loaded_name, allow_img2img=allow_img2img):
        # Only the folder this load created. A rejected class must not delete
        # an offload directory that still belongs to the previous model.
        if _active_offload_generation != generation:
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


class _StepTimer:
    """Wall clock around one ``pipe()`` call, with a count of denoising steps."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.steps = 0

    def mark(self, step_index=None) -> None:
        if isinstance(step_index, int) and step_index >= 0:
            self.steps = max(self.steps, step_index + 1)
        else:
            self.steps += 1

    def log(self) -> None:
        if self.steps <= 0:
            return
        elapsed = time.perf_counter() - self.started
        print(_format_step_timing(self.steps, elapsed), file=sys.stderr, flush=True)


def _bind_step_timer(timer: _StepTimer):
    """Step-end callback that records timing and still honors cancel."""

    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        timer.mark(step_index)
        return _on_diffusion_step_end(pipe, step_index, timestep, callback_kwargs)

    return callback_on_step_end


def _arm_scheduler_stop(pipe, timer: _StepTimer | None = None) -> None:
    """Watch ``scheduler.step`` when the pipeline has no step-end callback.

    The denoising loop calls ``scheduler.step`` once per step, which is the
    same boundary as ``callback_on_step_end``. Rebinding each generation keeps
    the timer on the current call.
    """
    scheduler = getattr(pipe, "scheduler", None)
    step = getattr(scheduler, "step", None)
    if not callable(step):
        return
    original = getattr(step, "_glaux_original_step", step)

    def hooked(*args, **kwargs):
        result = original(*args, **kwargs)
        if timer is not None:
            timer.mark()
        if _generation_stop_requested():
            _request_pipeline_interrupt(pipe)
            raise DiffusionStopped("Generation canceled.")
        return result

    hooked._glaux_stop_hook = True
    hooked._glaux_original_step = original
    scheduler.step = hooked


def generate_text_to_image(pipe, prompt: str, output_path: str, image_path: str | None = None) -> str:
    """Run the pipeline on ``prompt`` and write one image to ``output_path``.

    ``image_path`` is the optional init image for image-to-image. Strength
    and the other sampling settings stay at the pipeline defaults.

    Prompt and image are passed by name. Some pipelines, including Flux2,
    take the image as the first positional argument, so a positional prompt
    lands in that slot.

    ``chat_stop()`` ends the run at the next denoising step. The partial image
    is not saved. Stderr then reports the step count, the average seconds per
    step, and the wall time of the pipeline call.
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
    timer = _StepTimer()
    if _declares_step_end_callback(pipe):
        call["callback_on_step_end"] = _bind_step_timer(timer)
    else:
        _arm_scheduler_stop(pipe, timer)
    try:
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
    finally:
        timer.log()
