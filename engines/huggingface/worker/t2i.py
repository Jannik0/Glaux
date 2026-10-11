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
import traceback
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
# CUDA cast weights may use this share of currently available RAM before disk.
# psutil's available figure already excludes memory the OS has given out.
_RAM_RESIDENT_FRACTION = 0.70
# CPU loads, including the reload after a CUDA failure, keep the smaller
# resident threshold and the equal Accelerate split. Those numbers are not
# the CUDA shares above.
_CPU_RAM_RESIDENT_FRACTION = 0.55
_CPU_ACCELERATE_RAM_FRACTION = 0.28
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


# Names ``from_pretrained`` resolves when ``variant`` is omitted and
# ``use_safetensors`` stays at its default. Diffusers 0.41 ``_fetch_index_file``
# looks for the non-variant index first and, on a hit, loads only the shards in
# ``weight_map``. Otherwise it loads the non-variant single file. Sibling
# ``*.fp16.safetensors`` files and a root single-file checkpoint are not selected.
_DIFFUSERS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
_DIFFUSERS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
_TRANSFORMERS_WEIGHTS_NAME = "model.safetensors"
_TRANSFORMERS_INDEX_NAME = "model.safetensors.index.json"
# Safetensors header dtype → stored bytes per value. Runtime size is numel
# times the dtype we cast to, not this width and not the file size.
_SAFETENSORS_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
    "C64": 8,
    "C128": 16,
}


def _safetensors_header(path: Path) -> dict:
    """JSON header of a safetensors file. The tensor body is not read."""
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"safetensors header is truncated: {path}")
        length = int.from_bytes(raw_len, "little")
        payload = handle.read(length)
    if len(payload) < length:
        raise ValueError(f"safetensors header is truncated: {path}")
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return header


def _header_numel(header: dict) -> int:
    total = 0
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape") or []
        numel = 1
        for dim in shape:
            numel *= int(dim)
        if numel:
            total += numel
    return total


def _weight_filenames(library: str) -> tuple[str, str]:
    if library == "transformers":
        return _TRANSFORMERS_WEIGHTS_NAME, _TRANSFORMERS_INDEX_NAME
    return _DIFFUSERS_WEIGHTS_NAME, _DIFFUSERS_INDEX_NAME


def _component_checkpoint_files(folder: Path, library: str) -> list[Path]:
    """Files one component loads from *folder* with ``variant=None``.

    An index wins over a sibling single file, matching ``_fetch_index_file``
    then ``_get_checkpoint_shard_files``. Shard names come from ``weight_map``
    and are counted once. A ``.fp16`` (or any other) variant is a different
    filename and is left out.
    """
    weights_name, index_name = _weight_filenames(library)
    index_path = folder / index_name
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if isinstance(weight_map, dict) and weight_map:
            shards: list[Path] = []
            seen: set[str] = set()
            for shard in weight_map.values():
                if not isinstance(shard, str) or shard in seen:
                    continue
                seen.add(shard)
                path = folder / shard
                if path.is_file():
                    shards.append(path)
            if shards:
                return shards
    single = folder / weights_name
    if single.is_file():
        return [single]
    return []


def _runtime_tensor_bytes(files: list[Path], dtype) -> int:
    """``numel * target itemsize`` across *files*. Headers only."""
    itemsize = int(getattr(dtype, "itemsize", 2) or 2)
    numel = 0
    for path in files:
        numel += _header_numel(_safetensors_header(path))
    return numel * itemsize


def measure_component_bytes(model_path: str, dtype) -> dict[str, int]:
    """Runtime bytes of the checkpoint each ``model_index.json`` component loads.

    The size is the cast dtype, not the stored width: an fp32 or fp8 file loaded
    as fp16 occupies ``numel * 2``. Components that are not a subdirectory are
    skipped; ``from_pretrained`` would load those from the repo root and a root
    single-file checkpoint is not one of them.
    """
    index = _read_model_index(model_path)
    root = Path(model_path)
    sizes: dict[str, int] = {}
    for name, value in index.items():
        if name.startswith("_") or not isinstance(value, list) or len(value) != 2:
            continue
        folder = root / name
        if not folder.is_dir():
            continue
        files = _component_checkpoint_files(folder, str(value[0]))
        if not files:
            continue
        nbytes = _runtime_tensor_bytes(files, dtype)
        if nbytes > 0:
            sizes[name] = nbytes
    return sizes


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
    """Scale a 2-byte-per-value estimate to *dtype*.

    Callers that already measured ``numel * itemsize`` from safetensors headers
    must not use this. It remains for a raw byte count that assumed fp16 storage.
    """
    itemsize = int(getattr(dtype, "itemsize", 2) or 2)
    return max(weight_bytes, weight_bytes * itemsize // 2)


def _resident_fraction(device: str) -> float:
    if device == "cpu":
        return _CPU_RAM_RESIDENT_FRACTION
    return _RAM_RESIDENT_FRACTION


def _weights_need_disk_offload(device: str, available: int | None, weights: int, dtype) -> bool:
    """True when a 2-byte checkpoint estimate would exceed the resident RAM share.

    CPU uses ``_CPU_RAM_RESIDENT_FRACTION`` and doubles a float32 load. CUDA
    compares the estimate to ``_RAM_RESIDENT_FRACTION``. Loads that can read
    headers use :func:`measure_component_bytes` instead of this estimate.
    """
    if device not in ("cuda", "cpu") or available is None:
        return False
    resident = _offload_disk_bytes(weights, dtype) if device == "cpu" else weights
    return resident > int(available * _resident_fraction(device))


def _resident_exceeds(device: str, available: int | None, resident: int) -> bool:
    """True when *resident* runtime bytes exceed that device's RAM share."""
    if device not in ("cuda", "cpu") or available is None:
        return False
    return resident > int(available * _resident_fraction(device))


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
    """Pick a CUDA placement from runtime component sizes and free memory.

    *component_bytes* and *extra_bytes* are already ``numel * `` the target
    itemsize (see :func:`measure_component_bytes`). *dtype* is unused; it is
    kept so callers can show which cast the sizes belong to. ``extra_bytes``
    count toward the RAM and VRAM totals but not toward the largest component.
    The loader passes 0: a root single-file checkpoint is not loaded.

    Non-denoiser modules are leaf-offloaded even when this returns
    ``group_offload``, so a text encoder larger than VRAM does not onload as
    one group. See :func:`_offload_type_for_component`.
    """
    del dtype
    total = sum(component_bytes.values()) + max(0, int(extra_bytes))
    largest = max(component_bytes.values(), default=0)
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


def clear_cuda_tier_limit(model_path: str) -> None:
    """Forget a lowered tier for *model_path*.

    The cap is read once at the start of a load, so the immediate retry still
    uses it. A successful load of that model clears it, and so does an explicit
    unload. A later load measures free memory again instead of keeping a cap
    that was only true during GPU contention.
    """
    _cuda_tier_limits.pop(str(Path(model_path).resolve()), None)


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
        budget = max(0, remain)
    else:
        budget = min(int(remain * 0.5), remain)
    # A few mebibytes does not fit a denoiser block, and a zero budget makes
    # Accelerate set ``main_device`` to CPU so the whole forward stays there.
    # When the card can hold the minimum shard, use that instead of the remainder.
    if budget < _MIN_GPU_OFFLOAD_BYTES <= free:
        return _MIN_GPU_OFFLOAD_BYTES
    return budget


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


def _disk_component_gpu_budgets(
    large: dict[str, int],
    small_bytes: int,
    free_vram: int | None,
) -> dict[str, int]:
    """GPU ``max_memory`` for each large component on the CUDA disk tier.

    Accelerate's device map is static. A component whose budget is only CPU
    and disk has no parameter on the GPU, so ``dispatch_model`` sets
    ``main_device`` to CPU and the forward runs there — a bf16 text encoder
    then encodes the prompt on the CPU. A non-zero GPU budget keeps execution
    on the GPU, and that slice stays resident for the whole denoise because
    the map is not streamed.

    Each text encoder therefore takes its full cast size when the denoiser
    still keeps at least ``_MIN_GPU_OFFLOAD_BYTES``, and otherwise one minimum
    shard so prompt encoding stays on the GPU without parking the whole
    encoder beside the denoiser. The denoiser gets what remains. A remainder
    below the minimum shard is raised when free VRAM can hold the shard;
    otherwise the entry stays 0 and the loader logs that the denoiser forward
    stays on the CPU.
    """
    assigned = {name: 0 for name in large}
    denoiser = _preferred_gpu_component(large)
    if denoiser is None:
        return assigned
    budget = _denoiser_gpu_budget(free_vram, small_bytes)
    encoders = sorted(
        (
            name
            for name in large
            if name.startswith("text_encoder") and name != denoiser
        ),
        key=lambda name: large[name],
    )
    for name in encoders:
        size = large[name]
        if budget - size >= _MIN_GPU_OFFLOAD_BYTES:
            assigned[name] = size
            budget -= size
        elif budget - _MIN_GPU_OFFLOAD_BYTES >= _MIN_GPU_OFFLOAD_BYTES:
            assigned[name] = _MIN_GPU_OFFLOAD_BYTES
            budget -= _MIN_GPU_OFFLOAD_BYTES
    if 0 < budget < _MIN_GPU_OFFLOAD_BYTES and int(free_vram or 0) >= _MIN_GPU_OFFLOAD_BYTES:
        budget = _MIN_GPU_OFFLOAD_BYTES
    assigned[denoiser] = max(0, budget)
    return assigned


def _cpu_disk_budgets(large: dict[str, int], available_ram: int, cuda_visible: bool) -> dict[str, dict]:
    """Equal CPU cap for every large component. GPU stays at 0 when it is visible.

    This is the resident share used before the CUDA tiers: about 28% of
    available RAM, split across the large components, and at least 1 GiB.
    A CUDA failure that reloads on CPU uses it, so the fallback does not keep
    the larger CUDA RAM share or place weights on the GPU.
    """
    count = max(len(large), 1)
    cpu_bytes = max(1024**3, int(available_ram * _CPU_ACCELERATE_RAM_FRACTION / count))
    budgets: dict[str, dict] = {}
    for name in large:
        entry: dict = {"cpu": cpu_bytes}
        if cuda_visible:
            entry[0] = 0
        budgets[name] = entry
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

    CUDA gives the denoiser most of the spare VRAM and a text encoder enough
    to run prompt encoding on the GPU (see :func:`_disk_component_gpu_budgets`).
    CPU loads use the equal 0.28 split and publish a zero GPU budget when CUDA
    is visible, so a fallback after an out-of-memory error does not place
    weights on the GPU.
    """
    large, small = _partition_components(component_bytes)
    if not large:
        return {}
    if device != "cuda":
        return _cpu_disk_budgets(large, available_ram, cuda_visible)
    cpu_budgets = _split_cpu_budgets(large, int(available_ram * _RAM_RESIDENT_FRACTION))
    gpu_budgets = _disk_component_gpu_budgets(large, small, free_vram)
    budgets: dict[str, dict] = {}
    for name, cpu_bytes in cpu_budgets.items():
        budgets[name] = {"cpu": cpu_bytes, 0: gpu_budgets.get(name, 0)}
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


def _format_step_timing(
    steps: int,
    wall: float,
    encode: float | None = None,
    denoise: float | None = None,
    decode: float | None = None,
) -> str:
    """Wall time of one generation, plus a host-side phase split when known.

    The split is prompt encode (start until the first denoising step returns),
    denoise (first step return until the last), and decode (the remainder,
    which is the VAE). It is not an average of kernel time per step.
    """
    text = f"Text-to-image generation: {steps} denoising steps, wall {wall:.1f}s"
    if encode is not None and denoise is not None and decode is not None:
        text += (
            f" (prompt encode {encode:.2f}s, denoise {denoise:.2f}s, decode {decode:.2f}s)"
        )
    return text + "."


def select_compute_dtype_name(device: str, *, bf16_native: bool) -> str:
    """``float32`` on CPU, hardware ``bfloat16`` on CUDA, otherwise ``float16``."""
    if device == "cpu":
        return "float32"
    if device == "cuda" and bf16_native:
        return "bfloat16"
    return "float16"


def _cuda_bf16_without_emulation(torch_mod) -> bool:
    """True when CUDA bf16 does not depend on software emulation.

    ``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True``,
    so Turing (compute capability 7.5) reports support. Prefer the flag that
    excludes emulation. An older signature falls back to capability 8 or newer.
    """
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None:
        return False
    supported = getattr(cuda, "is_bf16_supported", None)
    if callable(supported):
        try:
            return bool(supported(including_emulation=False))
        except TypeError:
            pass
    capability = getattr(cuda, "get_device_capability", None)
    if not callable(capability):
        return False
    try:
        major, _minor = capability()
    except Exception:
        return False
    return int(major) >= 8


def _group_offload_kwargs(offload_type: str, *, pre_pin: bool, use_stream: bool = True) -> dict:
    """Keyword arguments for diffusers 0.41 ``apply_group_offloading``.

    Streams require ``num_blocks_per_group`` of 1; a larger group is forced
    back to 1 and disables the overlap the stream was opened for. Disk
    offload is intentionally absent: see ``_load_disk_offloaded``. Streams
    also require a CUDA onload device, so a CPU onload passes ``use_stream``
    false. ``low_cpu_mem_usage`` only changes streamed pinning: false pre-pins
    the host copy, true pins each group when it is copied.
    """
    streaming = bool(use_stream)
    return {
        "offload_type": offload_type,
        "num_blocks_per_group": 1 if offload_type == "block_level" else None,
        "non_blocking": streaming,
        "use_stream": streaming,
        "record_stream": streaming,
        "low_cpu_mem_usage": (not pre_pin) if streaming else False,
    }


def _empty_host_cache() -> None:
    """Release pinned host blocks when this torch build exposes the call."""
    try:
        import torch
    except ImportError:
        return
    accelerator = getattr(torch, "accelerator", None)
    empty_host = getattr(accelerator, "empty_host_cache", None)
    if not callable(empty_host):
        return
    try:
        empty_host()
    except Exception:
        pass


def _empty_cuda_cache() -> None:
    _empty_host_cache()
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
    """Call *attempt(tier)* until one returns. Drop a tier on OOM or if it is unsupported.

    The exception is not kept across attempts. Its traceback holds the failed
    pipeline in the attempt's frame locals; storing the exception, or loading
    the next tier inside the ``except``, keeps that pipeline alive while the
    next one is measured. Frames are cleared before the next attempt, then
    ``gc.collect()`` runs after the ``except`` has released the exception.
    """
    last_summary = "No text-to-image placement was attempted."
    for index, tier in enumerate(tiers):
        generation = _active_offload_generation
        try:
            return attempt(tier)
        except Exception as exc:
            last_summary = _placement_failure_summary(exc)
            drop = index < len(tiers) - 1 and _should_drop_placement(exc)
            created = _active_offload_generation != generation
            if not drop:
                if created:
                    release_disk_offload()
                raise
            nxt = tiers[index + 1]
            if on_drop is not None:
                on_drop(tier, nxt)
            print(
                f"Text-to-image placement {tier} did not fit "
                f"({last_summary}); trying {nxt}.",
                file=sys.stderr,
                flush=True,
            )
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
        if created:
            release_disk_offload()
        gc.collect()
        _empty_cuda_cache()
    raise RuntimeError(last_summary)


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

    On CUDA the denoiser gets free VRAM minus the activation headroom, and each
    text encoder gets a GPU budget so prompt encoding is not a bf16 CPU
    forward. That GPU slice stays resident during denoising. See
    ``_disk_component_gpu_budgets``.
    """
    from diffusers import DiffusionPipeline

    index = _read_model_index(model_path)
    root = Path(model_path)
    cast = measure_component_bytes(model_path, dtype)
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
        gpu_note = ""
        if device == "cuda":
            if gpu_budget < _MIN_GPU_OFFLOAD_BYTES:
                gpu_note = (
                    " Free VRAM is below the minimum denoiser shard, so the "
                    "denoiser forward stays on CPU."
                )
            else:
                gpu_note = f" Denoiser GPU budget {_format_gib(gpu_budget)}."
            encoder_notes = [
                f"{name} {_format_gib(entry.get(0, 0))}"
                for name, entry in sorted(budgets.items())
                if name.startswith("text_encoder") and entry.get(0, 0) > 0
            ]
            if encoder_notes:
                gpu_note += " Text encoder GPU budget " + ", ".join(encoder_notes) + "."
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


def _component_has_block_modules(module) -> bool:
    """True when diffusers block offload would split this module.

    Block offload only groups direct ``ModuleList`` / ``Sequential`` children
    and names listed in ``_group_offload_block_modules``. Without those, the
    hook wraps ``forward`` alone. ``decode`` and ``encode`` never call
    ``forward``, so a VAE stays on the offload device while its inputs are on
    the onload device. The attribute is defined on AutoencoderKL and two other
    autoencoders, not on the other VAEs or on transformers text encoders.
    """
    names = getattr(module, "_group_offload_block_modules", None)
    if names:
        return True
    try:
        import torch
    except ImportError:
        return False
    children = getattr(module, "named_children", None)
    if not callable(children):
        return False
    for _name, child in children():
        if isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)) and len(child) > 0:
            return True
    return False


def _offload_type_for_component(name: str, module, requested: str) -> str:
    """Leaf groups for everything except a denoiser that actually has blocks.

    Text encoders and the VAE run once per prompt, so leaf groups do not change
    step time, and they onload the ``Linear`` / ``Conv`` that ``decode`` and
    ``encode`` call. A denoiser with no block modules cannot use block offload
    (the hook would miss every path that does not call ``forward``), so it uses
    leaf groups as well. ``block_level`` stays the request only for a denoiser
    diffusers can split.
    """
    if requested == "block_level" and _is_denoiser_component(name) and _component_has_block_modules(module):
        return "block_level"
    return "leaf_level"


def _apply_group_offload(pipe, offload_type: str, *, pre_pin: bool, onload_device=None) -> None:
    """Prefetch groups onto *onload_device*. Every module is included.

    ``enable_group_offload(..., exclude_modules=...)`` moves excluded modules
    onto the onload device and leaves them there. A text encoder excluded that
    way would stay resident through denoising. Applying the hook to every
    ``nn.Module`` keeps it on the onload device only during its own forward.
    Non-denoisers, and a denoiser without block modules, use leaf groups even
    when *offload_type* is ``block_level``.

    A component that cannot take the hook raises ``_PlacementUnsupported`` so
    the next tier is tried. Leaving it on CPU would surface as a device
    mismatch only when generation runs.
    """
    try:
        from diffusers.hooks import apply_group_offloading
    except ImportError as exc:
        raise _PlacementUnsupported("diffusers group offload is unavailable") from exc
    import torch

    components = getattr(pipe, "components", None)
    if not isinstance(components, dict):
        raise _PlacementUnsupported("pipeline has no components")
    onload = torch.device("cuda") if onload_device is None else torch.device(onload_device)
    offload = torch.device("cpu")
    use_stream = onload.type == "cuda"
    applied = 0
    choices: list[str] = []
    for name, component in components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        chosen = _offload_type_for_component(name, component, offload_type)
        choices.append(f"{name}={chosen}")
        kwargs = _group_offload_kwargs(chosen, pre_pin=pre_pin, use_stream=use_stream)
        try:
            apply_group_offloading(
                component,
                onload_device=onload,
                offload_device=offload,
                **kwargs,
            )
            applied += 1
        except Exception as exc:
            if _is_cuda_oom_error(exc) or isinstance(exc, MemoryError):
                raise
            raise _PlacementUnsupported(
                f"{name} group offload failed: {_placement_failure_summary(exc)}"
            ) from exc
    if applied == 0:
        raise _PlacementUnsupported("pipeline has no modules to group-offload")
    print(
        "Text-to-image group offload: " + ", ".join(choices) + ".",
        file=sys.stderr,
        flush=True,
    )


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
    pre_pin: bool | None = None,
):
    """Load *model_path* and apply one CUDA tier. The caller retries on failure.

    *pre_pin* overrides the group-offload pin choice. ``False`` sets
    ``low_cpu_mem_usage=True``, which pins each group as it is copied instead
    of pinning the whole host duplicate up front.
    """
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
            if pre_pin is None:
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
    """Load a CUDA pipeline, dropping a tier when the choice does not fit.

    A group-offload tier that pinned host memory and then ran out of GPU memory
    is retried once on the same tier with ``pre_pin=False`` before the next
    tier. Pinning failures are reported as CUDA out of memory. The retry pins
    on the fly (``low_cpu_mem_usage=True``), which costs the stream overlap a
    pre-pinned duplicate would have given. The tier cap recorded for an
    out-of-memory drop is cleared after this load succeeds so a later load of
    the same model measures again. The cap still applies to this call: it was
    read into ``tiers`` before the attempts.
    """
    resolved = str(Path(model_path).resolve())
    cast = measure_component_bytes(model_path, dtype)
    preferred = select_cuda_tier(
        cast,
        free_vram or 0,
        available_ram,
        dtype,
    )
    tiers = cuda_tiers_from(preferred, _cuda_tier_limits.get(resolved))
    dtype_name = _dtype_name(dtype)

    def publish(pipe, tier: str):
        _mark_cuda_pipe(pipe, tier, resolved)
        print(
            _format_placement_log(tier, dtype_name, free_vram, available_ram, cast),
            file=sys.stderr,
            flush=True,
        )
        return pipe

    def attempt(tier: str):
        pre_pin = None
        if tier in (TIER_GROUP, TIER_GROUP_LEAF):
            pre_pin = group_offload_pre_pin(sum(cast.values()), available_ram)
        try:
            pipe = _place_cuda_tier(
                model_path,
                dtype,
                tier,
                cast,
                available_ram=available_ram,
                free_vram=free_vram,
                pre_pin=pre_pin,
            )
        except Exception as exc:
            if not pre_pin or not _is_cuda_oom_error(exc):
                raise
            print(
                f"Text-to-image placement {tier} ran out of memory while pinning "
                "host weights; retrying the same tier with low_cpu_mem_usage=True. "
                "Weights pin on the fly, so stream overlap costs an extra copy.",
                file=sys.stderr,
                flush=True,
            )
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
        else:
            return publish(pipe, tier)
        gc.collect()
        _empty_cuda_cache()
        pipe = _place_cuda_tier(
            model_path,
            dtype,
            tier,
            cast,
            available_ram=available_ram,
            free_vram=free_vram,
            pre_pin=False,
        )
        return publish(pipe, tier)

    def on_drop(failed: str, _nxt: str) -> None:
        _note_tier_limit(resolved, failed)

    pipe = _run_placement_attempts(tiers, attempt, on_drop)
    clear_cuda_tier_limit(resolved)
    return pipe


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

    # float32 on CPU. bfloat16 on CUDA only when the GPU runs it in hardware
    # (compute capability 8 or newer). ``is_bf16_supported()`` defaults to
    # counting software emulation, which is true on Turing (7.5) and would
    # pick bf16 there. float16 otherwise. Steps, size, seed, and CFG stay at
    # the pipeline defaults.
    dtype_name = select_compute_dtype_name(
        device, bf16_native=_cuda_bf16_without_emulation(torch)
    )
    dtype = getattr(torch, dtype_name)

    available = _available_ram_bytes()
    generation = _active_offload_generation
    resident = sum(measure_component_bytes(model_path, dtype).values())
    if device == "cuda":
        pipe = _load_cuda_pipeline(model_path, dtype, available, _free_vram_bytes())
    elif _resident_exceeds(device, available, resident):
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


def _sync_timing_device() -> None:
    """Wait for queued CUDA work so a timestamp is not only the launch time."""
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    try:
        cuda.synchronize()
    except Exception:
        pass


class _StepTimer:
    """Wall clock around one ``pipe()`` call, with a count of denoising steps.

    The first mark is the end of prompt encode plus the first step. The last
    mark is the end of denoising. Whatever remains until :meth:`log` is the
    VAE decode. CUDA is synchronized on the first mark and by the caller
    before :meth:`log`, not on every step, so group-offload streams are not
    drained once per step just to print the split.
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.steps = 0
        self.first_mark: float | None = None
        self.last_mark: float | None = None

    def mark(self, step_index=None) -> None:
        if self.first_mark is None:
            _sync_timing_device()
            self.first_mark = time.perf_counter()
        self.last_mark = time.perf_counter()
        if isinstance(step_index, int) and step_index >= 0:
            self.steps = max(self.steps, step_index + 1)
        else:
            self.steps += 1

    def log(self) -> None:
        if self.steps <= 0:
            return
        ended = time.perf_counter()
        encode = denoise = decode = None
        if self.first_mark is not None and self.last_mark is not None:
            encode = max(0.0, self.first_mark - self.started)
            denoise = max(0.0, self.last_mark - self.first_mark)
            decode = max(0.0, ended - self.last_mark)
        print(
            _format_step_timing(self.steps, ended - self.started, encode, denoise, decode),
            file=sys.stderr,
            flush=True,
        )


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
    is not saved. Stderr then reports the denoising step count, the wall time
    of the pipeline call, and a prompt-encode / denoise / decode split.
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
    _sync_timing_device()
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
        _sync_timing_device()
        timer.log()
