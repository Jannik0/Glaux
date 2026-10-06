"""Diffusers text-to-image and image-to-image loading and one-shot generation.

``worker.chat`` branches here for those pipeline tags. Inpaint and
unconditional pipelines are rejected. An image-to-image load may use an
img2img class. Callers pass only the current prompt and, for image-to-image,
an optional init image. Size, steps, seed, and CFG stay at the pipeline
defaults.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

_REJECTED_CLASS_MARKERS = ("inpaint", "unconditional")
# A component this large is offloaded to disk instead of staged in RAM.
_DISK_OFFLOAD_MIN_BYTES = 1536 * 1024 * 1024


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


def _disk_offload_budgets(large_count: int) -> dict:
    """Cap how much of each large component stays resident.

    The rest is written to disk and loaded for the forward pass. Budgets are
    split across every large component so they can be resident together.
    """
    import torch

    available = _available_ram_bytes() or (6 * 1024**3)
    cpu_bytes = max(1024**3, int(available * 0.28 / max(large_count, 1)))
    gpu_bytes = 1024**3
    if torch.cuda.is_available():
        free_vram, _total = torch.cuda.mem_get_info()
        gpu_bytes = max(512 * 1024**2, min(2 * 1024**3, int(free_vram * 0.35)))
    return {0: gpu_bytes, "cpu": cpu_bytes}


def _load_disk_offloaded(model_path: str, dtype):
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
    budgets = _disk_offload_budgets(len(large))
    offload_root = Path(tempfile.gettempdir()) / "glaux-t2i-offload"
    if offload_root.exists():
        shutil.rmtree(offload_root, ignore_errors=True)
    offload_root.mkdir(parents=True, exist_ok=True)
    print(
        "Text-to-image weights are larger than available RAM; offloading them to disk.",
        file=sys.stderr,
        flush=True,
    )
    preloaded = {}
    for name in large:
        library, class_name = index[name]
        component_cls = _import_component_class(library, class_name)
        preloaded[name] = component_cls.from_pretrained(
            root / name,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=budgets,
            offload_folder=str(offload_root / name),
            offload_state_dict=True,
        )
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
        **preloaded,
    )
    # Disk-offloaded modules move themselves during the forward pass. Smaller
    # ones (the VAE) stay on CPU unless they are placed on the GPU explicitly,
    # and the latents arriving from the transformer are already CUDA.
    import torch

    for component in pipe.components.values():
        if not isinstance(component, torch.nn.Module):
            continue
        if getattr(component, "hf_device_map", None):
            continue
        component.to("cuda")
    return pipe


def load_text_to_image_pipeline(model_path: str, device: str, *, allow_img2img: bool = False):
    """Load the diffusers pipeline named in ``model_index.json``.

    ``DiffusionPipeline`` follows ``_class_name``, so a class added after the
    last numbered diffusers release still loads. Inpaint classes are rejected
    by name. Img2img classes load only for image-to-image.

    Weights that fit in RAM stay there and stream to the GPU one layer at a
    time. Larger checkpoints are offloaded to disk.
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
    if device == "cuda" and available is not None and weights > int(available * 0.55):
        pipe = _load_disk_offloaded(model_path, dtype)
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
        raise RuntimeError(f"This diffusion class is not supported (loaded {loaded_name}).")
    return pipe


def generate_text_to_image(pipe, prompt: str, output_path: str, image_path: str | None = None) -> str:
    """Run the pipeline on ``prompt`` and write one image to ``output_path``.

    ``image_path`` is the optional init image for image-to-image. Strength
    and the other sampling settings stay at the pipeline defaults.

    Prompt and image are passed by name. Some pipelines, including Flux2,
    take the image as the first positional argument, so a positional prompt
    lands in that slot.
    """
    text = prompt.strip()
    if not text:
        raise ValueError("Image generation requires a prompt.")
    if not output_path:
        raise ValueError("Image generation output path is not configured.")
    call = {"prompt": text}
    if image_path:
        from PIL import Image

        with Image.open(image_path) as opened:
            call["image"] = opened.convert("RGB")
    result = pipe(**call)
    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError("Text-to-image pipeline returned no images.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination)
    return str(destination)
