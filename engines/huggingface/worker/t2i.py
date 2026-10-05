"""Diffusers text-to-image loading and one-shot generation.

``worker.chat`` branches here when ``pipeline_tag`` is ``text-to-image``.
Img2img, inpaint, and unconditional pipelines are rejected. Callers pass only
the current prompt — this module never reads chat history. Size, steps, seed,
and CFG stay at the pipeline defaults.
"""

import json
from pathlib import Path

_REJECTED_CLASS_MARKERS = ("img2img", "inpaint", "unconditional")


def _class_name_rejected(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(marker in lowered for marker in _REJECTED_CLASS_MARKERS)


def assert_text_to_image_repo(model_path: str) -> str:
    """Read diffusers ``model_index.json`` and reject non-text-to-image classes."""
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
    if class_name and _class_name_rejected(class_name):
        raise RuntimeError(
            f"Only text-to-image is supported (model class is {class_name})."
        )
    return class_name


def load_text_to_image_pipeline(model_path: str, device: str):
    """Load ``AutoPipelineForText2Image`` on ``device`` (``cpu``, ``cuda``, or ``mps``)."""
    assert_text_to_image_repo(model_path)
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise RuntimeError(
            "diffusers is not installed. Rebuild the Python runtime (npm run build:python)."
        ) from exc
    import torch

    # float32 on CPU; float16 on GPU so the default pipeline fits typical VRAM.
    # Sampling steps, size, seed, and CFG are left at the pipeline defaults.
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    loaded_name = type(pipe).__name__
    if _class_name_rejected(loaded_name):
        raise RuntimeError(f"Only text-to-image is supported (loaded {loaded_name}).")
    return pipe.to(device)


def generate_text_to_image(pipe, prompt: str, output_path: str) -> str:
    """Run the pipeline on ``prompt`` alone and write one image to ``output_path``."""
    text = prompt.strip()
    if not text:
        raise ValueError("Text-to-image requires a prompt.")
    if not output_path:
        raise ValueError("Text-to-image output path is not configured.")
    result = pipe(text)
    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError("Text-to-image pipeline returned no images.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination)
    return str(destination)
