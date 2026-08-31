# Third-party notices (packaged binaries)

Glaux source code is licensed under the MIT License — see [LICENSE](LICENSE).

A **packaged** Glaux build (installer, zip, dmg, AppImage, or unpacked `dist/` tree) also redistributes independently licensed components. Those licenses apply to the corresponding files, not to Glaux source. This notice is shipped next to the app as `THIRD_PARTY_LICENSES.md` (under Electron `extraResources`).

Glaux does not modify the third-party binaries it stages. Hub **model weights** are downloaded by the user at runtime and are **not** part of Glaux; each model remains under its own Hub license (for example Google Gemma terms).

Electron already writes Chromium and Electron license files into the install directory (`LICENSES.chromium.html`, `LICENSE.electron.txt`). Python wheels keep their license texts under `site-packages/*.dist-info`.


## Native inference engines

### llama.cpp

- **What:** `llama-server` and ggml backend modules under `vendor/llamacpp` (packaged as `resources/llamacpp`)
- **License:** MIT
- **Upstream:** https://github.com/ggml-org/llama.cpp

### transcribe.cpp

- **What:** `transcribe-cli` and ggml backend modules under `vendor/transcribe` (packaged as `resources/transcribe`)
- **License:** MIT
- **Upstream:** https://github.com/handy-computer/transcribe.cpp


## FFmpeg / ffprobe / dav1d

- **What:** shared `ffmpeg` and `ffprobe` plus `libav*` / `libdav1d` under `vendor/ffmpeg` (packaged as `resources/ffmpeg`), built by `npm run build:ffmpeg` from FFmpeg **7.1.1** and dav1d **1.5.1**
- **How Glaux uses them:** as **separate processes** (not linked into the Glaux executable). llama.cpp, transcribe.cpp, and the Hugging Face worker all spawn these binaries.
- **License:** FFmpeg is **LGPL 2.1 or later** in this decode-only shared build (no libx264/libx265 or other GPL-only encoders). dav1d is **BSD-2-Clause**. License texts are staged next to the binaries (`COPYING.LGPLv2.1`, `DAV1D.COPYING`).
- **Upstream source:** https://ffmpeg.org (tag `n7.1.1`) and https://code.videolan.org/videolan/dav1d (tag `1.5.1`). Configure flags live in `scripts/build-ffmpeg.js`.

If you redistribute Glaux installers that include these binaries, you must preserve the FFmpeg and dav1d license texts and offer corresponding source for that unmodified FFmpeg 7.1.1 / dav1d 1.5.1 build (the links above). Omit `vendor/ffmpeg` if you need a tree without bundled FFmpeg.


## NVIDIA CUDA redistributables

On Windows and Linux GPU builds, Glaux copies CUDA **runtime** libraries (not the driver) once into `vendor/cuda` (packaged as `resources/cuda`). PyTorch, `llama-server`, and `transcribe-cli` all load that shared CUDA 13 folder:

- `cudart`, `cublas`, `cublasLt`, `nvJitLink` (`.dll` on Windows, `.so` on Linux)

These files are NVIDIA proprietary software, redistributed under the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) (redistributable subset). They are not licensed under MIT. End users still need a current NVIDIA **driver** for CUDA inference; the Toolkit itself is not required on the end-user machine.

PyTorch CUDA wheels in `vendor/python` still include Torch-only NVIDIA components (for example cuDNN, cuFFT, nvrtc) covered by the same family of NVIDIA terms. Overlapping CUDA 13 runtime libraries are stripped from `torch/lib` so they are not shipped twice.


## Python runtime and ML stack

- **CPython** from [python-build-standalone](https://github.com/astral-sh/python-build-standalone) (PSF License for CPython; see that project for packaging terms)
- **PyTorch** / **TorchVision** — BSD-style license (https://github.com/pytorch/pytorch)
- **Hugging Face Transformers**, **huggingface_hub**, **Accelerate**, **safetensors**, and related Hub client libraries — Apache License 2.0
- Other pinned packages from `engines/huggingface/requirements.txt` and their transitive dependencies — licenses are in each wheel’s `*.dist-info`

The Hugging Face engine also uses **DOMPurify**, **marked**, and **pdf-parse** from the Electron `package.json` (Apache-2.0 / MPL-2.0, MIT, and MIT respectively) for renderer-side HTML sanitization, markdown, and PDF text extraction.


## Electron

- **Electron** — MIT
- **Chromium** and other bundled libraries — see `LICENSES.chromium.html` in the packaged app


## Vulkan / Metal / MPS

Glaux does not ship the Vulkan or Metal drivers. ggml Vulkan/Metal backends come from llama.cpp / transcribe.cpp (MIT). PyTorch MPS uses Apple’s system frameworks.
