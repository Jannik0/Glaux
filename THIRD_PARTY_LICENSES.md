# Third-party notices (packaged binaries)

Glaux source code is licensed under the MIT License — see [LICENSE](LICENSE).

A **packaged** Glaux build (installer, zip, dmg, deb, rpm, tar.gz, or unpacked `dist/` tree) also redistributes independently licensed components. Those licenses apply to the corresponding files, not to Glaux source. This notice is shipped next to the app as `THIRD_PARTY_LICENSES.md` (under Electron `extraResources`).

Staged third-party binaries are shipped unmodified. On Windows and Linux GPU builds, the NVIDIA NCCL, cuSPARSELt, NVSHMEM, and cuFile libraries are the exception: those binaries are not included. `npm run build:python` compiles tiny loader stubs that keep the original SONAME or DLL name so libtorch still loads. The stubs are Glaux build output, not NVIDIA code and not modified NVIDIA binaries. Hub **model weights** are downloaded by the user at runtime and are **not** part of Glaux; each model remains under its own Hub license (for example Google Gemma terms).

Electron already writes Chromium and Electron license files into the install directory (`LICENSES.chromium.html`, `LICENSE.electron.txt`). Python wheels keep their license texts under `site-packages/*.dist-info`.


## Native inference engines

### llama.cpp

- **What:** `llama-server` and ggml backend modules under `vendor/llamacpp` (packaged as `resources/llamacpp`)
- **License:** MIT
- **Upstream:** https://github.com/ggml-org/llama.cpp

### transcribe.cpp

- **What:** `transcribe-cli` and ggml backend modules under `vendor/transcribe` (packaged as `resources/transcribe`). The CUDA backend file is a link to llama.cpp’s `libggml-cuda` / `ggml-cuda.dll` in `vendor/llamacpp`. Vulkan and the other backends are the transcribe.cpp build.
- **License:** MIT (the shared CUDA module is the llama.cpp binary, also MIT)
- **Upstream:** https://github.com/handy-computer/transcribe.cpp

### stable-diffusion.cpp

- **What:** `sd-cli` and ggml backend modules under `vendor/stablediffusion` (packaged as `resources/stablediffusion`). CUDA runtime libraries are the shared copies in `vendor/cuda`. The ggml CUDA fatbin is this build’s own file, not a link to llama.cpp: stable-diffusion.cpp sets `GGML_MAX_NAME=160`, which changes the ggml tensor layout versus llama.cpp and transcribe.cpp (default 64).
- **License:** MIT
- **Upstream:** https://github.com/leejet/stable-diffusion.cpp (ggml submodule https://github.com/ggml-org/ggml, also MIT)


## FFmpeg / ffprobe / dav1d

- **What:** shared `ffmpeg` and `ffprobe` plus `libav*` / `libdav1d` under `vendor/ffmpeg` (packaged as `resources/ffmpeg`), built by `npm run build:ffmpeg` from FFmpeg **7.1.1** and dav1d **1.5.1**
- **How Glaux uses them:** as **separate processes** (not linked into the Glaux executable). llama.cpp, transcribe.cpp, and the Hugging Face worker all spawn these binaries. Text-to-image does not.
- **License:** FFmpeg is **LGPL 2.1 or later** in this decode-only shared build (no libx264/libx265 or other GPL-only encoders). dav1d is **BSD-2-Clause**. License texts are staged next to the binaries (`COPYING.LGPLv2.1`, `DAV1D.COPYING`).
- **Upstream source:** https://ffmpeg.org (tag `n7.1.1`) and https://code.videolan.org/videolan/dav1d (tag `1.5.1`). Configure flags live in `scripts/build-ffmpeg.js`.

If you redistribute Glaux installers that include these binaries, you must preserve the FFmpeg and dav1d license texts and offer corresponding source for that unmodified FFmpeg 7.1.1 / dav1d 1.5.1 build (the links above). Omit `vendor/ffmpeg` if you need a tree without bundled FFmpeg.


## NVIDIA CUDA redistributables

On Windows and Linux GPU builds, Glaux copies CUDA **runtime** libraries (not the driver) once into `vendor/cuda` (packaged as `resources/cuda`). PyTorch, `llama-server`, `transcribe-cli`, and `sd-cli` all load that shared CUDA 13 folder:

- `cudart`, `cublas`, `cublasLt`, `nvJitLink` (`.dll` on Windows, `.so` on Linux)

These files are NVIDIA proprietary software, redistributed under the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) (redistributable subset). They are not licensed under MIT. End users still need a current NVIDIA **driver** for CUDA inference; the Toolkit itself is not required on the end-user machine.

PyTorch CUDA wheels in `vendor/python` still include the NVIDIA components libtorch calls for single-GPU inference (cuDNN, cuFFT, nvrtc, cuSOLVER, cuSPARSE, CUPTI), covered by the same family of NVIDIA terms. PyTorch’s own `libtorch_nvshmem.so` stays; that file is part of the PyTorch wheel, not the NVIDIA NVSHMEM library. Overlapping CUDA 13 runtime libraries are replaced with links to `vendor/cuda` (from both `torch/lib` and `nvidia/cu13`) so they are not shipped twice.

The NVIDIA NCCL, cuSPARSELt, NVSHMEM, and cuFile binaries are not shipped. On Windows and Linux, Glaux replaces those shared libraries with the loader stubs described above. NVSHMEM device bitcode and bootstrap plugins are deleted. The wheel `*.dist-info` directories for those four packages, including their NVIDIA license texts, remain under `site-packages`. Headers, static libs, NVTX, `nvperf`, `cusolverMg`, and `nvrtc*.alt` are omitted from the packaged tree.


## Python runtime and ML stack

- **CPython** from [python-build-standalone](https://github.com/astral-sh/python-build-standalone) (PSF License for CPython; see that project for packaging terms)
- **PyTorch** / **TorchVision** — BSD-style license (https://github.com/pytorch/pytorch)
- **Hugging Face Transformers**, **Diffusers**, **huggingface_hub**, **Accelerate**, **safetensors**, and related Hub client libraries — Apache License 2.0. Diffusers is pinned at 0.35.2 (the release transformers 5.17.0 declares) for safetensors text-to-image.
- Other pinned packages from `engines/huggingface/requirements.txt` and their transitive dependencies — licenses are in each wheel’s `*.dist-info`

The Hugging Face engine also uses **DOMPurify**, **marked**, and **pdf-parse** from the Electron `package.json` (Apache-2.0 / MPL-2.0, MIT, and MIT respectively) for renderer-side HTML sanitization, markdown, and PDF text extraction.


## Electron

- **Electron** — MIT
- **Chromium** and other bundled libraries — see `LICENSES.chromium.html` in the packaged app


## Vulkan / Metal / MPS

Glaux does not ship the Vulkan or Metal drivers. ggml Vulkan/Metal backends come from llama.cpp / transcribe.cpp / stable-diffusion.cpp (MIT). PyTorch MPS uses Apple’s system frameworks. Diffusers text-to-image on Apple Silicon uses that same MPS path.
