# Third-party notices (packaged binaries)

Glaux source code is licensed under the MIT License — see [LICENSE](LICENSE).

A **packaged** Glaux build (installer, zip, deb, rpm, tar.gz, dmg, or unpacked `dist/` tree) also redistributes independently licensed components. Those licenses apply to the corresponding files, not to Glaux source. This notice is shipped next to the app as `THIRD_PARTY_LICENSES.md` (under Electron `extraResources`).

Staged third-party binaries are shipped unmodified. On Windows and Linux GPU builds, the NVIDIA NCCL, cuSPARSELt, NVSHMEM, and cuFile libraries are the exception: those binaries are not included. `npm run build:python` compiles tiny loader stubs that keep the original SONAME or DLL name so libtorch still loads. The stubs are Glaux build output, not NVIDIA code and not modified NVIDIA binaries. Hub **model weights** are downloaded by the user at runtime and are **not** part of Glaux; each model remains under its own Hub license.

Electron already writes Chromium and Electron license files into the install directory (`LICENSES.chromium.html`, `LICENSE.electron.txt`). Python wheels keep their license texts under `site-packages/*.dist-info`.


## Native inference engines

### llama.cpp

- **What:** `llama-server` and ggml backend modules under `vendor/llamacpp` (packaged as `resources/llamacpp`)
- **License:** MIT
- **Upstream:** https://github.com/ggml-org/llama.cpp

### transcribe.cpp

- **What:** `transcribe-cli` and ggml backend modules under `vendor/transcribe` (packaged as `resources/transcribe`). The CUDA backend file is the shared `libggml-cuda` / `ggml-cuda.dll` in `vendor/cuda` (stable-diffusion.cpp’s build when that engine is packaged, otherwise the llama.cpp copy). Vulkan and the other backends are the transcribe.cpp build.
- **License:** MIT
- **Upstream:** https://github.com/handy-computer/transcribe.cpp

### stable-diffusion.cpp

- **What:** `sd-cli` and ggml backend modules under `vendor/stablediffusion` (packaged as `resources/stablediffusion`). CUDA runtime libraries and this build’s ggml CUDA fatbin are the shared copies in `vendor/cuda`. llama.cpp and transcribe.cpp are compiled with the same `GGML_MAX_NAME=160` and load that fatbin from the sibling `cuda` directory.
- **License:** MIT
- **Upstream:** https://github.com/leejet/stable-diffusion.cpp at `a1ded76` (ggml submodule https://github.com/leejet/ggml at `89c4413`, also MIT; that ggml tree vendors no further libraries)

`npm run build:stablediffusion` copies the notice files named below into `vendor/stablediffusion`, next to `sd-cli`. `package.json` `extraResources` copies that directory to `resources/stablediffusion` with no filter, and `afterPack` does not remove them. The pieces below are compiled into `sd-cli` or the shared `stable-diffusion` library staged beside it. No separate library file ships for them.

**libwebp** is statically linked into `sd-cli` (libwebp, libwebpmux, and libsharpyuv).

- **License:** BSD-3-Clause. The upstream text is staged as `LIBWEBP.COPYING`. The WebM additional patent grant is staged as `LIBWEBP.PATENTS`.
- **Copyright:** Copyright (c) 2010, Google Inc. All rights reserved.
- **Upstream:** https://github.com/webmproject/libwebp at `0c9546f7efc61eac7f79ae115c3f99c91c21c443` (`LIBWEBP_REV` in `scripts/build-stablediffusion.js`)

```
Copyright (c) 2010, Google Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

**Oniguruma** 6.9.10 is linked into the shared `stable-diffusion` library.

- **License:** BSD-2-Clause. Staged as `ONIGURUMA.COPYING`.
- **Copyright:** Copyright (c) 2002-2021 K.Kosako
- **Upstream:** https://github.com/kkos/oniguruma at `4ef89209a239c1aea328cf13c05a2807e5c146d1`

```
Copyright (c) 2002-2021  K.Kosako  <kkosako0@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
```

**darts-clone** is compiled into the T5 unigram tokenizer in that same library.

- **License:** BSD-3-Clause. Staged as `DARTS.LICENSE` (upstream `thirdparty/LICENSE.darts_clone.txt`, including its `<ORGANIZATION>` placeholder).
- **Copyright:** Copyright (c) 2008-2011, Susumu Yata
- **Upstream:** https://github.com/google/sentencepiece (`third_party/darts_clone`)

```
Copyright (c) 2008-2011, Susumu Yata
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

**utf8proc** 2.10.0 is linked into the shared library. The same text is staged as `UTF8PROC.LICENSE.md`.

- **License:** MIT, plus the Unicode data license for `utf8proc_data.c`
- **Upstream:** https://github.com/JuliaStrings/utf8proc at `a1b99daa2a3393884220264c927a48ba1251a9c6` (Unicode 16.0.0)

```
utf8proc license

utf8proc is a software package originally developed
by Jan Behrens and the rest of the Public Software Group, who
deserve nearly all of the credit for this library, that is now maintained by the Julia-language developers.  Like the original utf8proc,
whose copyright and license statements are reproduced below, all new
work on the utf8proc library is licensed under the MIT "expat" license:

Copyright (c) 2014-2021 by Steven G. Johnson, Jiahao Chen, Tony Kelman, Jonas Fonseca, and other contributors listed in the git history.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Original utf8proc license

Copyright (c) 2009, 2013 Public Software Group e. V., Berlin, Germany

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Unicode data license

This software contains data (utf8proc_data.c) derived from processing
the Unicode data files. The following license applies to that data:

COPYRIGHT AND PERMISSION NOTICE

Copyright (c) 1991-2007 Unicode, Inc. All rights reserved. Distributed
under the Terms of Use in http://www.unicode.org/copyright.html.

Permission is hereby granted, free of charge, to any person obtaining a
copy of the Unicode data files and any associated documentation (the "Data
Files") or Unicode software and any associated documentation (the
"Software") to deal in the Data Files or Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, and/or sell copies of the Data Files or Software, and
to permit persons to whom the Data Files or Software are furnished to do
so, provided that (a) the above copyright notice(s) and this permission
notice appear with all copies of the Data Files or Software, (b) both the
above copyright notice(s) and this permission notice appear in associated
documentation, and (c) there is clear notice in each modified Data File or
in the Software as well as in the documentation associated with the Data
File(s) or Software that the data or software has been modified.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS
INCLUDED IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR
CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THE DATA FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.

Unicode and the Unicode logo are trademarks of Unicode, Inc., and may be
registered in some jurisdictions. All other trademarks and registered
trademarks mentioned herein are the property of their respective owners.
```

**nlohmann/json** 3.11.2 is compiled into `sd-cli` and the shared library.

- **License:** MIT
- **Upstream:** https://github.com/nlohmann/json at `v3.11.2`

```
MIT License

Copyright (c) 2013-2022 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**zip** is compiled into `sd-cli` and the shared library. The upstream license file has no separate copyright line.

- **License:** MIT
- **Upstream:** https://github.com/kuba--/zip

```
All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

**stb_image** 2.28, **stb_image_write** 1.16, and **stb_image_resize** 0.90 are compiled into `sd-cli`.

- **License:** public domain
- **Upstream:** https://github.com/nothings/stb

**miniz** 2.2.0 is compiled into that zip reader.

- **License:** The Unlicense (public domain)
- **Upstream:** https://github.com/richgel999/miniz


## FFmpeg / ffprobe / dav1d

- **What:** shared `ffmpeg` and `ffprobe` plus `libav*` / `libdav1d` under `vendor/ffmpeg` (packaged as `resources/ffmpeg`), built by `npm run build:ffmpeg` from FFmpeg **7.1.1** and dav1d **1.5.1**
- **How Glaux uses them:** as **separate processes** (not linked into the Glaux executable). llama.cpp, transcribe.cpp, and the Hugging Face worker all spawn these binaries.
- **License:** FFmpeg is **LGPL 2.1 or later** in this decode-only shared build (no libx264/libx265 or other GPL-only encoders). dav1d is **BSD-2-Clause**. License texts are staged next to the binaries (`COPYING.LGPLv2.1`, `DAV1D.COPYING`).
- **Upstream source:** https://ffmpeg.org (tag `n7.1.1`) and https://code.videolan.org/videolan/dav1d (tag `1.5.1`). Configure flags live in `scripts/build-ffmpeg.js`.

If you redistribute Glaux installers that include these binaries, you must preserve the FFmpeg and dav1d license texts and offer corresponding source for that unmodified FFmpeg 7.1.1 / dav1d 1.5.1 build (the links above). Omit `vendor/ffmpeg` if you need a tree without bundled FFmpeg.


## NVIDIA CUDA redistributables

On Windows and Linux GPU builds, Glaux copies CUDA **runtime** libraries (not the driver) once into `vendor/cuda` (packaged as `resources/cuda`). PyTorch, `llama-server`, `transcribe-cli`, and `sd-cli` all load that shared CUDA 13 folder:

- `cudart`, `cublas`, `cublasLt`, `nvJitLink` (`.dll` on Windows, `.so` on Linux)

The same folder also holds one ggml CUDA backend module (`ggml-cuda.dll` / `libggml-cuda.so`). That file is the stable-diffusion.cpp build (MIT), not an NVIDIA redistributable. The three GGUF engines load it from this directory.

These files are NVIDIA proprietary software, redistributed under the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) (redistributable subset). They are not licensed under MIT. End users still need a current NVIDIA **driver** for CUDA inference; the Toolkit itself is not required on the end-user machine.

PyTorch CUDA wheels in `vendor/python` still include the NVIDIA components libtorch calls for single-GPU inference (cuDNN, cuFFT, nvrtc, cuSOLVER, cuSPARSE, CUPTI), covered by the same family of NVIDIA terms. PyTorch’s own `libtorch_nvshmem.so` stays; that file is part of the PyTorch wheel, not the NVIDIA NVSHMEM library. Overlapping CUDA 13 runtime libraries are replaced with links to `vendor/cuda` (from both `torch/lib` and `nvidia/cu13`) so they are not shipped twice.

The NVIDIA NCCL, cuSPARSELt, NVSHMEM, and cuFile binaries are not shipped. On Windows and Linux, Glaux replaces those shared libraries with the loader stubs described above. NVSHMEM device bitcode and bootstrap plugins are deleted. The wheel `*.dist-info` directories for those four packages, including their NVIDIA license texts, remain under `site-packages`. Headers, static libs, NVTX, `nvperf`, `cusolverMg`, and `nvrtc*.alt` are omitted from the packaged tree.


## Python runtime and ML stack

- **CPython** from [python-build-standalone](https://github.com/astral-sh/python-build-standalone) (PSF License for CPython; see that project for packaging terms)
- **PyTorch** / **TorchVision** — BSD-style license (https://github.com/pytorch/pytorch)
- **Hugging Face Transformers**, **Diffusers**, **huggingface_hub**, **Accelerate**, **safetensors**, and related Hub client libraries — Apache License 2.0.
- Other pinned packages from `engines/huggingface/requirements.txt` and their transitive dependencies — licenses are in each wheel’s `*.dist-info`

The Hugging Face engine also uses **DOMPurify**, **marked**, and **pdf-parse** from the Electron `package.json` (Apache-2.0 / MPL-2.0, MIT, and MIT respectively) for renderer-side HTML sanitization, markdown, and PDF text extraction.


## Electron

- **Electron** — MIT
- **Chromium** and other bundled libraries — see `LICENSES.chromium.html` in the packaged app


## Vulkan / Metal / MPS

Glaux does not ship the Vulkan or Metal drivers. ggml Vulkan/Metal backends come from llama.cpp / transcribe.cpp / stable-diffusion.cpp (MIT). PyTorch MPS uses Apple’s system frameworks. Diffusers on Apple Silicon uses that same MPS path.
