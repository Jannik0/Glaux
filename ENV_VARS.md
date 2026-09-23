# Environment variables

Glaux reads the following variables from the **process environment** of the app (or a `npm run build:*` / `npm run dist*` script). They must be visible to that process: `export VAR=value` then launch, or `VAR=value npm *`. Changing a variable while Glaux is already running has no effect until you restart.

**Packaged app:** export the variable in the shell (or system environment) that starts `Glaux.exe` / the `.app` / `glaux`, then launch.

## Runtime (app)

These are the variables the running app inspects. Unset means the documented default.


| Variable | Values | Effect |
| -------- | ------ | ------ |
| `GLAUX_FORCE_CPU` | `1`, `true`, `yes` (case-insensitive). Unset, `0`, `false`, and any other value leave GPU on. | Force CPU inference on every engine: Hugging Face pipelines pin `device="cpu"` and hide CUDA/HIP (`CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` cleared); llama-server is started with `--device none -ngl 0 --no-mmproj-offload`; transcribe-cli uses `--backend cpu` instead of `auto`. Not a build flag — GPU backends are still shipped. Read when a model is loaded or a transcription starts. |
| `GLAUX_LLAMA_CTX` | Positive integer (token count), e.g. `8192`. Unset, `0`, negative, or non-numeric → ignored. | Pins llama-server `--ctx-size` (`-c`) to that window. When unset, `-c` is omitted so `--fit` can keep the model’s trained context or shrink it (floor 4096) to stay on GPU. An explicit value is not shrunk by `--fit`. Chat GGUFs only. |
| `GLAUX_DOWNLOAD_MAX_WORKERS` | Positive integer. Default `16`. | Concurrent HTTP file downloads when fetching a Hub snapshot (sharded checkpoints, tokenizer files, GGUF variants). Read when the Hugging Face downloader module loads. |
| `GLAUX_LLAMA_DEBUG` | Exactly `1`. Any other value (including `true` / `yes`) is ignored. | Writes llama-server stderr to the Electron process stderr (`[llama-server] …`) and logs llama.cpp context-usage failures. Development / troubleshooting. |
| `PYTHON` | Absolute path to a Python interpreter. | Hugging Face worker interpreter. Resolution order: `PYTHON` if set, then bundled `vendor/python` (or packaged `resources/python`), then `python` / `python3` on `PATH`. |
| `GLAUX_LLAMA_SERVER` | Absolute path to a `llama-server` binary. | Overrides the bundled `vendor/llamacpp` (or packaged `resources/llamacpp`) binary. Useful in development with a locally built llama.cpp. |
| `GLAUX_TRANSCRIBE_CLI` | Absolute path to a `transcribe-cli` binary. | Overrides the bundled `vendor/transcribe` (or packaged `resources/transcribe`) binary. Useful in development with a locally built transcribe.cpp. |
| `LC_ALL`, `LANG` | Locale string, e.g. `en_US.UTF-8`. | Fallback OS language when Electron’s locale APIs are unavailable. The in-app language preference (Settings / `preferences.json`) takes precedence. |

## Hugging Face Hub

The Python worker inherits the full process environment, so Hub libraries can also see tokens and their own settings:


| Variable | Values | Effect |
| -------- | ------ | ------ |
| `HF_TOKEN` | Hugging Face access token (`hf_…`). | Authenticates Hub downloads of **gated** models. A read-only token is enough. See [Gated models](README.md#gated-models-hf_token). |
| `HUGGING_FACE_HUB_TOKEN` | Same as `HF_TOKEN`. | Alternate name accepted by `huggingface_hub` if `HF_TOKEN` is unset. |
| `HF_HUB_DISABLE_XET` | Unset (Glaux default `1`), or any value `huggingface_hub` accepts. | Glaux sets this to `1` before starting Python **only if it is not already set**, so Hub downloads use HTTP with timeout and resume (`hf_xet` is not shipped). Set it yourself to override that default. |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Seconds as an integer string. Glaux default `30` if unset (`huggingface_hub`’s own default is `10`). | Per-request timeout for Hub downloads. Glaux sets `30` only if the variable is not already set. |

## Child-process paths (set by Glaux)

Glaux also **writes** these on child processes when the matching vendor tree exists. You do not normally set them; they are listed because the engines read them:


| Variable | Values | Effect |
| -------- | ------ | ------ |
| `GLAUX_MODELS_CACHE_DIR` | Directory path. | Hugging Face worker cache root (`namespace/repo` folders). The app always configures this to `<appData>/Glaux/Models` at startup, which overrides a user-set value for a normal session. Relevant if you run the Python worker standalone. |
| `GLAUX_FFMPEG` / `GLAUX_FFPROBE` | Absolute paths to `ffmpeg` / `ffprobe`. | Media decode in the Hugging Face worker (not PyAV). The JS bridge sets these from `vendor/ffmpeg` when that tree is present (overwriting any prior value). If ffmpeg was not built, a user-set path is passed through. |
| `GLAUX_CUDA_DIR` | Directory containing the shared CUDA 13 runtime (`vendor/cuda` / `resources/cuda`). | Windows Python `sitecustomize` calls `os.add_dll_directory` on this path so Torch can load `cudart` / `cublas` without relying on `PATH`. Set automatically when the CUDA vendor dir exists. |

## Build scripts

Used only by `npm run build:*`, `npm run dist*` / `scripts/*.js` on a developer machine. They have no effect on a packaged end-user install.


| Variable | Values | Effect |
| -------- | ------ | ------ |
| `CUDA_PATH`, `CUDA_HOME`, `CUDA_ROOT` | CUDA Toolkit install directory (checked in that order). | Locate `nvcc` / CUDA headers when compiling llama.cpp and transcribe.cpp GPU backends. If unset, the scripts look for `nvcc` on `PATH`, then common install locations (`/usr/local/cuda` on Linux). The toolkit `bin` directory is added to `PATH` for cmake even when Debian/Ubuntu did not put `nvcc` on `PATH`. |
| `VULKAN_SDK` | Vulkan SDK root directory. | Locate Vulkan headers/libs for those same native builds. If unset, the scripts search `C:\VulkanSDK\<version>` on Windows and system include paths on Linux. |
| `MSYS2_BASH` | Absolute path to MSYS2 `bash.exe`. | Windows-only: which bash runs the ffmpeg configure/build (`npm run build:ffmpeg`). |
| `MSYS2_PATH` | MSYS2 install root (e.g. `C:\msys64`). | Windows-only: fallback if `MSYS2_BASH` is unset; the script uses `%MSYS2_PATH%\usr\bin\bash.exe`. Otherwise it tries `C:\msys64`, `D:\msys64`, `C:\msys32`, then `bash` on `PATH`. |
| `GLAUX_PACKAGING_TMP` | Absolute directory path. Linux only. Unset → `dist/.tmp`. | Temp directory for `npm run dist` / `dist:linux` (electron-builder). Default is `dist/.tmp` on the project disk so packaging does not fill a small `/tmp` tmpfs (`ENOSPC`). That default directory is deleted after packaging (success or failure). A custom path is left in place. No effect on Windows/macOS. |
