'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('child_process');
const path = require('path');
const { pickPython } = require('../engines/common/runtimePaths');

const REPO = path.resolve(__dirname, '..');

/**
 * Same lookup as the Hugging Face engine: PYTHON, bundled vendor/python, then
 * python on Windows or python3 elsewhere. A missing command and the Windows
 * Store app-execution alias (exit 9009) are both "not found".
 * @param {string} command
 * @returns {string | false}
 */
function pythonSkipReason(command) {
  const result = spawnSync(command, ['-c', 'import sys'], {
    encoding: 'utf8',
    windowsHide: true,
    timeout: 15000,
  });
  if (result.status === 0) return false;
  const pathName = process.platform === 'win32' ? 'python' : 'python3';
  const output = `${result.stderr || ''}\n${result.stdout || ''}`;
  const notFound =
    result.error?.code === 'ENOENT' ||
    result.status === 9009 ||
    /Python was not found/i.test(output);
  if (notFound) {
    return `No Python interpreter found (${command}). Lookup order: PYTHON, then vendor/python, then ${pathName} on PATH.`;
  }
  const detail = (result.stderr || result.stdout || result.error?.message || `exit ${result.status}`).trim();
  return `Python interpreter failed to start (${command}): ${detail}`;
}

const PYTHON = pickPython();
const PYTHON_SKIP = pythonSkipReason(PYTHON);

// Imports worker/t2i.py without worker/__init__.py, which pulls in torch.
const SCRIPT = String.raw`
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))

worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker

download = types.ModuleType("worker.download")
state = {"cache": None}

def models_cache_dir(path=None):
    if path is not None:
        state["cache"] = Path(path).resolve()
        return state["cache"]
    if state["cache"] is None:
        raise RuntimeError("Models cache directory is not configured.")
    return state["cache"]

download.models_cache_dir = models_cache_dir
sys.modules["worker.download"] = download

import worker.t2i as t2i

class DType:
    def __init__(self, itemsize):
        self.itemsize = itemsize

# Resident bytes are already numel * itemsize. CPU 0.55 vs CUDA 0.70:
# 600 of 1000 is over the CPU share and under the CUDA share.
assert t2i._resident_exceeds("cuda", 1000, 600) is False
assert t2i._resident_exceeds("cuda", 1000, 701) is True
assert t2i._resident_exceeds("cuda", 1000, 700) is False
assert t2i._resident_exceeds("cpu", 1000, 400) is False
assert t2i._resident_exceeds("cpu", 1000, 800) is True
assert t2i._resident_exceeds("cpu", 1000, 500) is False
assert t2i._resident_exceeds("cpu", 1000, 600) is True
assert t2i._CPU_RAM_RESIDENT_FRACTION == 0.55
assert t2i._CPU_ACCELERATE_RAM_FRACTION == 0.28
assert t2i._resident_exceeds("mps", 1000, 900) is False
assert t2i._resident_exceeds("cuda", None, 900) is False

load_kwargs = t2i._large_component_load_kwargs(DType(4), {"cpu": 1024, 0: 0}, "/var/tmp/offload")
assert load_kwargs["low_cpu_mem_usage"] is True
assert load_kwargs["device_map"] == "auto"
assert load_kwargs["max_memory"] == {"cpu": 1024, 0: 0}
assert "offload_state_dict" not in load_kwargs

class TilingPipe:
    def __init__(self):
        self.calls = 0
    def enable_vae_tiling(self):
        self.calls += 1

tiled = TilingPipe()
assert t2i._enable_vae_tiling(tiled) is True
assert tiled.calls == 1

class TilingVae:
    def __init__(self):
        self.calls = 0
    def enable_tiling(self):
        self.calls += 1

class VaePipe:
    def __init__(self):
        self.vae = TilingVae()

vae_pipe = VaePipe()
assert t2i._enable_vae_tiling(vae_pipe) is True
assert vae_pipe.vae.calls == 1
assert t2i._enable_vae_tiling(object()) is False

assert t2i._unescape_mount(r"/tmp/foo\040bar") == "/tmp/foo bar"

fstype = t2i._filesystem_type(Path("/tmp"))
if fstype in {"tmpfs", "ramfs", "devtmpfs"}:
    assert t2i._is_ram_disk(Path("/tmp"))
    assert t2i._is_ram_disk(Path("/tmp/glaux-missing"))
elif fstype is not None:
    assert not t2i._is_ram_disk(Path("/tmp"))

if not t2i._is_ram_disk(repo):
    state["cache"] = repo
    chosen = t2i._disk_offload_root(str(repo / "org" / "repo"))
    assert chosen == repo / ".glaux-t2i-offload", chosen

if t2i._is_ram_disk(Path("/tmp")):
    state["cache"] = Path("/tmp")
    chosen = t2i._disk_offload_root("/tmp/org/repo")
    assert chosen.name == ".glaux-t2i-offload"
    assert not t2i._is_ram_disk(chosen.parent), chosen

def path_is_under(path, root):
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    return resolved == root or root in resolved.parents

# /var/tmp is the real Linux fallback. Path("/var/tmp") is not a directory on
# Windows (it resolves onto the current drive), so there the same skip-RAM
# rule is checked with a directory this host actually has.
var_tmp = Path("/var/tmp")
real_is_ram = t2i._is_ram_disk
original_candidates = t2i._offload_candidates
fallback = var_tmp if var_tmp.is_dir() else Path(tempfile.mkdtemp(prefix="glaux-vartmp-"))
ram_root = Path("/tmp") if var_tmp.is_dir() else Path(tempfile.mkdtemp(prefix="glaux-ram-"))
if not var_tmp.is_dir():
    def candidates_with_fallback(model_path):
        found = original_candidates(model_path)
        if fallback not in found:
            found.append(fallback)
        return found

    t2i._offload_candidates = candidates_with_fallback

def only_ram_root_is_ram(path):
    return path_is_under(path, ram_root)

t2i._is_ram_disk = only_ram_root_is_ram
try:
    state["cache"] = ram_root
    chosen = t2i._disk_offload_root(str(ram_root / "org" / "repo"))
    assert not only_ram_root_is_ram(chosen.parent), chosen
    assert chosen == fallback / ".glaux-t2i-offload", chosen
finally:
    t2i._is_ram_disk = real_is_ram
    t2i._offload_candidates = original_candidates
    if not var_tmp.is_dir():
        shutil.rmtree(fallback, ignore_errors=True)
        shutil.rmtree(ram_root, ignore_errors=True)

class Usage:
    def __init__(self, free):
        self.free = free

original_usage = t2i.shutil.disk_usage
try:
    t2i.shutil.disk_usage = lambda path: Usage(10)
    try:
        t2i._require_offload_space(repo, 100)
        raise SystemExit("space check did not fail")
    except RuntimeError as exc:
        assert "Not enough free disk space" in str(exc)
    t2i._require_offload_space(repo, 0)
    t2i.shutil.disk_usage = lambda path: Usage(1000)
    t2i._require_offload_space(repo, 100)
finally:
    t2i.shutil.disk_usage = original_usage

def real_disk_mkdtemp(prefix):
    # Offload files cannot live on tmpfs, and the test must not create a
    # directory inside the repo. Prefer the default temp dir; if that is a
    # RAM disk, use /var/tmp, which is the same fallback the loader uses.
    candidate = Path(tempfile.mkdtemp(prefix=prefix))
    if not t2i._is_ram_disk(candidate):
        return candidate
    shutil.rmtree(candidate, ignore_errors=True)
    var_tmp = Path("/var/tmp")
    if var_tmp.is_dir() and not t2i._is_ram_disk(var_tmp):
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(var_tmp)))
    return None

base = real_disk_mkdtemp("glaux-offload-")
if base is not None:
    try:
        state["cache"] = base
        parent = base / ".glaux-t2i-offload"
        dead = parent / "pid-1"
        live = parent / "pid-424242"
        dead.mkdir(parents=True)
        (dead / "weight.dat").write_text("stale", encoding="utf-8")
        live.mkdir()
        (live / "weight.dat").write_text("live", encoding="utf-8")
        real_alive = t2i._pid_alive
        t2i._pid_alive = lambda pid: pid == os.getpid() or pid == 424242
        try:
            root = t2i._prepare_disk_offload(str(base / "org" / "repo"), 1)
            assert root == parent / f"pid-{os.getpid()}", root
            assert not dead.exists()
            assert (live / "weight.dat").read_text(encoding="utf-8") == "live"
        finally:
            t2i._pid_alive = real_alive
        assert root.is_dir()
        assert not t2i._is_ram_disk(root.parent.parent)
        (root / "weight.dat").write_text("x", encoding="utf-8")
        t2i.release_disk_offload()
        assert not root.exists()
        assert (live / "weight.dat").read_text(encoding="utf-8") == "live"
        assert t2i._active_offload_root is None
    finally:
        shutil.rmtree(base, ignore_errors=True)
`;

const SAMPLING_SCRIPT = String.raw`
import sys
import types
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))

worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker

download = types.ModuleType("worker.download")
download.models_cache_dir = lambda path=None: repo
sys.modules["worker.download"] = download

# _keep_sampling_off_meta only reads device.type and calls torch.device().
class _Device:
    def __init__(self, kind):
        self.type = kind

class _Tensor:
    def __init__(self, device):
        self.device = device if isinstance(device, _Device) else _Device(device)

    @property
    def is_meta(self):
        return self.device.type == "meta"

class _Parameter(_Tensor):
    def __init__(self, data):
        super().__init__(data.device)

class _Module:
    def __init__(self):
        super().__init__()

def _empty(*_shape, device="cpu"):
    return _Tensor(device)

torch = types.ModuleType("torch")
torch.device = _Device
torch.empty = _empty
nn = types.ModuleType("torch.nn")
nn.Module = _Module
nn.Parameter = _Parameter
torch.nn = nn
sys.modules["torch"] = torch
sys.modules["torch.nn"] = nn

import worker.t2i as t2i

class Text(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))

class Pipe:
    def __init__(self):
        self.text_encoder = Text()

    @property
    def device(self):
        return self.text_encoder.weight.device

    @property
    def _execution_device(self):
        return self.device

pipe = Pipe()
assert pipe.device.type == "meta"
assert pipe._execution_device.type == "meta"
t2i._keep_sampling_off_meta(pipe)
assert type(pipe).__name__ == "Pipe"
assert pipe.device.type == "cpu"
assert pipe._execution_device.type == "cpu"
assert pipe.text_encoder.weight.is_meta
t2i._keep_sampling_off_meta(pipe)
assert pipe.device.type == "cpu"

class CudaPipe:
    @property
    def device(self):
        return torch.device("cuda")

    @property
    def _execution_device(self):
        return self.device

cuda = CudaPipe()
t2i._keep_sampling_off_meta(cuda)
assert cuda.device.type == "cuda"
assert cuda._execution_device.type == "cuda"
`;

const PLACEMENT_SCRIPT = String.raw`
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))

worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker

download = types.ModuleType("worker.download")
state = {"cache": None}

def models_cache_dir(path=None):
    if path is not None:
        state["cache"] = Path(path).resolve()
        return state["cache"]
    if state["cache"] is None:
        raise RuntimeError("Models cache directory is not configured.")
    return state["cache"]

download.models_cache_dir = models_cache_dir
sys.modules["worker.download"] = download

chat = types.ModuleType("worker.chat")
chat.CHAT_GENERATION_STOP = False
sys.modules["worker.chat"] = chat

import worker.t2i as t2i

class DType:
    def __init__(self, itemsize):
        self.itemsize = itemsize

GiB = 1024**3
head = t2i._ACTIVATION_HEADROOM_BYTES
block_min = t2i._GROUP_BLOCK_MIN_VRAM_BYTES
fraction = t2i._RAM_RESIDENT_FRACTION
dtype = DType(2)

# Whole pipeline fits in free VRAM after the activation headroom.
assert t2i.select_cuda_tier({"unet": head}, head + head, 10 * GiB) == "cuda"
# Largest component fits, the total does not, and the total still fits in RAM.
assert t2i.select_cuda_tier(
    {"transformer": head, "text_encoder": head},
    head + head,
    10 * GiB,
) == "model_cpu_offload"
# Largest would fit in VRAM, but the checkpoint does not fit the RAM budget,
# so it must not be staged with from_pretrained.
assert t2i.select_cuda_tier({"unet": 1000}, head + 5000, 1000) == "disk"
# Fits in RAM, not in VRAM, and usable VRAM can hold a prefetched block.
assert t2i.select_cuda_tier(
    {"transformer": block_min + 1},
    head + block_min,
    10 * GiB,
) == "group_offload"
# Same, but the usable VRAM is below one prefetched block.
assert t2i.select_cuda_tier(
    {"transformer": 10 * GiB},
    head + block_min - 1,
    100 * GiB,
) == "group_offload_leaf"
# No free VRAM left after the headroom, and the weights fit in RAM: leaf groups.
assert t2i.select_cuda_tier({"unet": GiB}, head, 10 * GiB) == "group_offload_leaf"
# Unknown RAM does not force disk. Bytes outside component folders still count.
assert t2i.select_cuda_tier({"unet": 100}, head + 10_000, None, extra_bytes=100) == "cuda"
assert t2i.select_cuda_tier({}, 0, 1000, extra_bytes=800) == "disk"
# Nothing measured is not "fits on the GPU".
assert t2i.select_cuda_tier({}, 24 * GiB, 64 * GiB) == "model_cpu_offload"
assert t2i.select_group_offload_type(head + block_min) == "block_level"
assert t2i.select_group_offload_type(head + block_min - 1) == "leaf_level"
assert t2i.group_offload_pre_pin(100, 1000) is True
assert t2i.group_offload_pre_pin(400, 1000) is False
assert t2i.group_offload_pre_pin(100, None) is False

block = t2i._group_offload_kwargs("block_level", pre_pin=True)
assert block["use_stream"] is True
assert block["non_blocking"] is True
assert block["record_stream"] is True
assert block["num_blocks_per_group"] == 1
assert block["low_cpu_mem_usage"] is False
assert "offload_to_disk_path" not in block
leaf = t2i._group_offload_kwargs("leaf_level", pre_pin=False)
assert leaf["num_blocks_per_group"] is None
assert leaf["low_cpu_mem_usage"] is True

assert t2i._denoiser_gpu_budget(20 * GiB, GiB) == 16 * GiB
assert t2i._denoiser_gpu_budget(4 * GiB, 0) == GiB
assert t2i._denoiser_gpu_budget(2 * GiB, 0) == (2 * GiB - t2i._GPU_SLIVER_BYTES) // 2
assert t2i._denoiser_gpu_budget(200 * 1024**2, 0) == 0
# A tight card is not raised to 512 MiB. That offer is still checked against
# the real layer floor before anything is called a GPU shard.
tight_offer = t2i._MIN_GPU_OFFLOAD_BYTES + t2i._GPU_SLIVER_BYTES + 4096
assert t2i._denoiser_gpu_budget(tight_offer, 0) == (t2i._MIN_GPU_OFFLOAD_BYTES + 4096) // 2
assert t2i.gpu_shard_floor(100, 100) == 200
assert t2i.gpu_shard_floor(100, 0) == 200
assert t2i.gpu_shard_floor(0, 0) == 0

wide = t2i._component_disk_budgets(
    {"transformer": 12 * GiB, "text_encoder": 5 * GiB, "vae": 100},
    device="cuda",
    available_ram=32 * GiB,
    free_vram=20 * GiB,
    cuda_visible=True,
    gpu_floors={"transformer": 1, "text_encoder": 1, "vae": 1},
)
denoiser_spare = 20 * GiB - head - 100
slack = t2i._ENCODER_GPU_SLACK_BYTES
assert wide["text_encoder"][0] == 5 * GiB + slack
assert wide["transformer"][0] == denoiser_spare - 5 * GiB - slack
assert "vae" not in wide
assert wide["transformer"]["cpu"] == 12 * GiB
assert wide["text_encoder"]["cpu"] == 5 * GiB

tight_budget = int(10 * GiB * fraction)
tight = t2i._component_disk_budgets(
    {"transformer": 8 * GiB, "text_encoder": 2 * GiB},
    device="cuda",
    available_ram=10 * GiB,
    free_vram=24 * GiB,
    cuda_visible=True,
    gpu_floors={"transformer": 1, "text_encoder": 1},
)
assert tight["transformer"]["cpu"] == tight_budget - t2i._CPU_BUDGET_FLOOR
assert tight["text_encoder"]["cpu"] == t2i._CPU_BUDGET_FLOOR
assert tight["text_encoder"][0] == 2 * GiB + slack
# The remainder is larger than the transformer, so the grant is the cast size.
# Encoder slack comes out of the spare before that grant and does not shrink
# a denoiser that still fits.
assert tight["transformer"][0] == 8 * GiB

cpu_budgets = t2i._component_disk_budgets(
    {"unet": 4 * GiB},
    device="cpu",
    available_ram=16 * GiB,
    free_vram=24 * GiB,
    cuda_visible=True,
)
assert cpu_budgets["unet"][0] == 0
assert cpu_budgets["unet"]["cpu"] == max(GiB, int(16 * GiB * t2i._CPU_ACCELERATE_RAM_FRACTION))
hidden = t2i._component_disk_budgets(
    {"unet": 4 * GiB},
    device="cpu",
    available_ram=16 * GiB,
    cuda_visible=False,
)
assert 0 not in hidden["unet"]

# Components under the per-file threshold are still device-mapped when their
# sum crosses it, so a float32 CPU load cannot pull them all into RAM.
promoted = t2i._component_disk_budgets(
    {"unet": GiB, "vae": GiB},
    device="cpu",
    available_ram=16 * GiB,
    cuda_visible=False,
)
assert set(promoted) == {"unet", "vae"}
each_cpu = max(GiB, int(16 * GiB * t2i._CPU_ACCELERATE_RAM_FRACTION / 2))
split_cpu = t2i._component_disk_budgets(
    {"unet": 4 * GiB, "text_encoder": 4 * GiB},
    device="cpu",
    available_ram=16 * GiB,
    cuda_visible=True,
)
assert split_cpu["unet"]["cpu"] == each_cpu
assert split_cpu["text_encoder"]["cpu"] == each_cpu
assert split_cpu["unet"][0] == 0
assert split_cpu["text_encoder"][0] == 0

# Spare is 5 GiB. The encoder's full size does not leave the denoiser its
# floor, but a one-block floor does, so the encoder gets that floor only.
partial_encoder = t2i._component_disk_budgets(
    {"transformer": 20 * GiB, "text_encoder": 30 * GiB},
    device="cuda",
    available_ram=80 * GiB,
    free_vram=8 * GiB,
    cuda_visible=True,
    gpu_floors={"transformer": 2 * GiB, "text_encoder": GiB},
)
assert partial_encoder["text_encoder"][0] == GiB + slack
assert partial_encoder["transformer"][0] == 5 * GiB - GiB - slack
assert partial_encoder["transformer"][0] >= 2 * GiB
# Unknown layer sizes publish no GPU shard, including when spare VRAM is large.
unknown_floor = t2i._component_disk_budgets(
    {"transformer": 12 * GiB, "text_encoder": 5 * GiB},
    device="cuda",
    available_ram=32 * GiB,
    free_vram=20 * GiB,
    cuda_visible=True,
)
assert unknown_floor["text_encoder"][0] == 0
assert unknown_floor["transformer"][0] == 0
# 512 MiB of spare is below a 386 MiB block's floor (layer + one block).
block = 386 * 1024 * 1024
below_floor = t2i._component_disk_budgets(
    {"transformer": 12 * GiB, "text_encoder": 9 * GiB},
    device="cuda",
    available_ram=64 * GiB,
    free_vram=head + t2i._MIN_GPU_OFFLOAD_BYTES,
    cuda_visible=True,
    gpu_floors={"transformer": block * 2, "text_encoder": block * 2},
)
assert below_floor["text_encoder"][0] == 0
assert below_floor["transformer"][0] == 0
# A component smaller than its floor is granted the floor, not its byte size.
# Otherwise Accelerate reserves the layer and places nothing.
tiny_grant = t2i._disk_component_gpu_budgets(
    {"text_encoder": 100, "transformer": 5000},
    0,
    20 * GiB,
    {"text_encoder": 200, "transformer": 400},
)
assert tiny_grant["text_encoder"] == 200 + slack
assert tiny_grant["transformer"] == 5000

assert t2i.cuda_tiers_from("cuda") == [
    "cuda",
    "model_cpu_offload",
    "group_offload",
    "group_offload_leaf",
    "disk",
]
assert t2i.cuda_tiers_from("cuda", "model_cpu_offload")[0] == "model_cpu_offload"
assert t2i.cuda_tiers_from("disk", "model_cpu_offload") == ["disk"]

seen = []

def oom_then_leaf(tier):
    seen.append(tier)
    if tier != "group_offload_leaf":
        raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")
    return "ok"

assert t2i._run_placement_attempts(
    ["cuda", "model_cpu_offload", "group_offload", "group_offload_leaf", "disk"],
    oom_then_leaf,
) == "ok"
assert seen == ["cuda", "model_cpu_offload", "group_offload", "group_offload_leaf"]

def hard_fail(_tier):
    raise RuntimeError("disk full")

try:
    t2i._run_placement_attempts(["cuda", "disk"], hard_fail)
    raise SystemExit("non-OOM error was swallowed")
except RuntimeError as exc:
    assert "disk full" in str(exc)

assert t2i._should_drop_placement(t2i._PlacementUnsupported("no seq")) is True
assert t2i._should_drop_placement(MemoryError()) is True
assert t2i._should_drop_placement(RuntimeError("disk full")) is False

class TorchOOM(RuntimeError):
    pass

try:
    raise TorchOOM("CUDA out of memory")
except TorchOOM as exc:
    wrapped = RuntimeError("load failed")
    wrapped.__cause__ = exc
    assert t2i._is_cuda_oom_error(wrapped) is True

t2i._cuda_tier_limits.clear()

class Pipe:
    _glaux_placement_tier = "cuda"
    _glaux_model_path = "/models/org/repo"

assert t2i.lower_cuda_tier_after_oom(Pipe()) is True
assert t2i._cuda_tier_limits["/models/org/repo"] == "model_cpu_offload"
Pipe._glaux_placement_tier = "disk"
assert t2i.lower_cuda_tier_after_oom(Pipe()) is False
assert t2i._cuda_tier_limits["/models/org/repo"] == "model_cpu_offload"
assert t2i.cuda_tiers_from("cuda", t2i._cuda_tier_limits["/models/org/repo"])[0] == "model_cpu_offload"

noted = []

def note(failed, nxt):
    noted.append((failed, nxt))
    t2i._note_tier_limit("/models/again", failed)

assert t2i._run_placement_attempts(
    ["cuda", "model_cpu_offload"],
    lambda tier: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")) if tier == "cuda" else tier,
    note,
) == "model_cpu_offload"
assert noted == [("cuda", "model_cpu_offload")]
assert t2i._cuda_tier_limits["/models/again"] == "model_cpu_offload"

log = t2i._format_placement_log(
    "model_cpu_offload",
    "bfloat16",
    20 * GiB,
    None,
    {"unet": 5 * GiB, "text_encoder": GiB},
)
assert "model_cpu_offload" in log
assert "bfloat16" in log
assert "free_vram=20.00GiB" in log
assert "available_ram=unknown" in log
assert "text_encoder=1.00GiB" in log
assert "unet=5.00GiB" in log
assert t2i._format_step_timing(4, 10.0) == (
    "Text-to-image generation: 4 denoising steps, wall 10.0s."
)
assert t2i._format_step_timing(4, 10.0, encode=1.25, denoise=7.5, decode=1.25) == (
    "Text-to-image generation: 4 denoising steps, wall 10.0s "
    "(prompt encode 1.25s, denoise 7.50s, decode 1.25s)."
)
assert t2i.select_compute_dtype_name("cpu", bf16_native=True) == "float32"
assert t2i.select_compute_dtype_name("cuda", bf16_native=True) == "bfloat16"
assert t2i.select_compute_dtype_name("cuda", bf16_native=False) == "float16"
assert t2i.select_compute_dtype_name("mps", bf16_native=True) == "float16"

class _Bf16Cuda:
    def __init__(self, native, capability):
        self._native = native
        self._capability = capability

    def is_bf16_supported(self, including_emulation=True):
        if including_emulation:
            return True
        return self._native

    def get_device_capability(self):
        return self._capability

class _TorchBf16:
    def __init__(self, cuda):
        self.cuda = cuda

assert t2i._cuda_bf16_without_emulation(_TorchBf16(_Bf16Cuda(False, (7, 5)))) is False
assert t2i._cuda_bf16_without_emulation(_TorchBf16(_Bf16Cuda(True, (8, 0)))) is True

class _LegacyBf16Cuda:
    def is_bf16_supported(self):
        return True

    def get_device_capability(self):
        return (7, 5)

assert t2i._cuda_bf16_without_emulation(_TorchBf16(_LegacyBf16Cuda())) is False
_LegacyBf16Cuda.get_device_capability = lambda self: (8, 6)
assert t2i._cuda_bf16_without_emulation(_TorchBf16(_LegacyBf16Cuda())) is True

timer = t2i._StepTimer()
timer.mark(0)
timer.mark(2)
timer.mark(1)
assert timer.steps == 3
timer.mark()
assert timer.steps == 4
callback = t2i._bind_step_timer(timer)

class StepPipe:
    pass

assert callback(StepPipe(), 4, 0, {"latents": 1}) == {"latents": 1}
assert timer.steps == 5
chat.CHAT_GENERATION_STOP = True
try:
    callback(StepPipe(), 5, 0, {})
    raise SystemExit("cancel did not stop the step callback")
except t2i.DiffusionStopped:
    pass
assert timer.steps == 6
chat.CHAT_GENERATION_STOP = False

class Scheduler:
    def step(self, value):
        return value + 1

class SchedPipe:
    def __init__(self):
        self.scheduler = Scheduler()

sched = SchedPipe()
first = t2i._StepTimer()
t2i._arm_scheduler_stop(sched, first)
assert sched.scheduler.step(1) == 2
assert first.steps == 1
second = t2i._StepTimer()
t2i._arm_scheduler_stop(sched, second)
assert sched.scheduler.step(3) == 4
assert second.steps == 1
assert first.steps == 1
chat.CHAT_GENERATION_STOP = True
try:
    sched.scheduler.step(0)
    raise SystemExit("cancel did not stop the scheduler hook")
except t2i.DiffusionStopped:
    pass

def real_disk_mkdtemp(prefix):
    candidate = Path(tempfile.mkdtemp(prefix=prefix))
    if not t2i._is_ram_disk(candidate):
        return candidate
    shutil.rmtree(candidate, ignore_errors=True)
    var_tmp = Path("/var/tmp")
    if var_tmp.is_dir() and not t2i._is_ram_disk(var_tmp):
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(var_tmp)))
    return None

base = real_disk_mkdtemp("glaux-tier-")
if base is not None:
    try:
        state["cache"] = base
        holder = {}

        def fail_disk(tier):
            if tier != "disk":
                raise RuntimeError("CUDA out of memory")
            holder["root"] = t2i._prepare_disk_offload(str(base / "org" / "repo"), 1)
            (holder["root"] / "weight.dat").write_text("x", encoding="utf-8")
            raise RuntimeError("CUDA out of memory")

        try:
            t2i._run_placement_attempts(["cuda", "disk"], fail_disk)
            raise SystemExit("disk OOM should still raise on the last tier")
        except RuntimeError as exc:
            assert "CUDA out of memory" in str(exc)
        root = holder["root"]
        assert root == base / ".glaux-t2i-offload" / f"pid-{os.getpid()}"
        assert not root.exists()
        assert t2i._active_offload_root is None
    finally:
        shutil.rmtree(base, ignore_errors=True)

    base = real_disk_mkdtemp("glaux-tier-keep-")
    if base is None:
        raise SystemExit("no real disk for the tier-keep temp dir")
    try:
        state["cache"] = base
        kept = t2i._prepare_disk_offload(str(base / "org" / "repo"), 1)
        (kept / "weight.dat").write_text("keep", encoding="utf-8")

        def fail_eager(_tier):
            raise RuntimeError("CUDA out of memory")

        try:
            t2i._run_placement_attempts(["cuda", "model_cpu_offload"], fail_eager)
            raise SystemExit("eager OOM should raise when no tier works")
        except RuntimeError:
            pass
        assert kept.is_dir()
        assert (kept / "weight.dat").read_text(encoding="utf-8") == "keep"
        assert t2i._active_offload_root == kept
    finally:
        t2i.release_disk_offload()
        shutil.rmtree(base, ignore_errors=True)

import weakref

held = {}

def leaking_attempt(tier):
    if "ref" in held:
        assert held["ref"]() is None, "failed pipeline was still alive at the next attempt"
        return "collected"
    failed = type("Pipeline", (), {})()
    held["ref"] = weakref.ref(failed)
    raise MemoryError("CUDA out of memory")

assert t2i._run_placement_attempts(["cuda", "model_cpu_offload"], leaking_attempt) == "collected"

t2i._cuda_tier_limits["/models/sticky"] = "disk"
t2i._cuda_tier_limit_vram["/models/sticky"] = 3 * GiB
t2i.clear_cuda_tier_limit("/models/sticky")
assert "/models/sticky" not in t2i._cuda_tier_limits
assert "/models/sticky" not in t2i._cuda_tier_limit_vram

cap_path = "/models/sticky-cap"
cap_resolved = str(Path(cap_path).resolve())
assert t2i._note_tier_limit(cap_path, "cuda", free_vram=10 * GiB) == "model_cpu_offload"
assert t2i._cuda_tier_limit_vram[cap_resolved] == 10 * GiB
assert t2i._note_tier_limit(cap_path, "model_cpu_offload", free_vram=1) == "group_offload"
assert t2i._cuda_tier_limit_vram[cap_resolved] == 10 * GiB
growth = t2i._TIER_LIMIT_VRAM_GROWTH_BYTES
assert t2i.cuda_tier_limit_for(cap_path, 10 * GiB + growth) == "group_offload"
assert t2i.cuda_tier_limit_for(cap_path, 10 * GiB + growth + 1) is None
assert cap_resolved not in t2i._cuda_tier_limits
assert cap_resolved not in t2i._cuda_tier_limit_vram
t2i._note_tier_limit(cap_path, "cuda", free_vram=4 * GiB)
t2i.clear_cuda_tier_limit(cap_path)
assert cap_resolved not in t2i._cuda_tier_limits
t2i._cuda_tier_limits[cap_resolved] = "disk"
assert t2i.cuda_tier_limit_for(cap_path, 100 * GiB) == "disk"
t2i.clear_cuda_tier_limit(cap_path)

class ForceVae:
    def __init__(self):
        self.config = types.SimpleNamespace(force_upcast=True)

def upcast_vae(self):
    return None

# The pipeline method plus force_upcast decides. The class name does not.
force_pipe = type("StableDiffusionXLPipeline", (), {"upcast_vae": upcast_vae})()
force_pipe.vae = ForceVae()
assert t2i._pipeline_owns_vae_upcast(force_pipe) is True
assert t2i._prepare_fp16_vae(force_pipe, "float16") is False
assert not hasattr(force_pipe.vae, "_glaux_vae_io_cast")
flux_pipe = type("FluxPipeline", (), {})()
flux_pipe.vae = type("AutoencoderKL", (), {"config": types.SimpleNamespace(force_upcast=True)})()
assert t2i._pipeline_owns_vae_upcast(flux_pipe) is False
sd3_pipe = type("StableDiffusion3Pipeline", (), {})()
sd3_pipe.vae = type("AutoencoderKL", (), {})()
assert t2i._pipeline_owns_vae_upcast(sd3_pipe) is False
qwen_pipe = type("QwenImagePipeline", (), {})()
qwen_pipe.vae = type("AutoencoderKLQwenImage", (), {})()
assert t2i._pipeline_owns_vae_upcast(qwen_pipe) is False
plain_pipe = type("StableDiffusionXLPipeline", (), {"upcast_vae": upcast_vae})()
plain_pipe.vae = type("AutoencoderKL", (), {})()
plain_pipe.vae.config = types.SimpleNamespace(force_upcast=False)
assert t2i._pipeline_owns_vae_upcast(plain_pipe) is False
assert t2i._pipeline_owns_vae_upcast(type("FluxPipeline", (), {})()) is False

oom_path = "/models/oom-stamp"
oom_resolved = str(Path(oom_path).resolve())
stamp = 8 * GiB
resident = GiB
post_reload = 8 * GiB

class ResidentPipe:
    _glaux_placement_tier = "cuda"
    _glaux_model_path = oom_path
    _glaux_free_vram = stamp

t2i._free_vram_bytes = lambda: resident
assert t2i.lower_cuda_tier_after_oom(ResidentPipe()) is True
assert t2i._cuda_tier_limits[oom_resolved] == "model_cpu_offload"
assert t2i._cuda_tier_limit_vram[oom_resolved] == stamp
assert t2i.cuda_tier_limit_for(oom_path, post_reload) == "model_cpu_offload"
assert t2i.cuda_tier_limit_for(oom_path, stamp + growth) == "model_cpu_offload"
t2i.clear_cuda_tier_limit(oom_path)

class UnstampedPipe:
    _glaux_placement_tier = "cuda"
    _glaux_model_path = oom_path

assert t2i.lower_cuda_tier_after_oom(UnstampedPipe()) is True
assert t2i._cuda_tier_limit_vram[oom_resolved] == resident
assert t2i.cuda_tier_limit_for(oom_path, post_reload) is None

class NoneStamp:
    _glaux_placement_tier = "cuda"
    _glaux_model_path = "/models/none-stamp"
    _glaux_free_vram = None

assert t2i.lower_cuda_tier_after_oom(NoneStamp()) is True
none_resolved = str(Path("/models/none-stamp").resolve())
assert none_resolved not in t2i._cuda_tier_limit_vram
assert t2i.cuda_tier_limit_for("/models/none-stamp", 100 * GiB) == "model_cpu_offload"
t2i.clear_cuda_tier_limit("/models/none-stamp")

class Blockless:
    def named_children(self):
        return []

class Blocks:
    _group_offload_block_modules = ["down_blocks"]

assert t2i._offload_type_for_component("vae", Blockless(), "block_level") == "leaf_level"
assert t2i._offload_type_for_component("text_encoder", Blockless(), "block_level") == "leaf_level"
assert t2i._offload_type_for_component("unet", Blockless(), "block_level") == "leaf_level"
assert t2i._offload_type_for_component("unet", Blocks(), "block_level") == "block_level"
assert t2i._offload_type_for_component("transformer_2", Blocks(), "leaf_level") == "leaf_level"
assert t2i._component_has_block_modules(Blocks()) is True
`;

const HEADER_SCRIPT = String.raw`
import io
import json
import sys
import tempfile
import types
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))
worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker
download = types.ModuleType("worker.download")
download.models_cache_dir = lambda path=None: Path(path) if path else repo
sys.modules["worker.download"] = download
import worker.t2i as t2i

GiB = 1024**3

class DType:
    def __init__(self, itemsize):
        self.itemsize = itemsize

def write_header(path, tensors):
    header = {}
    offset = 0
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1}
    for name, dtype, shape in tensors:
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * widths[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    payload = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(payload).to_bytes(8, "little") + payload)

def numel_for(nbytes, itemsize):
    return nbytes // itemsize

fp16 = DType(2)

sdxl = Path(tempfile.mkdtemp(prefix="glaux-sdxl-"))
(sdxl / "model_index.json").write_text(json.dumps({
    "_class_name": "StableDiffusionXLPipeline",
    "unet": ["diffusers", "UNet2DConditionModel"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
    "vae": ["diffusers", "AutoencoderKL"],
}), encoding="utf-8")
write_header(sdxl / "unet" / "diffusion_pytorch_model.safetensors", [
    ("w", "F16", [numel_for(20 * GiB, 2)]),
])
write_header(sdxl / "unet" / "diffusion_pytorch_model.fp16.safetensors", [
    ("w", "F16", [numel_for(40 * GiB, 2)]),
])
write_header(sdxl / "unet" / "model.safetensors", [
    ("w", "F16", [numel_for(40 * GiB, 2)]),
])
write_header(sdxl / "text_encoder" / "model.safetensors", [
    ("w", "F16", [numel_for(4 * GiB, 2)]),
])
write_header(sdxl / "text_encoder" / "model.fp16.safetensors", [
    ("w", "F16", [numel_for(40 * GiB, 2)]),
])
write_header(sdxl / "text_encoder" / "diffusion_pytorch_model.safetensors", [
    ("w", "F16", [numel_for(40 * GiB, 2)]),
])
write_header(sdxl / "text_encoder_2" / "model.safetensors", [
    ("w", "F16", [numel_for(4 * GiB, 2)]),
])
write_header(sdxl / "vae" / "diffusion_pytorch_model.safetensors", [
    ("w", "F32", [1024 * 1024]),
])
write_header(sdxl / "vae" / "diffusion_pytorch_model.fp16.safetensors", [
    ("w", "F16", [numel_for(40 * GiB, 2)]),
])
write_header(sdxl / "sd_xl_base_1.0.safetensors", [
    ("w", "F16", [numel_for(80 * GiB, 2)]),
])
sdxl_sizes = t2i.measure_component_bytes(str(sdxl), fp16)
assert sdxl_sizes["unet"] == 20 * GiB, sdxl_sizes
assert sdxl_sizes["text_encoder"] == 4 * GiB, sdxl_sizes
assert sdxl_sizes["text_encoder_2"] == 4 * GiB, sdxl_sizes
assert sdxl_sizes["vae"] == 1024 * 1024 * 2, sdxl_sizes
sdxl_tier = t2i.select_cuda_tier(sdxl_sizes, 24 * GiB, 64 * GiB)
assert sdxl_tier == "model_cpu_offload", (sdxl_tier, sdxl_sizes)

flux = Path(tempfile.mkdtemp(prefix="glaux-flux-"))
(flux / "model_index.json").write_text(json.dumps({
    "_class_name": "FluxPipeline",
    "transformer": ["diffusers", "FluxTransformer2DModel"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}), encoding="utf-8")
(flux / "transformer").mkdir()
(flux / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
    "metadata": {"total_size": 1},
    "weight_map": {
        "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
        "b": "diffusion_pytorch_model-00001-of-00002.safetensors",
        "c": "diffusion_pytorch_model-00002-of-00002.safetensors",
    },
}), encoding="utf-8")
write_header(flux / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors", [
    ("a", "F16", [numel_for(8 * GiB, 2)]),
])
write_header(flux / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors", [
    ("c", "BF16", [numel_for(8 * GiB, 2)]),
])
write_header(flux / "transformer" / "diffusion_pytorch_model.safetensors", [
    ("w", "F16", [numel_for(40 * GiB, 2)]),
])
write_header(flux / "transformer" / "diffusion_pytorch_model.fp16.safetensors", [
    ("w", "F16", [numel_for(40 * GiB, 2)]),
])
write_header(flux / "text_encoder" / "model.safetensors", [
    ("w", "F16", [numel_for(1 * GiB, 2)]),
])
(flux / "text_encoder_2").mkdir()
(flux / "text_encoder_2" / "model.safetensors.index.json").write_text(json.dumps({
    "weight_map": {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00002-of-00002.safetensors",
    },
}), encoding="utf-8")
write_header(flux / "text_encoder_2" / "model-00001-of-00002.safetensors", [
    ("a", "F16", [numel_for(4 * GiB, 2)]),
])
write_header(flux / "text_encoder_2" / "model-00002-of-00002.safetensors", [
    ("b", "F8_E4M3", [numel_for(4 * GiB, 2)]),
])
# No model.safetensors here: transformers uses the index only when the single
# file is absent. A sibling single file would be the one that loads.
write_header(flux / "vae" / "diffusion_pytorch_model.safetensors", [
    ("w", "F16", [numel_for(GiB // 2, 2)]),
])
write_header(flux / "flux1-dev.safetensors", [
    ("w", "F16", [numel_for(80 * GiB, 2)]),
])
flux_sizes = t2i.measure_component_bytes(str(flux), fp16)
assert flux_sizes["transformer"] == 16 * GiB, flux_sizes
assert flux_sizes["text_encoder"] == GiB, flux_sizes
assert flux_sizes["text_encoder_2"] == 8 * GiB, flux_sizes
assert flux_sizes["vae"] == GiB // 2, flux_sizes
flux_tier = t2i.select_cuda_tier(flux_sizes, 24 * GiB, 64 * GiB)
assert flux_tier == "model_cpu_offload", (flux_tier, flux_sizes)
fp32 = DType(4)
flux_fp32 = t2i.measure_component_bytes(str(flux), fp32)
assert flux_fp32["transformer"] == 32 * GiB, flux_fp32

# Transformers prefers a single model.safetensors over a sibling index.
prefer = Path(tempfile.mkdtemp(prefix="glaux-prefer-"))
(prefer / "model_index.json").write_text(json.dumps({
    "_class_name": "FluxPipeline",
    "text_encoder_2": ["transformers", "T5EncoderModel"],
}), encoding="utf-8")
(prefer / "text_encoder_2").mkdir()
(prefer / "text_encoder_2" / "model.safetensors.index.json").write_text(json.dumps({
    "weight_map": {"a": "model-00001-of-00001.safetensors"},
}), encoding="utf-8")
write_header(prefer / "text_encoder_2" / "model-00001-of-00001.safetensors", [
    ("a", "F16", [numel_for(40 * GiB, 2)]),
])
write_header(prefer / "text_encoder_2" / "model.safetensors", [
    ("encoder.block.0.weight", "F16", [numel_for(GiB, 2)]),
    ("encoder.block.1.weight", "F16", [numel_for(GiB, 2)]),
])
prefer_layouts = t2i.measure_component_layouts(str(prefer), fp16)
assert prefer_layouts["text_encoder_2"][0] == 2 * GiB, prefer_layouts
# Two equal blocks: floor is one block reserved plus one block placed.
assert prefer_layouts["text_encoder_2"][1] == 2 * GiB, prefer_layouts

def capture_sizes(root):
    buf = io.StringIO()
    old_err = sys.stderr
    sys.stderr = buf
    try:
        sizes = t2i.measure_component_bytes(str(root), fp16)
        layouts = t2i.measure_component_layouts(str(root), fp16)
    finally:
        sys.stderr = old_err
    return sizes, layouts, buf.getvalue()

def bad_repo(raw, filename="diffusion_pytorch_model.safetensors"):
    root = Path(tempfile.mkdtemp(prefix="glaux-bad-"))
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
    }), encoding="utf-8")
    path = root / "unet" / filename
    path.parent.mkdir()
    path.write_bytes(raw)
    return root, path

huge_root, huge_path = bad_repo((200 * 1024 * 1024).to_bytes(8, "little"))
huge_sizes, huge_layouts, huge_log = capture_sizes(huge_root)
assert huge_sizes["unet"] == huge_path.stat().st_size, huge_sizes
assert huge_layouts["unet"][1] == 0
assert huge_log.count("Using the file size") == 2, huge_log

trunc_root, trunc_path = bad_repo((100).to_bytes(8, "little") + b"{}")
trunc_sizes, _, trunc_log = capture_sizes(trunc_root)
assert trunc_sizes["unet"] == trunc_path.stat().st_size
assert "Using the file size" in trunc_log

payload = b"not-json"
json_root, json_path = bad_repo(len(payload).to_bytes(8, "little") + payload)
json_sizes, _, json_log = capture_sizes(json_root)
assert json_sizes["unet"] == json_path.stat().st_size
assert "Using the file size" in json_log

negative = json.dumps({
    "w": {"dtype": "F16", "shape": [-4, 8], "data_offsets": [0, 1]},
}).encode()
neg_root, neg_path = bad_repo(len(negative).to_bytes(8, "little") + negative)
neg_sizes, neg_layouts, neg_log = capture_sizes(neg_root)
assert neg_sizes["unet"] == neg_path.stat().st_size
assert neg_layouts["unet"][1] == 0
assert "Using the file size" in neg_log

mixed = json.dumps({
    "bad": {"dtype": "F16", "shape": [-1], "data_offsets": [0, 1]},
    "good": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]},
}).encode()
mixed_root, _mixed_path = bad_repo(len(mixed).to_bytes(8, "little") + mixed)
mixed_sizes, mixed_layouts, mixed_log = capture_sizes(mixed_root)
assert mixed_sizes["unet"] == 16 * 2, mixed_sizes
assert mixed_layouts["unet"][1] == t2i.gpu_shard_floor(16, 0) * 2
assert "Using the file size" not in mixed_log

empty_root, _empty_path = bad_repo(b"")
empty_sizes, _, empty_log = capture_sizes(empty_root)
assert "unet" not in empty_sizes
assert "Using the file size" in empty_log
assert t2i.select_cuda_tier(empty_sizes, 24 * GiB, 64 * GiB) == "model_cpu_offload"

bin_root = Path(tempfile.mkdtemp(prefix="glaux-bin-"))
(bin_root / "model_index.json").write_text(json.dumps({
    "_class_name": "StableDiffusionPipeline",
    "unet": ["diffusers", "UNet2DConditionModel"],
}), encoding="utf-8")
bin_path = bin_root / "unet" / "diffusion_pytorch_model.bin"
bin_path.parent.mkdir()
bin_path.write_bytes(b"x" * 1000)
bin_sizes, bin_layouts, bin_log = capture_sizes(bin_root)
assert bin_sizes["unet"] == 1000, bin_sizes
assert bin_layouts["unet"][1] == 0
assert "file bytes" in bin_log
assert t2i.select_cuda_tier(bin_sizes, 24 * GiB, 64 * GiB) == "cuda"
assert t2i.select_cuda_tier({"unet": 40 * GiB}, 24 * GiB, 16 * GiB) == "disk"
`;

const APPLY_SCRIPT = String.raw`
import io
import sys
import types
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))
worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker
download = types.ModuleType("worker.download")
download.models_cache_dir = lambda path=None: Path(path) if path else repo
sys.modules["worker.download"] = download
chat = types.ModuleType("worker.chat")
chat.CHAT_GENERATION_STOP = False
sys.modules["worker.chat"] = chat
import worker.t2i as t2i

class DType:
    def __init__(self, itemsize):
        self.itemsize = itemsize

class Device:
    def __init__(self, name="cpu"):
        self.type = str(name)

class Module:
    def __init__(self, name, blocks=False):
        self.name = name
        self.fail = False
        if blocks:
            self._group_offload_block_modules = ["down"]

    def named_children(self):
        return []

    def to(self, device):
        self.moved = device
        return self

torch = types.ModuleType("torch")
torch.nn = types.ModuleType("torch.nn")
torch.nn.Module = Module
torch.nn.ModuleList = type("ModuleList", (Module,), {})
torch.nn.Sequential = type("Sequential", (Module,), {})
torch.device = Device
host_calls = []
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
torch.accelerator = types.SimpleNamespace(empty_host_cache=lambda: host_calls.append("host"))
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn

applied = []

def apply_group_offloading(module, **kwargs):
    applied.append((
        module.name,
        kwargs["offload_type"],
        kwargs["onload_device"].type,
        kwargs["low_cpu_mem_usage"],
        kwargs["use_stream"],
    ))
    if module.fail:
        raise RuntimeError("group offload rejected")

hooks = types.ModuleType("diffusers.hooks")
hooks.apply_group_offloading = apply_group_offloading
diffusers = types.ModuleType("diffusers")
diffusers.hooks = hooks
sys.modules["diffusers"] = diffusers
sys.modules["diffusers.hooks"] = hooks

class Pipe:
    def __init__(self):
        self.vae = Module("vae")
        self.text_encoder = Module("text_encoder")
        self.unet = Module("unet", blocks=True)

    @property
    def components(self):
        return {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "unet": self.unet,
            "scheduler": object(),
        }

pipe = Pipe()
t2i._apply_group_offload(pipe, "block_level", pre_pin=True, onload_device="cpu")
by_name = {name: rest for name, *rest in applied}
assert by_name["vae"] == ["leaf_level", "cpu", False, False], by_name
assert by_name["text_encoder"][0] == "leaf_level"
assert by_name["unet"] == ["block_level", "cpu", False, False], by_name
class SdxlPipe(Pipe):
    def upcast_vae(self):
        return None

sdxl = SdxlPipe()
sdxl.vae.config = types.SimpleNamespace(force_upcast=True)
applied.clear()
stream_log = io.StringIO()
old_stream_err = sys.stderr
sys.stderr = stream_log
try:
    t2i._apply_group_offload(sdxl, "block_level", pre_pin=True, onload_device="cuda")
finally:
    sys.stderr = old_stream_err
sdxl_by_name = {name: rest for name, *rest in applied}
assert sdxl_by_name["vae"][3] is False, sdxl_by_name
assert sdxl_by_name["unet"][3] is True, sdxl_by_name
assert "not streamed" in stream_log.getvalue(), stream_log.getvalue()
flux_stream = Pipe()
applied.clear()
t2i._apply_group_offload(flux_stream, "block_level", pre_pin=True, onload_device="cuda")
flux_by_name = {name: rest for name, *rest in applied}
assert flux_by_name["vae"][3] is True, flux_by_name

pipe.vae.fail = True
try:
    t2i._apply_group_offload(pipe, "block_level", pre_pin=False, onload_device="cpu")
    raise SystemExit("non-denoiser group-offload failure was swallowed")
except t2i._PlacementUnsupported as exc:
    assert "vae" in str(exc)

class ModelPipe:
    model_cpu_offload_seq = "text_encoder->unet"

    def enable_model_cpu_offload(self):
        self.enabled = True

model_pipe = ModelPipe()
t2i._apply_model_cpu_offload(model_pipe)
assert model_pipe.enabled is True

class NoSeq:
    model_cpu_offload_seq = None

try:
    t2i._apply_model_cpu_offload(NoSeq())
    raise SystemExit("missing sequence was accepted")
except t2i._PlacementUnsupported:
    pass

class FakePipe(Pipe):
    model_cpu_offload_seq = "unet"

    def to(self, device):
        self.device_set = device
        return self

    def enable_model_cpu_offload(self):
        self.model_offloaded = True

def from_pretrained(path, **kwargs):
    return FakePipe()

diffusers.DiffusionPipeline = types.SimpleNamespace(from_pretrained=from_pretrained)
cuda_pipe = t2i._place_cuda_tier("/models/m", DType(2), "cuda", {}, available_ram=None, free_vram=None)
assert cuda_pipe.device_set == "cuda"
model_tier = t2i._place_cuda_tier(
    "/models/m", DType(2), "model_cpu_offload", {}, available_ram=None, free_vram=None
)
assert model_tier.model_offloaded is True
sentinel = object()
t2i._load_disk_offloaded = lambda *args, **kwargs: sentinel
assert t2i._place_cuda_tier(
    "/models/m", DType(2), "disk", {}, available_ram=1, free_vram=1
) is sentinel

seen = []
resolved_holder = {}

def place_spy(model_path, dtype, tier, cast, *, available_ram, free_vram, pre_pin=None, layouts=None):
    resolved = str(Path(model_path).resolve())
    resolved_holder["path"] = resolved
    seen.append((tier, pre_pin, t2i._cuda_tier_limits.get(resolved)))
    if pre_pin is True:
        raise RuntimeError("CUDA error: out of memory")
    return FakePipe()

t2i._place_cuda_tier = place_spy
t2i.measure_component_layouts = lambda path, dtype: {"unet": (100, 0)}
t2i.select_cuda_tier = lambda *args, **kwargs: "cuda"
model_path = "/models/pin-retry"
resolved = str(Path(model_path).resolve())
t2i._cuda_tier_limits[resolved] = "group_offload"
buf = io.StringIO()
old_err = sys.stderr
sys.stderr = buf
try:
    loaded = t2i._load_cuda_pipeline(model_path, DType(2), 10_000, 8 * 1024**3)
finally:
    sys.stderr = old_err
assert seen == [
    ("group_offload", True, "group_offload"),
    ("group_offload", False, "group_offload"),
], seen
assert t2i._cuda_tier_limits.get(resolved) == "group_offload"
assert "low_cpu_mem_usage=True" in buf.getvalue()
assert isinstance(loaded, FakePipe)
assert loaded._glaux_free_vram == 8 * 1024**3
host_calls.clear()
t2i._empty_host_cache()
assert host_calls == ["host"]
`;

const CHAT_SCRIPT = String.raw`
import io
import sys
import types
import weakref
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))
worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker

transformers = types.ModuleType("transformers")
transformers.__path__ = []
transformers.TextIteratorStreamer = object
transformers.pipeline = lambda *args, **kwargs: None
generation = types.ModuleType("transformers.generation")
generation.__path__ = []
stopping = types.ModuleType("transformers.generation.stopping_criteria")
stopping.StoppingCriteria = object
stopping.StoppingCriteriaList = object
sys.modules["transformers"] = transformers
sys.modules["transformers.generation"] = generation
sys.modules["transformers.generation.stopping_criteria"] = stopping

context = types.ModuleType("worker.context")
context._invalidate_context_usage_cache = lambda: None
sys.modules["worker.context"] = context

download = types.ModuleType("worker.download")
download._assert_valid_model_id = lambda model_id: model_id
download._load_progress_tqdm_hook = lambda: None
download.model_local_dir = lambda model_id: Path("/models") / model_id
download.read_model_pipeline_tag = lambda model_id: "text-to-image"
download._load_progress_callback = None
sys.modules["worker.download"] = download

thinking = types.ModuleType("worker.thinking")
thinking._iter_stripped_non_thinking_markup = lambda *args, **kwargs: None
thinking._parse_tags_and_answer = lambda *args, **kwargs: ("", "")
thinking._thinking_open_tag_from_generation_suffix = lambda *args, **kwargs: None
sys.modules["worker.thinking"] = thinking

import worker.chat as chat
import worker.t2i as t2i

class Pipe:
    def __init__(self, tier="cuda"):
        self._glaux_uses_cuda = True
        self._glaux_placement_tier = tier
        self._glaux_model_path = "/models/org/repo"

def make_pipe(tier="cuda"):
    return Pipe(tier)

creates = []

def fake_create(model_id):
    creates.append(model_id)
    chat.CHATBOT = make_pipe("model_cpu_offload")
    chat.CHATBOT_MODEL_ID = model_id
    chat.CHATBOT_PIPELINE_TAG = "text-to-image"

chat.chatbot_create = fake_create
chat.CHATBOT_PIPELINE_TAG = "text-to-image"
chat.CHATBOT_MODEL_ID = "org/repo"
chat._cuda_unavailable = False
chat._cuda_oom_model_ids.clear()
t2i._cuda_tier_limits.clear()

box = {}

def generate_oom(pipe, prompt, output_path, image_path=None):
    if box.get("phase") == "retry":
        assert box["ref"]() is None, "failed pipeline was still alive when the retry started"
        return "/tmp/out.png"
    box["phase"] = "retry"
    box["ref"] = weakref.ref(pipe)
    raise RuntimeError("CUDA out of memory")

chat.CHATBOT = make_pipe()
chat.generate_text_to_image = generate_oom
saved = chat._generate_text_to_image_turn("a cat", None, None, None, "/tmp/out.png")
assert saved == "/tmp/out.png", saved
assert creates == ["org/repo"], creates
assert t2i._cuda_tier_limits["/models/org/repo"] == "model_cpu_offload"

creates.clear()
t2i._cuda_tier_limits.clear()
chat.CHATBOT = make_pipe()
chat.CHATBOT_MODEL_ID = "org/repo"
chat.CHATBOT_PIPELINE_TAG = "text-to-image"
calls = {"n": 0}

def generate_stop(pipe, prompt, output_path, image_path=None):
    calls["n"] += 1
    chat.CHAT_GENERATION_STOP = True
    raise RuntimeError("CUDA out of memory")

chat.generate_text_to_image = generate_stop
stop_log = io.StringIO()
old_err = sys.stderr
sys.stderr = stop_log
try:
    assert chat._generate_text_to_image_turn("a cat", None, None, None, "/tmp/out.png") == ""
finally:
    sys.stderr = old_err
assert calls["n"] == 1
assert creates == []
assert "not retrying" in stop_log.getvalue(), stop_log.getvalue()
assert "RuntimeError: CUDA out of memory" in stop_log.getvalue()

def generate_multiline(pipe, prompt, output_path, image_path=None):
    calls["n"] += 1
    chat.CHAT_GENERATION_STOP = True
    raise RuntimeError("CUDA out of memory.\nTried to allocate 20.00 GiB")

chat.CHATBOT = make_pipe()
chat.CHATBOT_MODEL_ID = "org/repo"
chat.generate_text_to_image = generate_multiline
multi_log = io.StringIO()
sys.stderr = multi_log
try:
    assert chat._generate_text_to_image_turn("a cat", None, None, None, "/tmp/out.png") == ""
finally:
    sys.stderr = old_err
assert calls["n"] == 2
assert "RuntimeError: CUDA out of memory." in multi_log.getvalue(), multi_log.getvalue()
assert "Tried to allocate" not in multi_log.getvalue()

creates.clear()
calls["n"] = 0
GiB = 1024 ** 3
stamp = 8 * GiB
resident = GiB

def live_vram():
    return resident

t2i._free_vram_bytes = live_vram
stamped = make_pipe("cuda")
stamped._glaux_free_vram = stamp
chat.CHATBOT = stamped
chat.CHATBOT_MODEL_ID = "org/repo"
chat.CHATBOT_PIPELINE_TAG = "text-to-image"
box.clear()

def generate_oom_stamp(pipe, prompt, output_path, image_path=None):
    if box.get("phase") == "retry":
        return "/tmp/out.png"
    box["phase"] = "retry"
    raise RuntimeError("CUDA out of memory")

chat.generate_text_to_image = generate_oom_stamp
assert chat._generate_text_to_image_turn("a cat", None, None, None, "/tmp/out.png") == "/tmp/out.png"
cap_resolved = str(Path("/models/org/repo").resolve())
assert t2i._cuda_tier_limits[cap_resolved] == "model_cpu_offload"
assert t2i._cuda_tier_limit_vram[cap_resolved] == stamp
assert t2i.cuda_tier_limit_for("/models/org/repo", stamp) == "model_cpu_offload"
t2i.clear_cuda_tier_limit("/models/org/repo")
bare = make_pipe("cuda")
assert t2i.lower_cuda_tier_after_oom(bare) is True
assert t2i._cuda_tier_limit_vram[cap_resolved] == resident
assert t2i.cuda_tier_limit_for("/models/org/repo", stamp) is None

def clear_boom(path):
    raise RuntimeError("cap clear failed\nsecond line")

chat.clear_cuda_tier_limit = clear_boom
chat.CHATBOT_MODEL_ID = "org/repo"
chat.CHATBOT = make_pipe()
cap_log = io.StringIO()
sys.stderr = cap_log
try:
    chat.chatbot_destroy()
finally:
    sys.stderr = old_err
assert "RuntimeError: cap clear failed" in cap_log.getvalue(), cap_log.getvalue()
assert "second line" not in cap_log.getvalue()

def release_boom():
    raise RuntimeError("cache release failed\nsecond line")

chat._release_torch_cache = release_boom
chat.chatbot_create = fake_create
chat.CHATBOT = make_pipe()
chat.CHATBOT_MODEL_ID = "org/repo"
chat.CHATBOT_PIPELINE_TAG = "text-to-image"
t2i._cuda_tier_limits.clear()
t2i._cuda_tier_limit_vram.clear()
box.clear()
chat.generate_text_to_image = generate_oom_stamp
cache_log = io.StringIO()
sys.stderr = cache_log
try:
    assert chat._generate_text_to_image_turn("a cat", None, None, None, "/tmp/out.png") == "/tmp/out.png"
finally:
    sys.stderr = old_err
assert "RuntimeError: cache release failed" in cache_log.getvalue(), cache_log.getvalue()
assert "second line" not in cache_log.getvalue()

creates.clear()
calls["n"] = 0
chat.CHATBOT = make_pipe()
chat.CHATBOT_MODEL_ID = "org/repo"
real_lower = chat.lower_cuda_tier_after_oom

def lower_and_stop(pipe):
    chat.CHAT_GENERATION_STOP = True
    return real_lower(pipe)

chat.lower_cuda_tier_after_oom = lower_and_stop

def generate_oom_again(pipe, prompt, output_path, image_path=None):
    calls["n"] += 1
    raise RuntimeError("CUDA out of memory")

chat.generate_text_to_image = generate_oom_again
assert chat._generate_text_to_image_turn("a cat", None, None, None, "/tmp/out.png") == ""
assert calls["n"] == 1, calls
assert creates == []
`;

const VAE_SCRIPT = String.raw`
import sys
import types
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))
worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker
download = types.ModuleType("worker.download")
download.models_cache_dir = lambda path=None: Path(path) if path else repo
sys.modules["worker.download"] = download

try:
    import torch
    import torch.nn as nn
    from diffusers.hooks import apply_group_offloading
except ImportError as exc:
    raise SystemExit("torch and diffusers 0.41.0 are required for the VAE offload test: " + str(exc))

import worker.t2i as t2i

def has_group_hook(module):
    registry = getattr(module, "_diffusers_hook", None)
    if registry is None:
        return False
    return registry.get_hook("group_offloading") is not None

class RecordingConv(nn.Conv2d):
    def __init__(self):
        super().__init__(1, 1, 1)
        self.forward_devices = []

    def forward(self, inputs):
        self.forward_devices.append(str(self.weight.device))
        return super().forward(inputs)

class TinyVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = RecordingConv()

    def forward(self, inputs):
        return self.conv(inputs)

    def decode(self, latents):
        return self.conv(latents)

    def encode(self, pixels):
        return self.conv(pixels)

class TinyUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, sample):
        return self.blocks[0](sample)

class Pipe:
    def __init__(self):
        self.vae = TinyVae()
        self.unet = TinyUnet()
        self.text_encoder = nn.Linear(4, 4)

    @property
    def components(self):
        return {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "unet": self.unet,
        }

old = TinyVae()
apply_group_offloading(
    old,
    onload_device=torch.device("cpu"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=1,
    use_stream=False,
)
assert has_group_hook(old.conv) is False

pipe = Pipe()
t2i._apply_group_offload(pipe, "block_level", pre_pin=False, onload_device=torch.device("cpu"))
assert has_group_hook(pipe.vae.conv) is True
assert has_group_hook(pipe.text_encoder) is True
assert has_group_hook(pipe.unet.blocks[0]) is True
latent = torch.zeros(1, 1, 4, 4)
decoded = pipe.vae.decode(latent)
encoded = pipe.vae.encode(latent)
assert tuple(decoded.shape) == (1, 1, 4, 4)
assert tuple(encoded.shape) == (1, 1, 4, 4)
assert pipe.vae.conv.forward_devices == ["cpu", "cpu"], pipe.vae.conv.forward_devices

from diffusers import AutoencoderKLFlux2

flux2 = AutoencoderKLFlux2(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",),
    up_block_types=("UpDecoderBlock2D",),
    block_out_channels=(32,),
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=4,
    sample_size=32,
    mid_block_add_attention=False,
    use_quant_conv=False,
    use_post_quant_conv=False,
)
flux_pipe = type("FluxPipeline", (), {})()
flux_pipe.vae = flux2

def flux_components(self):
    return {"vae": self.vae}

type(flux_pipe).components = property(flux_components)
assert t2i._pipeline_owns_vae_upcast(flux_pipe) is False
t2i._apply_group_offload(flux_pipe, "block_level", pre_pin=False, onload_device=torch.device("cpu"))
pixels = torch.zeros(1, 3, 32, 32)
with torch.no_grad():
    posterior = flux_pipe.vae.encode(pixels)
    flux_decoded = flux_pipe.vae.decode(posterior.latent_dist.mode())
assert tuple(flux_decoded.sample.shape) == (1, 3, 32, 32)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxImg2ImgPipeline,
    FluxPipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

def tiny_vae(dtype, force_upcast):
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=32,
        sample_size=32,
        mid_block_add_attention=False,
        force_upcast=force_upcast,
        shift_factor=0.0,
    )
    return vae.to(dtype=dtype)

class Stub(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = types.SimpleNamespace(**config)
        self.proj = nn.Linear(4, 4, bias=False)

    @property
    def dtype(self):
        return self.proj.weight.dtype

class Tok:
    model_max_length = 77

def flux_kwargs(vae):
    return {
        "scheduler": FlowMatchEulerDiscreteScheduler(),
        "vae": vae,
        "text_encoder": Stub(),
        "tokenizer": Tok(),
        "text_encoder_2": Stub(),
        "tokenizer_2": Tok(),
        "transformer": Stub(sample_size=32, patch_size=2),
    }

def sdxl_kwargs(vae):
    return {
        "vae": vae,
        "text_encoder": Stub(),
        "text_encoder_2": Stub(projection_dim=32),
        "tokenizer": Tok(),
        "tokenizer_2": Tok(),
        "unet": Stub(sample_size=32),
        "scheduler": DDIMScheduler(),
        "add_watermarker": False,
    }

def sd3_kwargs(vae):
    return {
        "transformer": Stub(sample_size=32, patch_size=2, joint_attention_dim=32),
        "scheduler": FlowMatchEulerDiscreteScheduler(),
        "vae": vae,
        "text_encoder": Stub(),
        "tokenizer": Tok(),
        "text_encoder_2": Stub(),
        "tokenizer_2": Tok(),
        "text_encoder_3": Stub(),
        "tokenizer_3": Tok(),
    }

# fp32 VAE + fp16 latents is the crash. The helper upcasts and casts.
bare = tiny_vae(torch.float16, force_upcast=True).to(dtype=torch.float32)
fp16_latents = torch.zeros(1, 4, 32, 32, dtype=torch.float16)
try:
    with torch.no_grad():
        bare.decode(fp16_latents)
    raise SystemExit("fp32 VAE accepted fp16 latents")
except RuntimeError as exc:
    assert "Half" in str(exc) or "half" in str(exc).lower() or "same" in str(exc), exc

flux = FluxPipeline(**flux_kwargs(tiny_vae(torch.float16, force_upcast=True)))
assert t2i._prepare_fp16_vae(flux, torch.float16) is True
assert flux.vae.dtype == torch.float32
assert t2i._prepare_fp16_vae(flux, torch.float16) is True
assert flux.vae._glaux_vae_io_cast is True
wrapped = flux.vae.decode
assert t2i._install_vae_io_cast(flux.vae, torch.float16) is False
assert flux.vae.decode is wrapped
with torch.no_grad():
    decoded = flux.vae.decode(fp16_latents, return_dict=False)[0]
    encoded = flux.vae.encode(torch.zeros(1, 3, 32, 32, dtype=torch.float16))
assert decoded.dtype == torch.float32, decoded.dtype
assert encoded.latent_dist.sample().dtype == torch.float16
assert encoded.latent_dist.mode().dtype == torch.float16
assert t2i._enable_vae_tiling(flux) is True
flux.vae.tile_latent_min_size = 16
flux.vae.tile_sample_min_size = 16
assert flux.vae.use_tiling is True
with torch.no_grad():
    tiled = flux.vae.decode(fp16_latents, return_dict=False)[0]
assert tiled.dtype == torch.float32
assert tuple(tiled.shape) == tuple(decoded.shape)
# The CPU/MPS loader calls pipe.to after the wrapper. The wrapper is an
# attribute of the VAE, so the move must leave decode and encode in place.
moved = flux.to("cpu")
assert moved.vae._glaux_vae_io_cast is True
with torch.no_grad():
    moved_image = moved.vae.decode(fp16_latents, return_dict=False)[0]
assert moved_image.dtype == torch.float32

flux_i2i = FluxImg2ImgPipeline(**flux_kwargs(tiny_vae(torch.float16, force_upcast=True)))
assert t2i._prepare_fp16_vae(flux_i2i, torch.float16) is True
with torch.no_grad():
    i2i_latents = flux_i2i._encode_vae_image(
        torch.zeros(1, 3, 32, 32, dtype=torch.float16), None
    )
assert i2i_latents.dtype == torch.float16, i2i_latents.dtype

t2i._apply_group_offload(flux, "leaf_level", pre_pin=False, onload_device=torch.device("cpu"))
with torch.no_grad():
    hooked = flux.vae.decode(fp16_latents, return_dict=False)[0]
    hooked_enc = flux.vae.encode(torch.zeros(1, 3, 32, 32, dtype=torch.float16))
assert hooked.dtype == torch.float32
assert hooked_enc.latent_dist.mode().dtype == torch.float16

# SDXL keeps fp16 and uses its own upcast, including under leaf group offload.
sdxl = StableDiffusionXLPipeline(**sdxl_kwargs(tiny_vae(torch.float16, force_upcast=True)))
assert t2i._pipeline_owns_vae_upcast(sdxl) is True
assert t2i._prepare_fp16_vae(sdxl, torch.float16) is False
assert sdxl.vae.dtype == torch.float16
t2i._apply_group_offload(sdxl, "leaf_level", pre_pin=False, onload_device=torch.device("cpu"))
assert sdxl.vae.dtype == torch.float16
needs = sdxl.vae.dtype == torch.float16 and sdxl.vae.config.force_upcast
assert needs is True
sdxl.upcast_vae()
cast_dtype = next(iter(sdxl.vae.post_quant_conv.parameters())).dtype
assert cast_dtype == torch.float32
with torch.no_grad():
    sdxl_image = sdxl.vae.decode(fp16_latents.to(dtype=cast_dtype), return_dict=False)[0]
assert sdxl_image.dtype == torch.float32, sdxl_image.dtype
sdxl.vae.to(dtype=torch.float16)
assert sdxl.vae.dtype == torch.float16

sdxl_i2i = StableDiffusionXLImg2ImgPipeline(**sdxl_kwargs(tiny_vae(torch.float16, force_upcast=True)))
assert t2i._prepare_fp16_vae(sdxl_i2i, torch.float16) is False
image = torch.zeros(1, 3, 32, 32, dtype=torch.float16)
image = image.float()
sdxl_i2i.vae.to(dtype=torch.float32)
with torch.no_grad():
    init_latents = sdxl_i2i.vae.encode(image).latent_dist.sample()
sdxl_i2i.vae.to(dtype=torch.float16)
init_latents = init_latents.to(dtype=torch.float16)
assert init_latents.dtype == torch.float16

# A streamed host copy restores fp16 after upcast_vae. That is why the VAE
# of an upcast pipeline is group-offloaded without a stream.
stream_vae = tiny_vae(torch.float16, force_upcast=True)
snapshot = {name: param.detach().clone() for name, param in stream_vae.named_parameters()}
stream_vae.to(dtype=torch.float32)
for name, param in stream_vae.named_parameters():
    param.data = snapshot[name]
try:
    with torch.no_grad():
        stream_vae.decode(fp16_latents.to(dtype=torch.float32))
    raise SystemExit("restored fp16 weights accepted fp32 latents")
except RuntimeError as exc:
    assert "same" in str(exc).lower() or "float" in str(exc).lower(), exc

sd3 = StableDiffusion3Pipeline(**sd3_kwargs(tiny_vae(torch.float16, force_upcast=True)))
assert t2i._pipeline_owns_vae_upcast(sd3) is False
assert t2i._prepare_fp16_vae(sd3, torch.float16) is True
with torch.no_grad():
    sd3_image = sd3.vae.decode(fp16_latents, return_dict=False)[0]
assert sd3_image.dtype == torch.float32
sd3_i2i = StableDiffusion3Img2ImgPipeline(**sd3_kwargs(tiny_vae(torch.float16, force_upcast=True)))
assert t2i._prepare_fp16_vae(sd3_i2i, torch.float16) is True
with torch.no_grad():
    sd3_latents = sd3_i2i.vae.encode(torch.zeros(1, 3, 32, 32, dtype=torch.float16)).latent_dist.mode()
assert sd3_latents.dtype == torch.float16, sd3_latents.dtype
`;

function runPython(script, label, command = PYTHON) {
  const result = spawnSync(command, ['-', REPO], {
    input: script,
    encoding: 'utf8',
    windowsHide: true,
  });
  assert.equal(
    result.status,
    0,
    `${label}\n${result.stderr || result.stdout || result.error?.message || `${command} failed`}`
  );
}

/**
 * The engine tests use the vendor runtime. That build may not have torch.
 * The VAE test needs a real torch and diffusers, so it uses the first
 * interpreter that can import both. Python itself missing still skips via
 * PYTHON_SKIP, which is the existing convention in this file.
 * @returns {string | false}
 */
function pythonWithModules(statement) {
  const candidates = [PYTHON, process.platform === 'win32' ? 'python' : 'python3'];
  const seen = new Set();
  for (const command of candidates) {
    if (!command || seen.has(command)) continue;
    seen.add(command);
    const result = spawnSync(command, ['-c', statement], {
      encoding: 'utf8',
      windowsHide: true,
      timeout: 30000,
    });
    if (result.status === 0) return command;
  }
  return false;
}

function pythonWithTorch() {
  return pythonWithModules('import torch, diffusers');
}

const TORCH_PYTHON = pythonWithTorch();
const TORCH_SKIP = TORCH_PYTHON
  ? false
  : 'No Python interpreter can import torch and diffusers. The VAE leaf-offload test needs both.';
const ACCEL_PYTHON = pythonWithModules('import torch, accelerate');
const ACCEL_SKIP = ACCEL_PYTHON
  ? false
  : 'No Python interpreter can import torch and accelerate. The GPU-budget test needs both.';

const ACCEL_SCRIPT = String.raw`
import sys
import types
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))
worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker
download = types.ModuleType("worker.download")
download.models_cache_dir = lambda path=None: Path(path) if path else repo
sys.modules["worker.download"] = download

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import get_max_layer_size, infer_auto_device_map, compute_module_sizes
import worker.t2i as t2i

class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin = nn.Linear(width, width, bias=False)

class Toy(nn.Module):
    _no_split_modules = ["Block"]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(64), Block(64)])

with init_empty_weights():
    model = Toy()

module_sizes = compute_module_sizes(model, dtype=torch.float32)
max_layer, _names = get_max_layer_size(list(model.named_modules()), module_sizes, ["Block"])
floor = t2i.gpu_shard_floor(max_layer, max_layer)
assert floor == max_layer * 2, (floor, max_layer)
assert max_layer > 0

def on_gpu(budget):
    mapping = infer_auto_device_map(
        model,
        max_memory={0: budget, "cpu": 10**12},
        no_split_module_classes=["Block"],
        dtype=torch.float32,
    )
    return any(device == 0 for device in mapping.values()), mapping

below, below_map = on_gpu(floor - 1)
assert below is False, below_map
placed, placed_map = on_gpu(floor)
assert placed is True, placed_map
`;

describe('text-to-image disk offload', () => {
  it('offloads CUDA and CPU loads to a real disk and tiles the VAE', { skip: PYTHON_SKIP }, () => {
    const result = spawnSync(PYTHON, ['-', REPO], {
      input: SCRIPT,
      encoding: 'utf8',
      windowsHide: true,
    });
    assert.equal(
      result.status,
      0,
      result.stderr || result.stdout || result.error?.message || `${PYTHON} failed`
    );
  });

  it('keeps sampling tensors off the meta device', { skip: PYTHON_SKIP }, () => {
    const result = spawnSync(PYTHON, ['-', REPO], {
      input: SAMPLING_SCRIPT,
      encoding: 'utf8',
      windowsHide: true,
    });
    assert.equal(
      result.status,
      0,
      result.stderr || result.stdout || result.error?.message || `${PYTHON} failed`
    );
  });

  it('selects a CUDA tier, budgets, and an OOM fallback without a GPU', { skip: PYTHON_SKIP }, () => {
    const result = spawnSync(PYTHON, ['-', REPO], {
      input: PLACEMENT_SCRIPT,
      encoding: 'utf8',
      windowsHide: true,
    });
    assert.equal(
      result.status,
      0,
      result.stderr || result.stdout || result.error?.message || `${PYTHON} failed`
    );
  });

  it('sizes SDXL and Flux checkpoints from the files diffusers loads', { skip: PYTHON_SKIP }, () => {
    runPython(HEADER_SCRIPT, 'header sizing');
  });

  it('places group offload, retries pinning, and keeps a tier cap', { skip: PYTHON_SKIP }, () => {
    runPython(APPLY_SCRIPT, 'placement hooks');
  });

  it('frees a failed pipeline before the next try and does not retry a stop', { skip: PYTHON_SKIP }, () => {
    runPython(CHAT_SCRIPT, 'chat retry');
  });

  it('leaf-offloads VAE decode and encode when block modules are absent', { skip: PYTHON_SKIP || TORCH_SKIP }, () => {
    runPython(VAE_SCRIPT, 'vae leaf offload', TORCH_PYTHON);
  });

  it('matches accelerate reservation with the GPU shard floor', { skip: PYTHON_SKIP || ACCEL_SKIP }, () => {
    runPython(ACCEL_SCRIPT, 'accelerate device map', ACCEL_PYTHON);
  });
});
