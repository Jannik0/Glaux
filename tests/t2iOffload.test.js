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

assert t2i._offload_disk_bytes(100, DType(4)) == 200
assert t2i._offload_disk_bytes(100, DType(2)) == 100
assert t2i._weights_need_disk_offload("cuda", 1000, 600, DType(2)) is False
assert t2i._weights_need_disk_offload("cuda", 1000, 701, DType(2)) is True
assert t2i._weights_need_disk_offload("cuda", 1000, 700, DType(2)) is False
assert t2i._weights_need_disk_offload("cpu", 1000, 400, DType(4)) is True
assert t2i._weights_need_disk_offload("cpu", 1000, 400, DType(2)) is False
assert t2i._weights_need_disk_offload("cpu", 1000, 500, DType(2)) is False
# 600 of 1000 is over the CPU 0.55 share and under the CUDA 0.70 share.
assert t2i._weights_need_disk_offload("cpu", 1000, 600, DType(2)) is True
assert t2i._CPU_RAM_RESIDENT_FRACTION == 0.55
assert t2i._CPU_ACCELERATE_RAM_FRACTION == 0.28
assert t2i._weights_need_disk_offload("mps", 1000, 900, DType(2)) is False
assert t2i._weights_need_disk_offload("cuda", None, 900, DType(2)) is False

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

if not t2i._is_ram_disk(repo):
    base = Path(tempfile.mkdtemp(prefix="glaux-offload-", dir=repo))
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
assert t2i.select_cuda_tier({"unet": head}, head + head, 10 * GiB, dtype) == "cuda"
# Largest component fits, the total does not, and the total still fits in RAM.
assert t2i.select_cuda_tier(
    {"transformer": head, "text_encoder": head},
    head + head,
    10 * GiB,
    dtype,
) == "model_cpu_offload"
# Largest would fit in VRAM, but the checkpoint does not fit the RAM budget,
# so it must not be staged with from_pretrained.
assert t2i.select_cuda_tier({"unet": 1000}, head + 5000, 1000, dtype) == "disk"
# Fits in RAM, not in VRAM, and usable VRAM can hold a prefetched block.
assert t2i.select_cuda_tier(
    {"transformer": block_min + 1},
    head + block_min,
    10 * GiB,
    dtype,
) == "group_offload"
# Same, but the usable VRAM is below one prefetched block.
assert t2i.select_cuda_tier(
    {"transformer": 10 * GiB},
    head + block_min - 1,
    100 * GiB,
    dtype,
) == "group_offload_leaf"
# No free VRAM left after the headroom, and the weights fit in RAM: leaf groups.
assert t2i.select_cuda_tier({"unet": GiB}, head, 10 * GiB, dtype) == "group_offload_leaf"
# Unknown RAM does not force disk. Bytes outside component folders still count.
assert t2i.select_cuda_tier({"unet": 100}, head + 10_000, None, dtype, extra_bytes=100) == "cuda"
assert t2i.select_cuda_tier({}, 0, 1000, dtype, extra_bytes=800) == "disk"
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
assert t2i._denoiser_gpu_budget(t2i._MIN_GPU_OFFLOAD_BYTES + t2i._GPU_SLIVER_BYTES + 4096, 0) == (
    t2i._MIN_GPU_OFFLOAD_BYTES
)

wide = t2i._component_disk_budgets(
    {"transformer": 12 * GiB, "text_encoder": 5 * GiB, "vae": 100},
    device="cuda",
    available_ram=32 * GiB,
    free_vram=20 * GiB,
    cuda_visible=True,
)
denoiser_spare = 20 * GiB - head - 100
assert wide["text_encoder"][0] == 5 * GiB
assert wide["transformer"][0] == denoiser_spare - 5 * GiB
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
)
assert tight["transformer"]["cpu"] == tight_budget - t2i._CPU_BUDGET_FLOOR
assert tight["text_encoder"]["cpu"] == t2i._CPU_BUDGET_FLOOR
assert tight["text_encoder"][0] == 2 * GiB
assert tight["transformer"][0] == 24 * GiB - head - 2 * GiB

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

partial_encoder = t2i._component_disk_budgets(
    {"transformer": 20 * GiB, "text_encoder": 30 * GiB},
    device="cuda",
    available_ram=80 * GiB,
    free_vram=8 * GiB,
    cuda_visible=True,
)
assert partial_encoder["text_encoder"][0] == t2i._MIN_GPU_OFFLOAD_BYTES
assert partial_encoder["transformer"][0] == 5 * GiB - t2i._MIN_GPU_OFFLOAD_BYTES

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

if not t2i._is_ram_disk(repo):
    base = Path(tempfile.mkdtemp(prefix="glaux-tier-", dir=str(repo)))
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

    base = Path(tempfile.mkdtemp(prefix="glaux-tier-keep-", dir=str(repo)))
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
t2i.clear_cuda_tier_limit("/models/sticky")
assert "/models/sticky" not in t2i._cuda_tier_limits

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
sdxl_tier = t2i.select_cuda_tier(sdxl_sizes, 24 * GiB, 64 * GiB, fp16)
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
write_header(flux / "text_encoder_2" / "model.safetensors", [
    ("w", "F16", [numel_for(40 * GiB, 2)]),
])
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
flux_tier = t2i.select_cuda_tier(flux_sizes, 24 * GiB, 64 * GiB, fp16)
assert flux_tier == "model_cpu_offload", (flux_tier, flux_sizes)
fp32 = DType(4)
flux_fp32 = t2i.measure_component_bytes(str(flux), fp32)
assert flux_fp32["transformer"] == 32 * GiB, flux_fp32
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

def place_spy(model_path, dtype, tier, cast, *, available_ram, free_vram, pre_pin=None):
    resolved = str(Path(model_path).resolve())
    resolved_holder["path"] = resolved
    seen.append((tier, pre_pin, t2i._cuda_tier_limits.get(resolved)))
    if pre_pin is True:
        raise RuntimeError("CUDA error: out of memory")
    return FakePipe()

t2i._place_cuda_tier = place_spy
t2i.measure_component_bytes = lambda path, dtype: {"unet": 100}
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
assert resolved not in t2i._cuda_tier_limits
assert "low_cpu_mem_usage=True" in buf.getvalue()
assert isinstance(loaded, FakePipe)
host_calls.clear()
t2i._empty_host_cache()
assert host_calls == ["host"]
`;

const CHAT_SCRIPT = String.raw`
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
assert chat._generate_text_to_image_turn("a cat", None, None, None, "/tmp/out.png") == ""
assert calls["n"] == 1
assert creates == []

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
function pythonWithTorch() {
  const candidates = [PYTHON, process.platform === 'win32' ? 'python' : 'python3'];
  const seen = new Set();
  for (const command of candidates) {
    if (!command || seen.has(command)) continue;
    seen.add(command);
    const result = spawnSync(command, ['-c', 'import torch, diffusers'], {
      encoding: 'utf8',
      windowsHide: true,
      timeout: 30000,
    });
    if (result.status === 0) return command;
  }
  return false;
}

const TORCH_PYTHON = pythonWithTorch();
const TORCH_SKIP = TORCH_PYTHON
  ? false
  : 'No Python interpreter can import torch and diffusers. The VAE leaf-offload test needs both.';

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

  it('places group offload, retries pinning, and clears a tier cap', { skip: PYTHON_SKIP }, () => {
    runPython(APPLY_SCRIPT, 'placement hooks');
  });

  it('frees a failed pipeline before the next try and does not retry a stop', { skip: PYTHON_SKIP }, () => {
    runPython(CHAT_SCRIPT, 'chat retry');
  });

  it('leaf-offloads VAE decode and encode when block modules are absent', { skip: PYTHON_SKIP || TORCH_SKIP }, () => {
    runPython(VAE_SCRIPT, 'vae leaf offload', TORCH_PYTHON);
  });
});
