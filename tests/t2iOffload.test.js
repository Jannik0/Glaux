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

wide = t2i._component_disk_budgets(
    {"transformer": 12 * GiB, "text_encoder": 5 * GiB, "vae": 100},
    device="cuda",
    available_ram=32 * GiB,
    free_vram=20 * GiB,
    cuda_visible=True,
)
assert wide["transformer"][0] == 20 * GiB - head - 100
assert wide["text_encoder"][0] == 0
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
assert tight["text_encoder"][0] == 0

cpu_budgets = t2i._component_disk_budgets(
    {"unet": 4 * GiB},
    device="cpu",
    available_ram=16 * GiB,
    free_vram=24 * GiB,
    cuda_visible=True,
)
assert cpu_budgets["unet"][0] == 0
assert cpu_budgets["unet"]["cpu"] == 4 * GiB
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
    "Text-to-image generation: 4 steps, average 2.50s/step, total 10.0s."
)

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
});
