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
assert t2i._weights_need_disk_offload("cuda", 1000, 600, DType(2)) is True
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
});
