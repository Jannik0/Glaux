'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('child_process');
const path = require('path');

const REPO = path.resolve(__dirname, '..');

// Imports worker/t2i.py without worker/__init__.py, which pulls in torch.
const SCRIPT = String.raw`
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

def only_tmp_is_ram(path):
    resolved = Path(path).resolve()
    tmp = Path("/tmp").resolve()
    return resolved == tmp or tmp in resolved.parents

real_is_ram = t2i._is_ram_disk
t2i._is_ram_disk = only_tmp_is_ram
try:
    state["cache"] = Path("/tmp")
    chosen = t2i._disk_offload_root("/tmp/org/repo")
    assert not only_tmp_is_ram(chosen.parent), chosen
    assert chosen == Path("/var/tmp") / ".glaux-t2i-offload", chosen
finally:
    t2i._is_ram_disk = real_is_ram

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
        root = t2i._prepare_disk_offload(str(base / "org" / "repo"), 1)
        assert root == base / ".glaux-t2i-offload", root
        assert root.is_dir()
        assert not t2i._is_ram_disk(root.parent)
        (root / "weight.dat").write_text("x", encoding="utf-8")
        t2i.release_disk_offload()
        assert not root.exists()
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

import torch
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
  it('offloads CUDA and CPU loads to a real disk and tiles the VAE', () => {
    const result = spawnSync('python3', ['-', REPO], {
      input: SCRIPT,
      encoding: 'utf8',
    });
    assert.equal(
      result.status,
      0,
      result.stderr || result.stdout || result.error?.message || 'python3 failed'
    );
  });

  it('keeps sampling tensors off the meta device', () => {
    const python = path.join(REPO, 'vendor', 'python', 'bin', 'python3');
    const result = spawnSync(python, ['-', REPO], {
      input: SAMPLING_SCRIPT,
      encoding: 'utf8',
    });
    assert.equal(
      result.status,
      0,
      result.stderr || result.stdout || result.error?.message || 'python3 failed'
    );
  });
});
