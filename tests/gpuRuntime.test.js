'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { isForceCpu, withForceCpuTorchEnv, withVendorLibPath, withSharedCudaLibPath } = require('../engines/common/gpuRuntime');
const { expectedGpuBackends, findBackendModule, findNvcc, isCuda13RedistName, isDroppedCudaDepName, missingSharedCudaRedists, withCudaToolkitEnv, copyFile, collapseDuplicateLibs, collapseDuplicateLibsRecursive, cudaArchitectureCmakeArgs, GGML_CUDA_ARCHITECTURES, shareTorchCuda13WithVendor, shareGgmlCudaBackend, ggmlCudaBackendFileName, which, requirePatchelf, elfNeeded, removeDroppedElfNeeded } = require('../scripts/gpuBackends');
const { isStubbedCudaFamilyFile, readPeImportsAndExports, stubTorchUnusedCudaDeps } = require('../scripts/cudaStubs');

describe('gpuRuntime', () => {
  it('treats 1/true/yes as force-cpu', () => {
    assert.equal(isForceCpu({ GLAUX_FORCE_CPU: '1' }), true);
    assert.equal(isForceCpu({ GLAUX_FORCE_CPU: 'true' }), true);
    assert.equal(isForceCpu({ GLAUX_FORCE_CPU: 'YES' }), true);
  });

  it('does not force CPU when unset or other values', () => {
    assert.equal(isForceCpu({}), false);
    assert.equal(isForceCpu({ GLAUX_FORCE_CPU: '' }), false);
    assert.equal(isForceCpu({ GLAUX_FORCE_CPU: '0' }), false);
    assert.equal(isForceCpu({ GLAUX_FORCE_CPU: 'false' }), false);
  });

  it('hides CUDA/HIP for the Hugging Face worker when forcing CPU', () => {
    const env = withForceCpuTorchEnv({ GLAUX_FORCE_CPU: '1', PATH: 'rest' });
    assert.equal(env.CUDA_VISIBLE_DEVICES, '');
    assert.equal(env.HIP_VISIBLE_DEVICES, '');
    assert.equal(env.PATH, 'rest');
  });

  it('leaves CUDA visibility unchanged when not forcing CPU', () => {
    const env = withForceCpuTorchEnv({ CUDA_VISIBLE_DEVICES: '0', PATH: 'rest' });
    assert.equal(env.CUDA_VISIBLE_DEVICES, '0');
    assert.equal(env.HIP_VISIBLE_DEVICES, undefined);
  });

  it('prepends the vendor dir to PATH', () => {
    const env = withVendorLibPath({ PATH: 'rest' }, '/tmp/vendor');
    assert.match(env.PATH, /vendor/);
    assert.match(env.PATH, /rest/);
  });

  it('prepends an existing shared CUDA dir and sets GLAUX_CUDA_DIR', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-cuda-'));
    try {
      const env = withSharedCudaLibPath({ PATH: 'rest' }, dir);
      assert.ok(env.PATH.startsWith(path.resolve(dir)));
      assert.match(env.PATH, /rest/);
      assert.equal(env.GLAUX_CUDA_DIR, path.resolve(dir));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('does not change PATH when the shared CUDA dir is missing', () => {
    const missing = path.join(os.tmpdir(), `glaux-no-cuda-${Date.now()}`);
    const env = withSharedCudaLibPath({ PATH: 'rest' }, missing);
    assert.equal(env.PATH, 'rest');
    assert.equal(env.GLAUX_CUDA_DIR, undefined);
  });
});

describe('expectedGpuBackends', () => {
  it('ships CUDA and Vulkan on Windows and Linux', () => {
    assert.deepEqual(expectedGpuBackends('win32'), ['cuda', 'vulkan']);
    assert.deepEqual(expectedGpuBackends('linux'), ['cuda', 'vulkan']);
  });

  it('ships Metal on macOS', () => {
    assert.deepEqual(expectedGpuBackends('darwin'), ['metal']);
  });
});

describe('findNvcc', () => {
  it('points cmake at toolkit nvcc even when it is not on PATH', () => {
    const nvcc = findNvcc();
    if (!nvcc) {
      return;
    }
    assert.match(nvcc, /nvcc(\.exe)?$/i);
    assert.equal(fs.existsSync(nvcc), true);
    const env = withCudaToolkitEnv({ PATH: 'rest' });
    assert.ok(env.PATH.startsWith(path.dirname(nvcc)));
    assert.equal(env.CUDACXX, nvcc);
  });
});

describe('findBackendModule', () => {
  it('finds ggml backend modules by stem', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-gpu-'));
    try {
      fs.writeFileSync(path.join(dir, 'ggml-cuda.dll'), '');
      fs.writeFileSync(path.join(dir, 'libggml-cpu.so'), '');
      assert.ok(findBackendModule(dir, 'cuda'));
      assert.ok(findBackendModule(dir, 'cpu'));
      assert.equal(findBackendModule(dir, 'vulkan'), null);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('CUDA 13 redistributable names', () => {
  it('accepts CUDA 13 runtime filenames', () => {
    assert.equal(isCuda13RedistName('cudart64_13.dll'), true);
    assert.equal(isCuda13RedistName('cublas64_13.dll'), true);
    assert.equal(isCuda13RedistName('cublasLt64_13.dll'), true);
    assert.equal(isCuda13RedistName('nvJitLink_130_0.dll'), true);
    assert.equal(isCuda13RedistName('libcudart.so.13'), true);
    assert.equal(isCuda13RedistName('libcublas.so.13.0.0'), true);
  });

  it('rejects CUDA 12 runtime filenames', () => {
    assert.equal(isCuda13RedistName('cudart64_12.dll'), false);
    assert.equal(isCuda13RedistName('cublasLt64_12.dll'), false);
    assert.equal(isCuda13RedistName('nvJitLink_120_0.dll'), false);
    assert.equal(isCuda13RedistName('libcudart.so.12'), false);
  });

  it('reports missing shared CUDA files', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-cuda-check-'));
    try {
      assert.ok(missingSharedCudaRedists(dir).length > 0);
      if (process.platform === 'win32') {
        fs.writeFileSync(path.join(dir, 'cudart64_13.dll'), '');
        fs.writeFileSync(path.join(dir, 'cublas64_13.dll'), '');
      } else {
        fs.writeFileSync(path.join(dir, 'libcudart.so.13'), '');
        fs.writeFileSync(path.join(dir, 'libcublas.so.13'), '');
      }
      assert.deepEqual(missingSharedCudaRedists(dir), []);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('copyFile symlinks', () => {
  it('recreates relative ELF SONAME links instead of copying the target', () => {
    if (process.platform === 'win32') {
      return;
    }
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-copy-'));
    const destDir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-copy-dest-'));
    try {
      const real = path.join(dir, 'libfoo.so.1.2.3');
      fs.writeFileSync(real, 'payload-bytes');
      fs.symlinkSync('libfoo.so.1.2.3', path.join(dir, 'libfoo.so.1'));
      fs.symlinkSync('libfoo.so.1', path.join(dir, 'libfoo.so'));
      copyFile(path.join(dir, 'libfoo.so.1.2.3'), path.join(destDir, 'libfoo.so.1.2.3'));
      copyFile(path.join(dir, 'libfoo.so.1'), path.join(destDir, 'libfoo.so.1'));
      copyFile(path.join(dir, 'libfoo.so'), path.join(destDir, 'libfoo.so'));
      assert.equal(fs.lstatSync(path.join(destDir, 'libfoo.so.1')).isSymbolicLink(), true);
      assert.equal(fs.readlinkSync(path.join(destDir, 'libfoo.so.1')), 'libfoo.so.1.2.3');
      assert.equal(fs.readFileSync(path.join(destDir, 'libfoo.so'), 'utf8'), 'payload-bytes');
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
      fs.rmSync(destDir, { recursive: true, force: true });
    }
  });
});

describe('collapseDuplicateLibs', () => {
  it('turns identical SONAME copies into relative symlinks', () => {
    if (process.platform === 'win32') {
      return;
    }
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-collapse-'));
    try {
      const payload = Buffer.alloc(2048, 7);
      fs.writeFileSync(path.join(dir, 'libfoo.so.1.2.3'), payload);
      fs.writeFileSync(path.join(dir, 'libfoo.so.1'), payload);
      fs.writeFileSync(path.join(dir, 'libfoo.so'), payload);
      fs.writeFileSync(path.join(dir, 'libbar.so.1'), Buffer.alloc(2048, 9));
      assert.equal(collapseDuplicateLibs(dir), 2);
      assert.equal(fs.lstatSync(path.join(dir, 'libfoo.so.1.2.3')).isSymbolicLink(), false);
      assert.equal(fs.readlinkSync(path.join(dir, 'libfoo.so')), 'libfoo.so.1.2.3');
      assert.equal(fs.readlinkSync(path.join(dir, 'libfoo.so.1')), 'libfoo.so.1.2.3');
      assert.equal(fs.lstatSync(path.join(dir, 'libbar.so.1')).isSymbolicLink(), false);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('walks nested lib directories', () => {
    if (process.platform === 'win32') {
      return;
    }
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-collapse-r-'));
    try {
      const nested = path.join(root, 'nvidia', 'cublas', 'lib');
      fs.mkdirSync(nested, { recursive: true });
      const payload = Buffer.alloc(2048, 3);
      fs.writeFileSync(path.join(nested, 'libcublas.so.13.1.1'), payload);
      fs.writeFileSync(path.join(nested, 'libcublas.so.13'), payload);
      fs.writeFileSync(path.join(nested, 'libcublas.so'), payload);
      assert.equal(collapseDuplicateLibsRecursive(root), 2);
      assert.equal(fs.readlinkSync(path.join(nested, 'libcublas.so')), 'libcublas.so.13.1.1');
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});

describe('dropped CUDA extras', () => {
  it('keeps inference libraries and leaves NCCL for the loader stub', () => {
    assert.equal(isDroppedCudaDepName('libnccl.so.2'), false);
    assert.equal(isDroppedCudaDepName('libnvshmem.so'), false);
    assert.equal(isDroppedCudaDepName('libcusparseLt.so.0'), false);
    assert.equal(isDroppedCudaDepName('libcupti.so.13'), false);
    assert.equal(isDroppedCudaDepName('libcufile.so.0'), false);
    assert.equal(isDroppedCudaDepName('libnvtx.so.1'), true);
    assert.equal(isDroppedCudaDepName('nvtx.py'), false);
    assert.equal(isDroppedCudaDepName('nvtx.pyi'), false);
    assert.equal(isDroppedCudaDepName('nvToolsExt64_1.dll'), true);
    assert.equal(isDroppedCudaDepName('libcudnn.so.9'), false);
    assert.equal(isDroppedCudaDepName('libcublas.so.13'), false);
    assert.equal(isDroppedCudaDepName('libcudart.so.13'), false);
    assert.equal(isDroppedCudaDepName('libcufft.so.12'), false);
  });
});

describe('removeDroppedElfNeeded', () => {
  it('is a no-op off Linux', () => {
    if (process.platform === 'linux') {
      return;
    }
    assert.equal(removeDroppedElfNeeded(os.tmpdir()), 0);
  });

  it('requires patchelf on Linux', () => {
    if (process.platform !== 'linux') {
      return;
    }
    assert.ok(requirePatchelf());
  });

  it('uses patchelf to drop NVTX DT_NEEDED', () => {
    if (process.platform !== 'linux') {
      return;
    }
    const gcc = which('gcc');
    if (!gcc) {
      return;
    }
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-elf-needed-'));
    const src = path.join(dir, 'foo.c');
    const nested = path.join(dir, 'torch', 'lib');
    const so = path.join(nested, 'libfoo.so');
    try {
      fs.mkdirSync(nested, { recursive: true });
      fs.writeFileSync(src, 'int foo(void) { return 1; }\n');
      const compiled = spawnSync(gcc, ['-shared', '-fPIC', '-o', so, src], { encoding: 'utf8' });
      assert.equal(compiled.status, 0, compiled.stderr);
      const patchelf = requirePatchelf();
      assert.equal(spawnSync(patchelf, ['--add-needed', 'libnvtx.so.1', so]).status, 0);
      assert.equal(spawnSync(patchelf, ['--add-needed', 'libcudart.so.13', so]).status, 0);
      assert.ok(elfNeeded(so).includes('libnvtx.so.1'));
      assert.ok(elfNeeded(so).includes('libcudart.so.13'));
      assert.equal(removeDroppedElfNeeded(dir), 1);
      const needed = elfNeeded(so);
      assert.equal(needed.includes('libnvtx.so.1'), false);
      assert.ok(needed.includes('libcudart.so.13'));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('cudaArchitectureCmakeArgs', () => {
  it('pins a shipping GPU list when CUDA is on', () => {
    const args = cudaArchitectureCmakeArgs({ cuda: true });
    assert.ok(args.some((arg) => arg.includes('CMAKE_CUDA_ARCHITECTURES')));
    assert.ok(args.some((arg) => arg.includes(GGML_CUDA_ARCHITECTURES.split(';')[0])));
    assert.equal(cudaArchitectureCmakeArgs({ cuda: false }).length, 0);
  });
});

describe('shareTorchCuda13WithVendor', () => {
  it('points nvidia/cu13 overlapping redists at the shared CUDA dir', () => {
    if (process.platform === 'win32') {
      return;
    }
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-share-cuda-'));
    try {
      const cudaDir = path.join(root, 'cuda');
      const nvidiaLib = path.join(
        root,
        'python',
        'lib',
        'python3.14',
        'site-packages',
        'nvidia',
        'cu13',
        'lib'
      );
      fs.mkdirSync(cudaDir, { recursive: true });
      fs.mkdirSync(nvidiaLib, { recursive: true });
      fs.writeFileSync(path.join(cudaDir, 'libcublasLt.so.13.8.0.4'), 'toolkit-cublasLt');
      fs.symlinkSync('libcublasLt.so.13.8.0.4', path.join(cudaDir, 'libcublasLt.so.13'));
      fs.writeFileSync(path.join(nvidiaLib, 'libcublasLt.so.13'), 'wheel-cublasLt');
      fs.writeFileSync(path.join(nvidiaLib, 'libcufft.so.12'), 'keep-cufft');
      const replaced = shareTorchCuda13WithVendor(path.join(root, 'python'), cudaDir);
      assert.ok(replaced.includes('libcublasLt.so.13'));
      assert.equal(fs.lstatSync(path.join(nvidiaLib, 'libcublasLt.so.13')).isSymbolicLink(), true);
      assert.equal(fs.readFileSync(path.join(nvidiaLib, 'libcublasLt.so.13'), 'utf8'), 'toolkit-cublasLt');
      assert.equal(fs.readFileSync(path.join(nvidiaLib, 'libcufft.so.12'), 'utf8'), 'keep-cufft');
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});

describe('cuda loader stubs', () => {
  it('matches NCCL, cuSPARSELt, NVSHMEM, and cuFile on Windows and Linux', () => {
    assert.equal(isStubbedCudaFamilyFile('libnccl.so.2'), true);
    assert.equal(isStubbedCudaFamilyFile('nccl.dll'), true);
    assert.equal(isStubbedCudaFamilyFile('libcusparseLt.so.0'), true);
    assert.equal(isStubbedCudaFamilyFile('cusparseLt64_0.dll'), true);
    assert.equal(isStubbedCudaFamilyFile('libnvshmem_host.so.3'), true);
    assert.equal(isStubbedCudaFamilyFile('libnvshmem_device.bc'), true);
    assert.equal(isStubbedCudaFamilyFile('nvshmem_bootstrap_uid.so.3'), true);
    assert.equal(isStubbedCudaFamilyFile('libcufile.so.0'), true);
    assert.equal(isStubbedCudaFamilyFile('cufile.dll'), true);
    assert.equal(isStubbedCudaFamilyFile('libcufile_rdma.so.1'), true);
    assert.equal(isStubbedCudaFamilyFile('libtorch_nvshmem.so'), false);
    assert.equal(isStubbedCudaFamilyFile('libcusparse.so.12'), false);
    assert.equal(isStubbedCudaFamilyFile('libcudnn.so.9'), false);
    assert.equal(isStubbedCudaFamilyFile('libcufft.so.12'), false);
    assert.equal(isStubbedCudaFamilyFile('nvtx.py'), false);
  });

  it('reads PE import symbol names the same way Windows stubs are chosen', () => {
    const buf = Buffer.alloc(0x800, 0);
    buf[0] = 0x4d;
    buf[1] = 0x5a;
    buf.writeUInt32LE(0x80, 0x3c);
    buf.writeUInt32LE(0x4550, 0x80);
    buf.writeUInt16LE(1, 0x84 + 2);
    buf.writeUInt16LE(224, 0x84 + 16);
    buf.writeUInt16LE(0x10b, 0x98);
    buf.writeUInt32LE(16, 0x98 + 92);
    const section = 0x98 + 224;
    buf.writeUInt32LE(0x1000, section + 12);
    buf.writeUInt32LE(0x600, section + 16);
    buf.writeUInt32LE(0x200, section + 20);
    const fileOff = (rva) => 0x200 + (rva - 0x1000);
    buf.writeUInt32LE(0x1400, 0x98 + 96);
    buf.writeUInt32LE(0x1000, 0x98 + 96 + 8);
    buf.writeUInt32LE(0x1100, fileOff(0x1000) + 12);
    buf.writeUInt32LE(0x1200, fileOff(0x1000));
    buf.write('nccl.dll\0', fileOff(0x1100), 'ascii');
    buf.writeUInt32LE(0x1300, fileOff(0x1200));
    buf.write('ncclAllReduce\0', fileOff(0x1300) + 2, 'ascii');
    const exp = fileOff(0x1400);
    buf.writeUInt32LE(1, exp + 24);
    buf.writeUInt32LE(0x1500, exp + 32);
    buf.writeUInt32LE(0x1580, fileOff(0x1500));
    buf.write('ncclAllReduce\0', fileOff(0x1580), 'ascii');

    const pePath = path.join(os.tmpdir(), `glaux-pe-${process.pid}.dll`);
    fs.writeFileSync(pePath, buf);
    try {
      const pe = readPeImportsAndExports(pePath);
      assert.ok(pe);
      assert.deepEqual(pe.imports, ['nccl.dll']);
      assert.deepEqual(pe.importSymbols, ['ncclAllReduce']);
      assert.deepEqual(pe.exports, ['ncclAllReduce']);
    } finally {
      fs.rmSync(pePath, { force: true });
    }
  });

  it('replaces a needed library with a stub and deletes family payload', () => {
    if (process.platform !== 'linux') {
      return;
    }
    const gcc = which('gcc');
    if (!gcc) {
      return;
    }
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-cuda-stub-'));
    const site = path.join(root, 'lib', 'python3.14', 'site-packages');
    const ncclDir = path.join(site, 'nvidia', 'nccl', 'lib');
    const nvDir = path.join(site, 'nvidia', 'nvshmem', 'lib');
    const cuDir = path.join(site, 'nvidia', 'cu13', 'lib');
    const torchLib = path.join(site, 'torch', 'lib');
    try {
      fs.mkdirSync(ncclDir, { recursive: true });
      fs.mkdirSync(nvDir, { recursive: true });
      fs.mkdirSync(cuDir, { recursive: true });
      fs.mkdirSync(torchLib, { recursive: true });

      const ncclC = path.join(root, 'nccl.c');
      const nvC = path.join(root, 'nvshmem.c');
      const nvMap = path.join(root, 'nvshmem.map');
      const useC = path.join(root, 'use.c');
      fs.writeFileSync(ncclC, 'void ncclAllReduce(void) {}\n');
      fs.writeFileSync(nvC, 'void nvshmem_barrier(void) {}\n');
      fs.writeFileSync(nvMap, 'NVSHMEM {\n  global:\n    nvshmem_barrier;\n};\n');
      fs.writeFileSync(
        useC,
        'void ncclAllReduce(void);\nvoid nvshmem_barrier(void);\nvoid use(void) { ncclAllReduce(); nvshmem_barrier(); }\n'
      );

      const ncclSo = path.join(ncclDir, 'libnccl.so.2');
      const nvSo = path.join(nvDir, 'libnvshmem_host.so.3');
      const useSo = path.join(torchLib, 'libtorch_cuda.so');
      assert.equal(
        spawnSync(gcc, ['-shared', '-fPIC', '-Wl,-soname,libnccl.so.2', '-o', ncclSo, ncclC], { encoding: 'utf8' }).status,
        0
      );
      assert.equal(
        spawnSync(
          gcc,
          ['-shared', '-fPIC', '-Wl,-soname,libnvshmem_host.so.3', '-Wl,--version-script,' + nvMap, '-o', nvSo, nvC],
          { encoding: 'utf8' }
        ).status,
        0
      );
      const linked = spawnSync(
        gcc,
        [
          '-shared',
          '-fPIC',
          '-o',
          useSo,
          useC,
          `-Wl,-rpath,${ncclDir}`,
          `-Wl,-rpath,${nvDir}`,
          ncclSo,
          nvSo,
        ],
        { encoding: 'utf8' }
      );
      assert.equal(linked.status, 0, linked.stderr);
      fs.writeFileSync(path.join(nvDir, 'libnvshmem_device.bc'), Buffer.alloc(64, 1));
      fs.writeFileSync(path.join(cuDir, 'libcufft.so.12'), 'keep-cufft');

      const before = fs.statSync(ncclSo).size;
      const result = stubTorchUnusedCudaDeps(root);
      assert.ok(result.stubbed.includes('libnccl.so.2'));
      assert.ok(result.stubbed.includes('libnvshmem_host.so.3'));
      assert.ok(fs.statSync(ncclSo).size > 0);
      assert.ok(fs.statSync(ncclSo).size < before + 500000);
      assert.equal(fs.existsSync(path.join(nvDir, 'libnvshmem_device.bc')), false);
      assert.equal(fs.readFileSync(path.join(cuDir, 'libcufft.so.12'), 'utf8'), 'keep-cufft');

      const mainC = path.join(root, 'main.c');
      const mainBin = path.join(root, 'main');
      fs.writeFileSync(mainC, 'void use(void);\nint main(void) { use(); return 0; }\n');
      const mainBuild = spawnSync(
        gcc,
        ['-o', mainBin, mainC, useSo, `-Wl,-rpath,${torchLib}`, `-Wl,-rpath,${ncclDir}`, `-Wl,-rpath,${nvDir}`],
        { encoding: 'utf8' }
      );
      assert.equal(mainBuild.status, 0, mainBuild.stderr);
      const run = spawnSync(mainBin, [], { encoding: 'utf8' });
      assert.equal(run.status, 0, run.stderr);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});

describe('shareGgmlCudaBackend', () => {
  it('points transcribe at the llama.cpp CUDA module', () => {
    const name = ggmlCudaBackendFileName();
    if (!name) {
      return;
    }
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-ggml-share-'));
    try {
      const llama = path.join(root, 'llamacpp');
      const transcribe = path.join(root, 'transcribe');
      fs.mkdirSync(llama);
      fs.mkdirSync(transcribe);
      fs.writeFileSync(path.join(llama, name), 'llama-cuda');
      fs.writeFileSync(path.join(transcribe, name), 'transcribe-cuda');
      assert.equal(shareGgmlCudaBackend(llama, transcribe), true);
      assert.equal(fs.readFileSync(path.join(transcribe, name), 'utf8'), 'llama-cuda');
      if (process.platform !== 'win32') {
        assert.equal(fs.lstatSync(path.join(transcribe, name)).isSymbolicLink(), true);
        assert.equal(fs.readlinkSync(path.join(transcribe, name)), path.join('..', 'llamacpp', name));
      }
      assert.equal(shareGgmlCudaBackend(llama, transcribe), true);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
