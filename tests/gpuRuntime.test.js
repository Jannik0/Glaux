'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { isForceCpu, withForceCpuTorchEnv, withVendorLibPath, withSharedCudaLibPath } = require('../engines/common/gpuRuntime');
const { expectedGpuBackends, findBackendModule, isCuda13RedistName, missingSharedCudaRedists } = require('../scripts/gpuBackends');

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
