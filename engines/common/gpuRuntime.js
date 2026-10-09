'use strict';

/**
 * Runtime GPU policy shared by llama.cpp, transcribe.cpp, stable-diffusion.cpp, and Hugging Face bridges.
 * GLAUX_FORCE_CPU=1|true|yes disables GPU even when backends are shipped.
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { getVendorRoot } = require('./runtimePaths');
const {
  cudaComputeCapabilitySupported,
  isNoCudaKernelImage,
  selectCudaVisibleDevices,
} = require('./cudaArch');

/**
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {boolean}
 */
function isForceCpu(env = process.env) {
  const value = String(env.GLAUX_FORCE_CPU || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes';
}

/**
 * Hide CUDA/HIP from a child process when forcing CPU. Must be set before that
 * process imports PyTorch; `device_map="cpu"` still initializes the CUDA driver
 * on Windows CUDA wheels and can native-crash (exit 3221225477 / 0xC0000005).
 *
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {NodeJS.ProcessEnv}
 */
function withForceCpuTorchEnv(env = process.env) {
  const next = { ...env };
  if (!isForceCpu(next)) {
    return next;
  }
  next.CUDA_VISIBLE_DEVICES = '';
  next.HIP_VISIBLE_DEVICES = '';
  return next;
}

/**
 * Prepend a vendor binary directory so sibling ggml backend modules resolve
 * on Windows (PATH), Linux (LD_LIBRARY_PATH), and macOS (DYLD_LIBRARY_PATH).
 *
 * @param {NodeJS.ProcessEnv} baseEnv
 * @param {string} vendorDir
 * @returns {NodeJS.ProcessEnv}
 */
function withVendorLibPath(baseEnv, vendorDir) {
  const env = { ...baseEnv };
  const resolved = path.resolve(vendorDir);
  const delim = path.delimiter;
  env.PATH = `${resolved}${delim}${env.PATH || ''}`;
  if (process.platform === 'linux') {
    env.LD_LIBRARY_PATH = `${resolved}${delim}${env.LD_LIBRARY_PATH || ''}`;
  } else if (process.platform === 'darwin') {
    env.DYLD_LIBRARY_PATH = `${resolved}${delim}${env.DYLD_LIBRARY_PATH || ''}`;
  }
  return env;
}

/**
 * Prepend the shared CUDA 13 redistributable directory (vendor/cuda or
 * resources/cuda) so Torch, llama-server, and transcribe-cli load one runtime.
 * No-ops when the directory is missing (macOS, CPU-only, incomplete vendor tree).
 *
 * @param {NodeJS.ProcessEnv} baseEnv
 * @param {string} [cudaDir]
 * @returns {NodeJS.ProcessEnv}
 */
function withSharedCudaLibPath(baseEnv, cudaDir) {
  const resolved = path.resolve(cudaDir == null ? getVendorRoot('cuda') : cudaDir);
  if (!fs.existsSync(resolved)) {
    return { ...baseEnv };
  }
  const env = withVendorLibPath(baseEnv, resolved);
  env.GLAUX_CUDA_DIR = resolved;
  return env;
}

/** @type {{ index: number, major: number, minor: number }[] | null | undefined} */
let nvidiaProbeCache;

/**
 * Physical NVIDIA GPUs from nvidia-smi. Cached. Null when the tool is missing
 * or reports nothing (macOS, CPU-only, driver not installed).
 * @returns {{ index: number, major: number, minor: number }[] | null}
 */
function queryNvidiaComputeCaps() {
  if (nvidiaProbeCache !== undefined) {
    return nvidiaProbeCache;
  }
  nvidiaProbeCache = null;
  if (process.platform === 'darwin') {
    return null;
  }
  const result = spawnSync(
    'nvidia-smi',
    ['--query-gpu=index,compute_cap', '--format=csv,noheader'],
    { encoding: 'utf8', timeout: 8000, windowsHide: true }
  );
  if (result.error || result.status !== 0 || !result.stdout) {
    return null;
  }
  /** @type {{ index: number, major: number, minor: number }[]} */
  const gpus = [];
  for (const line of result.stdout.split(/\r?\n/)) {
    const match = line.trim().match(/^(\d+)\s*,\s*(\d+)\.(\d+)\s*$/);
    if (!match) {
      continue;
    }
    gpus.push({
      index: Number(match[1]),
      major: Number(match[2]),
      minor: Number(match[3]),
    });
  }
  nvidiaProbeCache = gpus.length ? gpus : null;
  return nvidiaProbeCache;
}

/**
 * Hide NVIDIA GPUs that have no bundled cubin and are older than the 12.0 PTX
 * image. Vulkan still sees them. Leaves the env alone when nvidia-smi is
 * missing, the user already hid CUDA, or every visible GPU is supported.
 *
 * @param {NodeJS.ProcessEnv} [env]
 * @param {() => { index: number, major: number, minor: number }[] | null} [query]
 * @returns {NodeJS.ProcessEnv}
 */
function withUnsupportedCudaHidden(env = process.env, query = queryNvidiaComputeCaps) {
  const next = { ...env };
  if (isForceCpu(next)) {
    return next;
  }
  let gpus = null;
  try {
    gpus = query();
  } catch {
    return next;
  }
  if (!gpus || !gpus.length) {
    return next;
  }
  const visible = Object.prototype.hasOwnProperty.call(env, 'CUDA_VISIBLE_DEVICES')
    ? env.CUDA_VISIBLE_DEVICES
    : undefined;
  const selected = selectCudaVisibleDevices(gpus, visible);
  if (selected == null || selected === next.CUDA_VISIBLE_DEVICES) {
    return next;
  }
  if (selected === '-1') {
    process.stderr.write(
      '[glaux] This NVIDIA GPU is outside the bundled CUDA architectures; using Vulkan.\n'
    );
  } else {
    process.stderr.write(
      `[glaux] Hiding NVIDIA GPUs without a bundled CUDA image (CUDA_VISIBLE_DEVICES=${selected}).\n`
    );
  }
  next.CUDA_VISIBLE_DEVICES = selected;
  return next;
}

/**
 * @param {NodeJS.ProcessEnv} env
 * @returns {NodeJS.ProcessEnv}
 */
function withCudaHidden(env) {
  return { ...env, CUDA_VISIBLE_DEVICES: '-1' };
}

module.exports = {
  isForceCpu,
  withForceCpuTorchEnv,
  withVendorLibPath,
  withSharedCudaLibPath,
  withUnsupportedCudaHidden,
  withCudaHidden,
  queryNvidiaComputeCaps,
  cudaComputeCapabilitySupported,
  isNoCudaKernelImage,
  selectCudaVisibleDevices,
};
