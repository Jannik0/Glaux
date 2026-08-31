'use strict';

/**
 * Runtime GPU policy shared by llama.cpp, transcribe.cpp, and Hugging Face bridges.
 * GLAUX_FORCE_CPU=1|true|yes disables GPU even when backends are shipped.
 */

const fs = require('fs');
const path = require('path');
const { getVendorRoot } = require('./runtimePaths');

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

module.exports = {
  isForceCpu,
  withForceCpuTorchEnv,
  withVendorLibPath,
  withSharedCudaLibPath,
};
