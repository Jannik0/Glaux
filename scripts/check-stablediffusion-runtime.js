#!/usr/bin/env node
'use strict';

/**
 * Fail packaging if vendor/stablediffusion is missing sd-cli or expected
 * ggml backend modules for this OS.
 */

const fs = require('fs');
const path = require('path');
const { expectedGpuBackends, localGgmlCudaModule, missingBackendModules, sharedCudaDir } = require('./gpuBackends');

const root = path.resolve(__dirname, '..');
const vendor = path.join(root, 'vendor', 'stablediffusion');
const name = process.platform === 'win32' ? 'sd-cli.exe' : 'sd-cli';

/** License texts staged next to sd-cli. Packaging refuses to ship without them. */
const STAGED_LICENSE_FILES = [
  'LIBWEBP.COPYING',
  'LIBWEBP.PATENTS',
  'ONIGURUMA.COPYING',
  'DARTS.LICENSE',
  'UTF8PROC.LICENSE.md',
];

/**
 * @param {string} vendorDir
 * @returns {string[]}
 */
function missingStagedLicenses(vendorDir) {
  return STAGED_LICENSE_FILES.filter((fileName) => !fs.existsSync(path.join(vendorDir, fileName)));
}

/**
 * @param {{ vendor?: string, cudaDir?: string }} [opts]
 * @returns {string[]} human-readable problems; empty when the tree can be packaged
 */
function checkStableDiffusionRuntime(opts = {}) {
  const vendorDir = opts.vendor || vendor;
  const binary = path.join(vendorDir, name);
  const problems = [];
  if (!fs.existsSync(binary)) {
    problems.push(
      `Missing ${binary}.\nRun: npm run build:stablediffusion\n(Requires Git, CMake, and CUDA Toolkit + Vulkan SDK on Windows/Linux.)`
    );
  }
  const localCuda = localGgmlCudaModule(vendorDir);
  if (localCuda) {
    problems.push(
      `${localCuda} is next to sd-cli. The shared ggml CUDA module belongs in vendor/cuda.\n` +
        'Rebuild:\n  npm run build:stablediffusion'
    );
  }
  const required = ['cpu', ...expectedGpuBackends()];
  const missing = missingBackendModules(vendorDir, required, {
    cudaDir: opts.cudaDir || sharedCudaDir(),
  });
  if (missing.length) {
    problems.push(
      `Missing ggml backend module(s) for stable-diffusion.cpp: ${missing.map((b) => `ggml-${b}`).join(', ')}.\n` +
        `CUDA is loaded from vendor/cuda; the other backends stay in ${vendorDir}.\n` +
        `Rebuild with GPU backends enabled (do not pass --cpu-only):\n  npm run build:stablediffusion`
    );
  }
  const missingLicenses = missingStagedLicenses(vendorDir);
  if (missingLicenses.length) {
    problems.push(
      `Missing third-party license notice(s) in ${vendorDir}: ${missingLicenses.join(', ')}.\n` +
        'These files must sit next to sd-cli.\n' +
        'Rebuild:\n  npm run build:stablediffusion'
    );
  }
  return problems;
}

if (require.main === module) {
  const binary = path.join(vendor, name);
  const problems = checkStableDiffusionRuntime();
  if (problems.length) {
    for (const problem of problems) {
      console.error(problem);
    }
    process.exit(1);
  }
  console.log(`Found sd-cli at ${binary}`);
  console.log(`Found stable-diffusion.cpp backends: ${['cpu', ...expectedGpuBackends()].join(', ')}`);
  console.log(`Found license notices: ${STAGED_LICENSE_FILES.join(', ')}`);
}

module.exports = {
  checkStableDiffusionRuntime,
  STAGED_LICENSE_FILES,
};
