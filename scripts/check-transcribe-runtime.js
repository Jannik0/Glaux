#!/usr/bin/env node
'use strict';

/**
 * Fail packaging if vendor/transcribe is missing transcribe-cli or expected
 * ggml backend modules for this OS.
 */

const fs = require('fs');
const path = require('path');
const { expectedGpuBackends, localGgmlCudaModule, missingBackendModules, sharedCudaDir } = require('./gpuBackends');

const root = path.resolve(__dirname, '..');
const vendor = path.join(root, 'vendor', 'transcribe');
const name = process.platform === 'win32' ? 'transcribe-cli.exe' : 'transcribe-cli';
const binary = path.join(vendor, name);

if (!fs.existsSync(binary)) {
  console.error(
    `Missing ${binary}.\nRun: npm run build:transcribe\n(Requires Git, CMake, and CUDA Toolkit + Vulkan SDK on Windows/Linux.)`
  );
  process.exit(1);
}

console.log(`Found transcribe-cli at ${binary}`);

const localCuda = localGgmlCudaModule(vendor);
if (localCuda) {
  console.error(
    `${localCuda} is next to transcribe-cli. The shared ggml CUDA module belongs in vendor/cuda.\n` +
      'Rebuild:\n  npm run build:transcribe'
  );
  process.exit(1);
}

const required = ['cpu', ...expectedGpuBackends()];
const missing = missingBackendModules(vendor, required, { cudaDir: sharedCudaDir() });
if (missing.length) {
  console.error(
    `Missing ggml backend module(s) for transcribe.cpp: ${missing.map((b) => `ggml-${b}`).join(', ')}.\n` +
      `CUDA is loaded from vendor/cuda; the other backends stay in ${vendor}.\n` +
      `Rebuild with GPU backends enabled (do not pass --cpu-only):\n  npm run build:transcribe`
  );
  process.exit(1);
}
console.log(`Found transcribe.cpp backends: ${required.join(', ')}`);
