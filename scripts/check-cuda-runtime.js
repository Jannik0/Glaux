#!/usr/bin/env node
'use strict';

/**
 * Fail packaging on Windows/Linux if the shared CUDA 13 runtime is missing.
 * macOS ships no CUDA.
 */

const path = require('path');
const { missingSharedCudaRedists, sharedCudaDir } = require('./gpuBackends');

if (process.platform === 'darwin') {
  console.log('Skipping shared CUDA runtime check on macOS');
  process.exit(0);
}

const dir = sharedCudaDir();
const missing = missingSharedCudaRedists(dir);
if (missing.length) {
  console.error(
    `Missing shared CUDA 13 runtime under ${dir} (${missing.join(', ')}).\n` +
      `Rebuild a GPU vendor tree:\n  npm run build:python\n  npm run build:llamacpp\n  npm run build:transcribe`
  );
  process.exit(1);
}

console.log(`Shared CUDA 13 runtime OK: ${path.relative(path.join(__dirname, '..'), dir)}`);
