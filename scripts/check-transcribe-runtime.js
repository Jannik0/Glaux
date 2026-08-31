#!/usr/bin/env node
'use strict';

/**
 * Fail packaging if vendor/transcribe is missing transcribe-cli or expected
 * ggml backend modules for this OS.
 */

const fs = require('fs');
const path = require('path');
const { expectedGpuBackends, missingBackendModules } = require('./gpuBackends');

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

const required = ['cpu', ...expectedGpuBackends()];
const missing = missingBackendModules(vendor, required);
if (missing.length) {
  console.error(
    `Missing ggml backend module(s) in ${vendor}: ${missing.map((b) => `ggml-${b}`).join(', ')}.\n` +
      `Rebuild with GPU backends enabled (do not pass --cpu-only):\n  npm run build:transcribe`
  );
  process.exit(1);
}
console.log(`Found transcribe.cpp backends: ${required.join(', ')}`);
