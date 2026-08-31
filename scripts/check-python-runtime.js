#!/usr/bin/env node
'use strict';

/** Fail the packaging step if vendor/python is missing a usable interpreter.
 *  Windows/Linux must ship a CUDA 13 Torch build (CPU fallback still works at runtime).
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { withSharedCudaLibPath } = require('../engines/common/gpuRuntime');

const root = path.join(__dirname, '..', 'vendor', 'python');
const candidates =
  process.platform === 'win32'
    ? [path.join(root, 'python.exe')]
    : [path.join(root, 'bin', 'python3'), path.join(root, 'bin', 'python'), path.join(root, 'python3')];

const found = candidates.find((candidate) => fs.existsSync(candidate));
if (!found) {
  console.error(
    `Missing bundled Python runtime under vendor/python (looked for: ${candidates
      .map((p) => path.relative(path.join(__dirname, '..'), p))
      .join(', ')}).`
  );
  console.error('Run: npm run build:python');
  process.exit(1);
}

console.log(`Bundled Python OK: ${path.relative(path.join(__dirname, '..'), found)}`);

if (process.platform !== 'darwin') {
  const result = spawnSync(
    found,
    ['-c', 'import torch; print(torch.version.cuda or "")'],
    { encoding: 'utf8', env: withSharedCudaLibPath({ ...process.env }) }
  );
  if (result.status !== 0) {
    console.error(result.stderr || result.stdout || 'Failed to import torch from bundled Python.');
    process.exit(1);
  }
  const cuda = String(result.stdout || '').trim();
  if (!cuda) {
    console.error(
      'Bundled Torch is CPU-only. Rebuild with CUDA wheels (default):\n  npm run build:python'
    );
    process.exit(1);
  }
  if (!cuda.startsWith('13')) {
    console.error(
      `Bundled Torch CUDA is ${cuda}, expected 13.x so it can share vendor/cuda with llama.cpp and transcribe.cpp.\n` +
        `Rebuild:\n  npm run build:python`
    );
    process.exit(1);
  }
  console.log(`Bundled Torch CUDA: ${cuda}`);
}
