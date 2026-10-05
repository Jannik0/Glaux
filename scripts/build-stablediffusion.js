#!/usr/bin/env node
'use strict';

/**
 * Build sd-cli from deps/stable-diffusion.cpp into vendor/stablediffusion with
 * dynamic ggml backends (CPU + CUDA/Vulkan on Win/Linux, CPU + Metal on macOS).
 *
 * stable-diffusion.cpp 47e83d71 ("sync: update ggml"). Its ggml submodule is
 * 4e86b56f, eight commits ahead of and zero behind ggml 353b63b — the snapshot
 * vendored by llama.cpp b11256 (c85b92c) and transcribe.cpp 4807edaf. Same
 * lineage, not the identical commit. This tree defines GGML_MAX_NAME=160
 * (diffusion tensor names exceed ggml's default of 64), so libggml-cuda is not
 * ABI-compatible with llama/transcribe and is not shared. CUDA runtime
 * libraries (cudart/cublas) still go to vendor/cuda.
 *
 * One-shot CLI. No Glaux source patches: sd-cli already loads dynamic backends.
 *
 * Usage:
 *   node scripts/build-stablediffusion.js
 *   node scripts/build-stablediffusion.js --cpu-only
 *   node scripts/build-stablediffusion.js --out-dir=vendor/stablediffusion
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const {
  resolveBuildBackends,
  cmakeGpuArgs,
  cudaCompilerCmakeArgs,
  cudaArchitectureCmakeArgs,
  rpathCmakeArgs,
  cudaQuietCmakeArgs,
  cmakeBuildQuietArgs,
  cudaBuildJobs,
  stageNativeRuntime,
  stageSharedCudaRuntime,
  removeStagedCudaRedistributables,
  withCudaToolkitEnv,
  which,
  requirePatchelf,
} = require('./gpuBackends');
const { ensureGitDep } = require('./ensureGitDep');
const { pruneCudaFatbinsInTree } = require('./cudaFatbin');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_SRC = path.join(ROOT, 'deps', 'stable-diffusion.cpp');
const DEFAULT_OUT = path.join(ROOT, 'vendor', 'stablediffusion');
const SD_CPP_REPO = 'https://github.com/leejet/stable-diffusion.cpp';
const SD_CPP_REV = '47e83d713618cffcae4449d4fcc712e9005b8549';
const GGML_REPO = 'https://github.com/ggml-org/ggml';
const GGML_REV = '4e86b56f1658203d7f403a3ab8b5da852d7f6cd0';

function printHelp() {
  console.log(`Usage: node scripts/build-stablediffusion.js [options]

Options:
  --src-dir=<path>     stable-diffusion.cpp source tree (default: deps/stable-diffusion.cpp)
  --out-dir=<path>     Stage directory (default: vendor/stablediffusion)
  --jobs=<n>           Parallel build jobs (default: CPU count)
  --cpu-only           Skip CUDA/Vulkan/Metal (local iteration; cannot package)
  -h, --help           Show this help
`);
}

function parseArgs(argv) {
  const opts = {
    srcDir: DEFAULT_SRC,
    outDir: DEFAULT_OUT,
    jobs: Math.max(1, os.cpus().length || 4),
    cpuOnly: false,
  };
  for (const arg of argv) {
    if (arg === '-h' || arg === '--help') {
      opts.help = true;
    } else if (arg === '--cpu-only') {
      opts.cpuOnly = true;
    } else if (arg.startsWith('--src-dir=')) {
      opts.srcDir = path.resolve(arg.slice('--src-dir='.length));
    } else if (arg.startsWith('--out-dir=')) {
      opts.outDir = path.resolve(arg.slice('--out-dir='.length));
    } else if (arg.startsWith('--jobs=')) {
      opts.jobs = Math.max(1, Number(arg.slice('--jobs='.length)) || opts.jobs);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return opts;
}

function run(cmd, args, options = {}) {
  console.log(`> ${cmd} ${args.join(' ')}`);
  const shell = process.platform === 'win32';
  const result = spawnSync(shell ? [cmd, ...args].join(' ') : cmd, shell ? [] : args, {
    stdio: 'inherit',
    shell,
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed (${result.status}): ${cmd} ${args.join(' ')}`);
  }
}

function findBuiltCli(buildDir) {
  const name = process.platform === 'win32' ? 'sd-cli.exe' : 'sd-cli';
  const candidates =
    process.platform === 'win32'
      ? [
          path.join(buildDir, 'bin', 'Release', name),
          path.join(buildDir, 'bin', name),
          path.join(buildDir, 'Release', name),
          path.join(buildDir, 'examples', 'cli', 'Release', name),
          path.join(buildDir, name),
        ]
      : [
          path.join(buildDir, 'bin', name),
          path.join(buildDir, 'examples', 'cli', name),
          path.join(buildDir, name),
        ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

function ensureSources(srcDir) {
  if (srcDir === DEFAULT_SRC) {
    ensureGitDep({
      dest: DEFAULT_SRC,
      url: SD_CPP_REPO,
      rev: SD_CPP_REV,
      name: 'stable-diffusion.cpp',
    });
  } else if (!fs.existsSync(srcDir)) {
    throw new Error(`stable-diffusion.cpp source not found at ${srcDir}.`);
  }
  ensureGitDep({
    dest: path.join(srcDir, 'ggml'),
    url: GGML_REPO,
    rev: GGML_REV,
    name: 'ggml (stable-diffusion.cpp)',
  });
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printHelp();
    return;
  }

  ensureSources(opts.srcDir);
  if (!which('cmake')) {
    throw new Error('cmake not found on PATH. Install CMake to build stable-diffusion.cpp.');
  }
  requirePatchelf();

  const buildDir = path.join(opts.srcDir, 'build-glaux');
  fs.mkdirSync(opts.outDir, { recursive: true });
  fs.mkdirSync(buildDir, { recursive: true });

  const backends = resolveBuildBackends({ cpuOnly: opts.cpuOnly });
  console.log(
    `GPU backends: ${backends.cpuOnly ? 'cpu-only' : ['cpu', backends.cuda && 'cuda', backends.vulkan && 'vulkan', backends.metal && 'metal'].filter(Boolean).join(', ')}`
  );

  const cmakeArgs = [
    '-S',
    opts.srcDir,
    '-B',
    buildDir,
    '-DSD_BUILD_EXAMPLES=ON',
    '-DSD_BUILD_SHARED_LIBS=ON',
    '-DSD_BUILD_SHARED_GGML_LIB=ON',
    '-DBUILD_SHARED_LIBS=ON',
    '-DGGML_BACKEND_DL=ON',
    '-DGGML_NATIVE=OFF',
    '-DSD_WEBP=OFF',
    '-DSD_WEBM=OFF',
    '-DSD_HIPBLAS=OFF',
    '-DSD_OPENCL=OFF',
    '-DSD_SYCL=OFF',
    '-DSD_USE_UPSTREAM_GGML=OFF',
    ...cmakeGpuArgs('sd', backends),
    ...cudaCompilerCmakeArgs(backends),
    ...cudaArchitectureCmakeArgs(backends),
    ...rpathCmakeArgs(),
    ...cudaQuietCmakeArgs(backends),
  ];

  if (backends.metal) {
    cmakeArgs.push('-DGGML_METAL_EMBED_LIBRARY=ON');
  }

  if (process.platform !== 'win32') {
    cmakeArgs.push('-DCMAKE_BUILD_TYPE=Release');
  }

  const cmakeEnv = withCudaToolkitEnv();
  run('cmake', cmakeArgs, { env: cmakeEnv });

  const buildArgs = [
    '--build',
    buildDir,
    '--config',
    'Release',
    '--target',
    'sd-cli',
    '-j',
    String(cudaBuildJobs(opts.jobs, backends)),
    ...cmakeBuildQuietArgs(),
  ];
  run('cmake', buildArgs, { env: cmakeEnv });

  const built = findBuiltCli(buildDir);
  if (!built) {
    throw new Error(`sd-cli binary not found under ${buildDir}`);
  }
  const destName = process.platform === 'win32' ? 'sd-cli.exe' : 'sd-cli';
  stageNativeRuntime({
    builtPath: built,
    buildDir,
    outDir: opts.outDir,
    destName,
  });
  removeStagedCudaRedistributables(opts.outDir);
  if (backends.cuda) {
    stageSharedCudaRuntime({ required: true });
    pruneCudaFatbinsInTree(opts.outDir);
  }

  console.log('stable-diffusion.cpp build complete.');
  console.log(
    'ggml-cuda is kept next to sd-cli (GGML_MAX_NAME=160). It is not linked to the llama.cpp CUDA backend.'
  );
}

try {
  main();
} catch (err) {
  console.error(err.message || err);
  process.exit(1);
}
