#!/usr/bin/env node
'use strict';

/**
 * Build llama-server from deps/llama.cpp into vendor/llamacpp with dynamic
 * ggml backends (CPU + CUDA/Vulkan on Win/Linux, CPU + Metal on macOS).
 * Clones ggml-org/llama.cpp at LLAMA_CPP_REV into deps/ if missing.
 * Video multimodal support uses the shared vendor/ffmpeg tree
 * (`npm run build:ffmpeg`); this script does not stage ffmpeg/ffprobe.
 * CUDA kernels use a pinned architecture list (Turing–Blackwell) from
 * gpuBackends.cudaArchitectureCmakeArgs rather than the toolkit default.
 *
 * Usage:
 *   node scripts/build-llamacpp.js
 *   node scripts/build-llamacpp.js --cpu-only
 *   node scripts/build-llamacpp.js --out-dir=vendor/llamacpp
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
  shareGgmlCudaBackend,
  removeStagedCudaRedistributables,
  withCudaToolkitEnv,
  which,
  requirePatchelf,
} = require('./gpuBackends');
const { ensureGitDep } = require('./ensureGitDep');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_SRC = path.join(ROOT, 'deps', 'llama.cpp');
const DEFAULT_OUT = path.join(ROOT, 'vendor', 'llamacpp');
const LLAMA_CPP_REPO = 'https://github.com/ggml-org/llama.cpp';
const LLAMA_CPP_REV = '0882c7bc89074017c6a2a3149eae46929cb32ab3';

function printHelp() {
  console.log(`Usage: node scripts/build-llamacpp.js [options]

Options:
  --src-dir=<path>     llama.cpp source tree (default: deps/llama.cpp)
  --out-dir=<path>     Stage directory (default: vendor/llamacpp)
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
  // Windows joins into one string: Node warns (DEP0190) when shell:true is
  // paired with an args array, and that array is only concatenated anyway.
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

function findBuiltServer(buildDir) {
  const candidates =
    process.platform === 'win32'
      ? [
          path.join(buildDir, 'bin', 'Release', 'llama-server.exe'),
          path.join(buildDir, 'bin', 'llama-server.exe'),
          path.join(buildDir, 'Release', 'llama-server.exe'),
          path.join(buildDir, 'llama-server.exe'),
        ]
      : [
          path.join(buildDir, 'bin', 'llama-server'),
          path.join(buildDir, 'llama-server'),
        ];
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  return null;
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printHelp();
    return;
  }

  if (opts.srcDir === DEFAULT_SRC) {
    ensureGitDep({
      dest: DEFAULT_SRC,
      url: LLAMA_CPP_REPO,
      rev: LLAMA_CPP_REV,
      name: 'llama.cpp',
    });
  } else if (!fs.existsSync(opts.srcDir)) {
    throw new Error(`llama.cpp source not found at ${opts.srcDir}.`);
  }
  if (!which('cmake')) {
    throw new Error('cmake not found on PATH. Install CMake to build llama.cpp.');
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
    '-DBUILD_SHARED_LIBS=ON',
    '-DGGML_BACKEND_DL=ON',
    '-DGGML_NATIVE=OFF',
    '-DGGML_HIP=OFF',
    '-DGGML_SYCL=OFF',
    '-DLLAMA_BUILD_SERVER=ON',
    ...cmakeGpuArgs('ggml', backends),
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
    'llama-server',
    '-j',
    String(cudaBuildJobs(opts.jobs, backends)),
    ...cmakeBuildQuietArgs(),
  ];
  run('cmake', buildArgs, { env: cmakeEnv });

  const built = findBuiltServer(buildDir);
  if (!built) {
    throw new Error(`llama-server binary not found under ${buildDir}`);
  }
  const destName = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server';
  stageNativeRuntime({
    builtPath: built,
    buildDir,
    outDir: opts.outDir,
    destName,
  });
  removeStagedCudaRedistributables(opts.outDir);
  if (backends.cuda) {
    stageSharedCudaRuntime({ required: true });
  }
  shareGgmlCudaBackend(opts.outDir, path.join(ROOT, 'vendor', 'transcribe'));

  console.log('llama.cpp build complete.');
}

try {
  main();
} catch (err) {
  console.error(err.message || err);
  process.exit(1);
}
