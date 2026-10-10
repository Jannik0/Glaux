#!/usr/bin/env node
'use strict';

/**
 * Build sd-cli from deps/stable-diffusion.cpp into vendor/stablediffusion with
 * dynamic ggml backends (CPU + CUDA/Vulkan on Win/Linux, CPU + Metal on macOS).
 *
 * stable-diffusion.cpp master-945 (a1ded76). Its ggml submodule is 89c4413.
 * That commit keeps GGML_BACKEND_API_VERSION 2, the same loader header as
 * llama.cpp b11349 and transcribe.cpp v0.3.1, and adds the diffusion ops
 * those trees do not have. This build sets GGML_MAX_NAME=160. The ggml-cuda
 * module staged here is copied to vendor/cuda, and the copies next to each
 * engine are removed. llama.cpp and transcribe.cpp load that file from
 * ../cuda. CUDA runtime libraries (cudart/cublas) live in the same folder.
 * Vulkan stays next to this binary.
 *
 * One-shot CLI. One name-conversion patch maps a Hugging Face text encoder's
 * `embed_tokens` weight onto the tensor name sd-cli already expects.
 * WebP decoding uses the libwebp submodule pinned by this revision, linked
 * statically so the package does not gain another shared library. WebM stays off.
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
  stageSharedGgmlCudaBackend,
  patchGlauxCudaBackendSearch,
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
const SD_CPP_REV = 'a1ded76da5818803fca97a3b433669ef727d32cf';
const GGML_REPO = 'https://github.com/ggml-org/ggml';
const GGML_REV = '89c4413f5da6fb20cc796f16033d37f129be81fd';
/** Gitlink of thirdparty/libwebp at SD_CPP_REV. No system libwebp. */
const LIBWEBP_REPO = 'https://github.com/webmproject/libwebp.git';
const LIBWEBP_REV = '0c9546f7efc61eac7f79ae115c3f99c91c21c443';

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

/**
 * Some Hugging Face text encoders name the tied embedding `embed_tokens`.
 * The LLM name map only rewrote `word_embeddings`, so metadata validation
 * missed `text_encoders.llm.model.embed_tokens.weight`.
 * @param {string} srcDir
 */
function patchLlmEmbedTokens(srcDir) {
  const file = path.join(srcDir, 'src', 'name_conversion.cpp');
  const marker = '{"model.language_model.embed_tokens.", "model.embed_tokens."}';
  const needle = '{"model.language_model.word_embeddings.", "model.embed_tokens."}';
  let src = fs.readFileSync(file, 'utf8');
  if (src.includes(marker)) {
    return;
  }
  if (!src.includes(needle)) {
    throw new Error(
      'Could not locate the LLM embedding rename in name_conversion.cpp.'
    );
  }
  src = src.replace(
    needle,
    `${marker},\n        ${needle}`
  );
  fs.writeFileSync(file, src);
  console.log('Patched LLM embed_tokens name conversion.');
}

/**
 * The parent build sets BUILD_SHARED_LIBS=ON for ggml. libwebp would then be
 * a shared library that stageNativeRuntime does not copy (its name is not a
 * ggml/stable-diffusion module). Force a static libwebp so sd-cli links it in.
 * @param {string} srcDir
 */
function patchStaticLibwebp(srcDir) {
  const file = path.join(srcDir, 'thirdparty', 'CMakeLists.txt');
  const marker = 'GLAUX_STATIC_LIBWEBP';
  let src = fs.readFileSync(file, 'utf8');
  if (src.includes(marker)) {
    return;
  }
  const needle = 'add_subdirectory(libwebp EXCLUDE_FROM_ALL)';
  if (!src.includes(needle)) {
    throw new Error('Could not locate the libwebp subdirectory in thirdparty/CMakeLists.txt.');
  }
  const replacement = [
    `# ${marker}: link libwebp into sd-cli. A shared build would be an extra`,
    '# runtime library the packager does not stage next to the binary.',
    'set(_GLAUX_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")',
    'set(BUILD_SHARED_LIBS OFF)',
    'set(WEBP_LINK_STATIC ON CACHE BOOL "" FORCE)',
    needle,
    'set(BUILD_SHARED_LIBS "${_GLAUX_BUILD_SHARED_LIBS}")',
  ].join('\n');
  src = src.replace(needle, replacement);
  fs.writeFileSync(file, src);
  console.log('Patched libwebp to link statically into sd-cli.');
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
  ensureGitDep({
    dest: path.join(srcDir, 'thirdparty', 'libwebp'),
    url: LIBWEBP_REPO,
    rev: LIBWEBP_REV,
    name: 'libwebp',
  });
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printHelp();
    return;
  }

  ensureSources(opts.srcDir);
  patchLlmEmbedTokens(opts.srcDir);
  patchStaticLibwebp(opts.srcDir);
  patchGlauxCudaBackendSearch(path.join(opts.srcDir, 'ggml', 'src', 'ggml-backend-reg.cpp'));
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
    '-DSD_WEBP=ON',
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

  stageSharedGgmlCudaBackend({
    llamacpp: path.join(ROOT, 'vendor', 'llamacpp'),
    transcribe: path.join(ROOT, 'vendor', 'transcribe'),
    stablediffusion: opts.outDir,
  });

  console.log('stable-diffusion.cpp build complete.');
  console.log('ggml-cuda is staged in vendor/cuda (GGML_MAX_NAME=160).');
}

try {
  main();
} catch (err) {
  console.error(err.message || err);
  process.exit(1);
}
