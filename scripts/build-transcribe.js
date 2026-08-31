#!/usr/bin/env node
'use strict';

/**
 * Build transcribe-cli from deps/transcribe.cpp into vendor/transcribe with
 * dynamic ggml backends (CPU + CUDA/Vulkan on Win/Linux, CPU + Metal on macOS).
 * Clones handy-computer/transcribe.cpp at TRANSCRIBE_CPP_REV into deps/ if missing.
 *
 * Usage:
 *   node scripts/build-transcribe.js
 *   node scripts/build-transcribe.js --cpu-only
 *   node scripts/build-transcribe.js --out-dir=vendor/transcribe
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const {
  resolveBuildBackends,
  cmakeGpuArgs,
  rpathCmakeArgs,
  cudaQuietCmakeArgs,
  cmakeBuildQuietArgs,
  cudaBuildJobs,
  stageNativeRuntime,
  stageSharedCudaRuntime,
  removeStagedCudaRedistributables,
  which,
} = require('./gpuBackends');
const { ensureGitDep } = require('./ensureGitDep');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_SRC = path.join(ROOT, 'deps', 'transcribe.cpp');
const DEFAULT_OUT = path.join(ROOT, 'vendor', 'transcribe');
const TRANSCRIBE_CPP_REPO = 'https://github.com/handy-computer/transcribe.cpp';
const TRANSCRIBE_CPP_REV = '923d4a045ac8798ea4777f7eda557dc81963f4ed';

function printHelp() {
  console.log(`Usage: node scripts/build-transcribe.js [options]

Options:
  --src-dir=<path>     transcribe.cpp source tree (default: deps/transcribe.cpp)
  --out-dir=<path>     Stage directory (default: vendor/transcribe)
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
  const result = spawnSync(cmd, args, {
    stdio: 'inherit',
    shell: process.platform === 'win32',
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed (${result.status}): ${cmd} ${args.join(' ')}`);
  }
}

function findBuiltCli(buildDir) {
  const candidates =
    process.platform === 'win32'
      ? [
          path.join(buildDir, 'bin', 'Release', 'transcribe-cli.exe'),
          path.join(buildDir, 'bin', 'transcribe-cli.exe'),
          path.join(buildDir, 'Release', 'transcribe-cli.exe'),
          path.join(buildDir, 'examples', 'cli', 'Release', 'transcribe-cli.exe'),
          path.join(buildDir, 'transcribe-cli.exe'),
        ]
      : [
          path.join(buildDir, 'bin', 'transcribe-cli'),
          path.join(buildDir, 'examples', 'cli', 'transcribe-cli'),
          path.join(buildDir, 'transcribe-cli'),
        ];
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  return null;
}

/**
 * Glaux needs progressive stdout when Node pipes the CLI. On Windows, CRT full-
 * buffers stdout when redirected, so feed/partial lines only appear after exit
 * unless we unbuffer / fflush. Applied idempotently before each build.
 *
 * @param {string} srcDir
 */
function applyGlauxCliPatches(srcDir) {
  const cliMain = path.join(srcDir, 'examples', 'cli', 'main.cpp');
  if (!fs.existsSync(cliMain)) {
    throw new Error(`CLI source not found at ${cliMain}`);
  }
  let src = fs.readFileSync(cliMain, 'utf8');
  const nl = src.includes('\r\n') ? '\r\n' : '\n';
  let changed = false;

  const unbufMarker = 'GLAUX_STDOUT_UNBUFFERED';
  if (!src.includes(unbufMarker)) {
    const needle = `int main(int argc, char ** argv) {${nl}`;
    if (!src.includes(needle)) {
      throw new Error('Could not locate main() in transcribe-cli for Glaux stdout patch.');
    }
    src = src.replace(
      needle,
      `${needle}    // ${unbufMarker}: piped progressive stream lines must reach Glaux promptly.${nl}` +
        `    std::setvbuf(stdout, nullptr, _IONBF, 0);${nl}`
    );
    changed = true;
  }

  const feedFlushMarker = 'GLAUX_STREAM_FEED_FFLUSH';
  if (!src.includes(feedFlushMarker)) {
    const feedBlock =
      `                    std::printf("\\n");${nl}` + `                    ++feed_n;`;
    const feedReplacement =
      `                    std::printf("\\n");${nl}` +
      `                    std::fflush(stdout); // ${feedFlushMarker}${nl}` +
      `                    ++feed_n;`;
    if (!src.includes(feedBlock)) {
      throw new Error('Could not locate stream feed printf in transcribe-cli for Glaux fflush patch.');
    }
    // Only patch the single-file streaming path (first occurrence after stream_chunk_ms).
    src = src.replace(feedBlock, feedReplacement);
    changed = true;
  }

  // Prefer full_text for progressive partial= lines. committed_text freezes after
  // the first stable sentence for cache-aware models; Glaux uses snapshot/replace
  // UI updates so full_text rewrites are safe (no append loops).
  const fullPartialMarker = 'GLAUX_STREAM_FULL_PARTIAL';
  const committedPartialMarker = 'GLAUX_STREAM_COMMITTED_PARTIAL';
  if (!src.includes(fullPartialMarker)) {
    const committedBlock =
      `                    if (upd.result_changed) {${nl}` +
      `                        // ${committedPartialMarker}${nl}` +
      `                        struct transcribe_stream_text stxt;${nl}` +
      `                        transcribe_stream_text_init(&stxt);${nl}` +
      `                        const char * partial = nullptr;${nl}` +
      `                        if (transcribe_stream_get_text(ctx, &stxt) == TRANSCRIBE_OK &&${nl}` +
      `                            stxt.committed_text != nullptr && stxt.committed_text[0] != '\\0') {${nl}` +
      `                            partial = stxt.committed_text;${nl}` +
      `                        } else {${nl}` +
      `                            partial = transcribe_full_text(ctx);${nl}` +
      `                        }${nl}` +
      `                        std::printf("  partial=\\"%s\\"", (partial && *partial) ? partial : "");${nl}` +
      `                    }`;
    const upstreamBlock =
      `                    if (upd.result_changed) {${nl}` +
      `                        const char * partial = transcribe_full_text(ctx);${nl}` +
      `                        std::printf("  partial=\\"%s\\"", (partial && *partial) ? partial : "");${nl}` +
      `                    }`;
    const newPartial =
      `                    if (upd.result_changed) {${nl}` +
      `                        // ${fullPartialMarker}${nl}` +
      `                        const char * partial = transcribe_full_text(ctx);${nl}` +
      `                        std::printf("  partial=\\"%s\\"", (partial && *partial) ? partial : "");${nl}` +
      `                    }`;
    if (src.includes(committedBlock)) {
      src = src.replace(committedBlock, newPartial);
      changed = true;
    } else if (src.includes(upstreamBlock)) {
      src = src.replace(upstreamBlock, newPartial);
      changed = true;
    } else if (!src.includes(fullPartialMarker)) {
      throw new Error(
        'Could not locate stream partial printf in transcribe-cli for Glaux full_text patch.'
      );
    }
  }

  // Dynamic ggml backends (GGML_BACKEND_DL) are not compiled in. Upstream CLI
  // only calls transcribe_init_backends_default() for --list-devices, so a
  // normal transcription has zero devices and fails with
  // "failed to initialize CPU backend".
  const initBackendsMarker = 'GLAUX_INIT_BACKENDS';
  if (!src.includes(initBackendsMarker)) {
    const logSinkNeedle =
      `    // Install the log sink ONCE at startup, before any models or contexts${nl}` +
      `    // exist. This is the only supported usage model in 0.x; see the${nl}` +
      `    // threading contract in transcribe.h.${nl}` +
      `    if (!args.quiet) {${nl}` +
      `        transcribe_log_set(log_cb, nullptr);${nl}` +
      `    }${nl}`;
    const initBlock =
      `${nl}` +
      `    // ${initBackendsMarker}: load CPU/CUDA/Vulkan/Metal modules from next to libtranscribe.${nl}` +
      `    {${nl}` +
      `        const transcribe_status bst = transcribe_init_backends_default();${nl}` +
      `        if (bst != TRANSCRIBE_OK) {${nl}` +
      `            std::fprintf(stderr, "error: failed to load ggml backends: %s\\n",${nl}` +
      `                         transcribe_status_string(bst));${nl}` +
      `            return EXIT_FAILURE;${nl}` +
      `        }${nl}` +
      `    }${nl}`;
    if (!src.includes(logSinkNeedle)) {
      throw new Error(
        'Could not locate log-sink startup in transcribe-cli for Glaux backend-init patch.'
      );
    }
    src = src.replace(logSinkNeedle, `${logSinkNeedle}${initBlock}`);
    changed = true;
  }

  if (changed) {
    fs.writeFileSync(cliMain, src, 'utf8');
    console.log(`Applied Glaux CLI patches to ${cliMain}`);
  } else {
    console.log('Glaux CLI patches already applied.');
  }
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
      url: TRANSCRIBE_CPP_REPO,
      rev: TRANSCRIBE_CPP_REV,
      name: 'transcribe.cpp',
    });
  } else if (!fs.existsSync(opts.srcDir)) {
    throw new Error(`transcribe.cpp source not found at ${opts.srcDir}.`);
  }
  if (!which('cmake')) {
    throw new Error('cmake not found on PATH. Install CMake to build transcribe.cpp.');
  }

  applyGlauxCliPatches(opts.srcDir);

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
    '-DTRANSCRIBE_BUILD_SHARED=ON',
    '-DTRANSCRIBE_GGML_BACKEND_DL=ON',
    '-DTRANSCRIBE_BUILD_TESTS=OFF',
    '-DTRANSCRIBE_BUILD_EXAMPLES=ON',
    '-DTRANSCRIBE_BUILD_TOOLS=OFF',
    ...cmakeGpuArgs('transcribe', backends),
    ...rpathCmakeArgs(),
    ...cudaQuietCmakeArgs(backends),
  ];

  if (backends.metal) {
    cmakeArgs.push('-DGGML_METAL_EMBED_LIBRARY=ON');
  }

  if (process.platform !== 'win32') {
    cmakeArgs.push('-DCMAKE_BUILD_TYPE=Release');
  }

  run('cmake', cmakeArgs);

  const buildArgs = [
    '--build',
    buildDir,
    '--config',
    'Release',
    '--target',
    'transcribe-cli',
    '-j',
    String(cudaBuildJobs(opts.jobs, backends)),
    ...cmakeBuildQuietArgs(),
  ];
  run('cmake', buildArgs);

  const built = findBuiltCli(buildDir);
  if (!built) {
    throw new Error(`transcribe-cli binary not found under ${buildDir}`);
  }
  const destName = process.platform === 'win32' ? 'transcribe-cli.exe' : 'transcribe-cli';
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

  console.log('transcribe.cpp build complete.');
}

try {
  main();
} catch (err) {
  console.error(err.message || err);
  process.exit(1);
}
