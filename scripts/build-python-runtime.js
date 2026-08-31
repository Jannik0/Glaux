#!/usr/bin/env node
'use strict';

/**
 * Builds a relocatable Python runtime into vendor/python for electron-builder.
 * Uses astral-sh/python-build-standalone (install_only_stripped) on Windows, macOS, and Linux.
 *
 * Usage:
 *   node scripts/build-python-runtime.js
 *   node scripts/build-python-runtime.js --torch-variant=cpu
 *   node scripts/build-python-runtime.js --torch-variant=cu130
 *
 * Default torch variant is cu130 on Windows/Linux (CUDA wheels still run on CPU
 * when no NVIDIA GPU is present). Overlapping CUDA 13 redistributables are
 * moved into vendor/cuda so Torch shares one runtime with llama.cpp / transcribe.cpp.
 * macOS always installs the default PyPI wheels (MPS-capable when hardware supports it).
 *
 * After wheels are installed, packaging leftovers are removed (MSVC .lib files,
 * torch/include, pip/setuptools/ensurepip, tcl/tk, hf_xet, package tests/
 * trees, Torch cusolverMg / nvperf_host / nvrtc*.alt). Public testing modules
 * (numpy.testing, torch.testing) stay. CUPTI and the non-alt NVRTC stay. Rebuild
 * the runtime to add Python packages; the shipped tree cannot pip-install.
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const https = require('https');
const http = require('http');
const path = require('path');
const { pipeline } = require('stream/promises');
const { withSharedCudaLibPath } = require('../engines/common/gpuRuntime');
const {
  stageSharedCudaRuntime,
  shareTorchCuda13WithVendor,
  findTorchLibDir,
} = require('./gpuBackends');

const DEFAULT_PBS_TAG = '20260718';
const DEFAULT_PYTHON_VERSION = '3.14.6';
const TORCH_VERSION = '2.13.0';
const TORCHVISION_VERSION = '0.28.0';
const VALID_TORCH_VARIANTS = new Set(['cpu', 'cu126', 'cu128', 'cu130']);

function printHelp() {
  console.log(`Usage: node scripts/build-python-runtime.js [options]

Options:
  --torch-variant=<cpu|cu126|cu128|cu130>  PyTorch build (default: cu130 on Win/Linux, cpu on macOS)
  --python-version=<x.y.z>                 CPython version (default: ${DEFAULT_PYTHON_VERSION})
  --pbs-tag=<YYYYMMDD>                     python-build-standalone release tag (default: ${DEFAULT_PBS_TAG})
  --out-dir=<path>                         Output directory (default: vendor/python)
  -h, --help                               Show this help
`);
}

function defaultTorchVariant() {
  return process.platform === 'darwin' ? 'cpu' : 'cu130';
}

function parseArgs(argv) {
  const args = {
    torchVariant: defaultTorchVariant(),
    pythonVersion: DEFAULT_PYTHON_VERSION,
    pbsTag: DEFAULT_PBS_TAG,
    outDir: '',
  };

  for (const raw of argv) {
    if (raw === '-h' || raw === '--help') {
      printHelp();
      process.exit(0);
    }
    if (raw.startsWith('--torch-variant=')) {
      args.torchVariant = raw.slice('--torch-variant='.length);
      continue;
    }
    if (raw.startsWith('--python-version=')) {
      args.pythonVersion = raw.slice('--python-version='.length);
      continue;
    }
    if (raw.startsWith('--pbs-tag=')) {
      args.pbsTag = raw.slice('--pbs-tag='.length);
      continue;
    }
    if (raw.startsWith('--out-dir=')) {
      args.outDir = raw.slice('--out-dir='.length);
      continue;
    }
    throw new Error(`Unknown argument: ${raw}`);
  }

  if (!VALID_TORCH_VARIANTS.has(args.torchVariant)) {
    throw new Error(
      `Invalid --torch-variant=${args.torchVariant}. Expected one of: ${[...VALID_TORCH_VARIANTS].join(', ')}`
    );
  }

  return args;
}

function pbsTriple() {
  const arch =
    process.arch === 'x64' ? 'x86_64' : process.arch === 'arm64' ? 'aarch64' : null;
  if (!arch) {
    throw new Error(`Unsupported CPU architecture: ${process.arch}`);
  }

  if (process.platform === 'win32') {
    return `${arch}-pc-windows-msvc`;
  }
  if (process.platform === 'darwin') {
    return `${arch}-apple-darwin`;
  }
  if (process.platform === 'linux') {
    return `${arch}-unknown-linux-gnu`;
  }
  throw new Error(`Unsupported platform: ${process.platform}`);
}

function rmrf(target) {
  fs.rmSync(target, { recursive: true, force: true });
}

function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const follow = (currentUrl, redirectsLeft) => {
      const client = currentUrl.startsWith('https:') ? https : http;
      const req = client.get(currentUrl, (res) => {
        if (
          res.statusCode &&
          res.statusCode >= 300 &&
          res.statusCode < 400 &&
          res.headers.location &&
          redirectsLeft > 0
        ) {
          res.resume();
          const next = new URL(res.headers.location, currentUrl).toString();
          follow(next, redirectsLeft - 1);
          return;
        }
        if (res.statusCode !== 200) {
          res.resume();
          reject(new Error(`Download failed (${res.statusCode}): ${currentUrl}`));
          return;
        }
        const out = fs.createWriteStream(destPath);
        pipeline(res, out).then(resolve).catch(reject);
      });
      req.on('error', reject);
    };
    follow(url, 10);
  });
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    shell: false,
    ...options,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(' ')} exited with code ${result.status}`);
  }
}

function findSitePackages(runtimeRoot) {
  if (process.platform === 'win32') {
    const dir = path.join(runtimeRoot, 'Lib', 'site-packages');
    return fs.existsSync(dir) ? dir : null;
  }
  const libRoot = path.join(runtimeRoot, 'lib');
  if (!fs.existsSync(libRoot)) {
    return null;
  }
  for (const name of fs.readdirSync(libRoot)) {
    if (!/^python\d/.test(name)) {
      continue;
    }
    const dir = path.join(libRoot, name, 'site-packages');
    if (fs.existsSync(dir)) {
      return dir;
    }
  }
  return null;
}

function installCudaSitecustomize(runtimeRoot) {
  const sitePackages = findSitePackages(runtimeRoot);
  if (!sitePackages) {
    throw new Error(`Could not find site-packages under ${runtimeRoot} to install sitecustomize.py`);
  }
  const src = path.join(__dirname, 'glaux_sitecustomize.py');
  const dest = path.join(sitePackages, 'sitecustomize.py');
  fs.copyFileSync(src, dest);
  console.log(`Installed ${path.relative(path.join(__dirname, '..'), dest)}`);
}

function findStdlibDir(runtimeRoot) {
  if (process.platform === 'win32') {
    const dir = path.join(runtimeRoot, 'Lib');
    return fs.existsSync(dir) ? dir : null;
  }
  const libRoot = path.join(runtimeRoot, 'lib');
  if (!fs.existsSync(libRoot)) {
    return null;
  }
  for (const name of fs.readdirSync(libRoot)) {
    if (!/^python\d/.test(name)) {
      continue;
    }
    const dir = path.join(libRoot, name);
    if (fs.statSync(dir).isDirectory()) {
      return dir;
    }
  }
  return null;
}

function pathSize(target) {
  let total = 0;
  const walk = (p) => {
    let st;
    try {
      st = fs.lstatSync(p);
    } catch {
      return;
    }
    if (st.isSymbolicLink() || st.isFile()) {
      total += st.size;
      return;
    }
    if (!st.isDirectory()) {
      return;
    }
    for (const name of fs.readdirSync(p)) {
      walk(path.join(p, name));
    }
  };
  walk(target);
  return total;
}

/** @param {{ rel: string, bytes: number }[]} removed */
function pruneTarget(target, removed) {
  if (!fs.existsSync(target)) {
    return;
  }
  removed.push({ rel: target, bytes: pathSize(target) });
  rmrf(target);
}

function pruneSitePackage(sitePackages, name, removed) {
  pruneTarget(path.join(sitePackages, name), removed);
  for (const entry of fs.readdirSync(sitePackages)) {
    if (entry === name || (entry.startsWith(`${name}-`) && entry.endsWith('.dist-info'))) {
      pruneTarget(path.join(sitePackages, entry), removed);
    } else if (entry === `${name}.pth` || entry.endsWith(`-${name}.pth`)) {
      pruneTarget(path.join(sitePackages, entry), removed);
    }
  }
}

function pruneMsvcLibFiles(runtimeRoot, removed) {
  const walk = (dir) => {
    let ents;
    try {
      ents = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const ent of ents) {
      const full = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        walk(full);
      } else if (ent.isFile() && ent.name.toLowerCase().endsWith('.lib')) {
        pruneTarget(full, removed);
      }
    }
  };
  walk(runtimeRoot);
}

function prunePipScripts(runtimeRoot, removed) {
  const scriptDirs =
    process.platform === 'win32'
      ? [path.join(runtimeRoot, 'Scripts')]
      : [path.join(runtimeRoot, 'bin')];
  const pipName = /^(pip|pip\d+(\.\d+)*)(\.exe|\.cmd)?$/i;
  const easyInstall = /^easy_install(-[\d.]+)?(\.exe|\.cmd)?$/i;
  for (const dir of scriptDirs) {
    if (!fs.existsSync(dir)) {
      continue;
    }
    for (const name of fs.readdirSync(dir)) {
      if (pipName.test(name) || easyInstall.test(name)) {
        pruneTarget(path.join(dir, name), removed);
      }
    }
  }
}

/**
 * Drop wheel pytest trees (directories named tests). Keep public testing
 * packages (numpy.testing, torch.testing) — torch.utils.checkpoint imports
 * the latter.
 * @param {string} sitePackages
 * @param {{ rel: string, bytes: number }[]} removed
 */
function prunePackageTestSuites(sitePackages, removed) {
  const walk = (dir) => {
    let ents;
    try {
      ents = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const ent of ents) {
      if (!ent.isDirectory()) {
        continue;
      }
      const full = path.join(dir, ent.name);
      if (ent.name.toLowerCase() === 'tests') {
        pruneTarget(full, removed);
        continue;
      }
      walk(full);
    }
  };
  walk(sitePackages);
}

function pruneUnusedTorchCudaExtras(runtimeRoot, removed) {
  const torchLib = findTorchLibDir(runtimeRoot);
  if (!torchLib) {
    return;
  }
  for (const name of fs.readdirSync(torchLib)) {
    if (
      /cusolverMg/i.test(name) ||
      /^nvperf_host/i.test(name) ||
      /^libnvperf_host/i.test(name) ||
      /nvrtc.*\.alt/i.test(name)
    ) {
      pruneTarget(path.join(torchLib, name), removed);
    }
  }
}

function pruneTclTk(runtimeRoot, removed) {
  pruneTarget(path.join(runtimeRoot, 'tcl'), removed);
  const dllDir = path.join(runtimeRoot, 'DLLs');
  if (fs.existsSync(dllDir)) {
    for (const name of fs.readdirSync(dllDir)) {
      if (/^(tcl|tk|_tkinter)/i.test(name)) {
        pruneTarget(path.join(dllDir, name), removed);
      }
    }
  }
  const libRoot = path.join(runtimeRoot, 'lib');
  if (!fs.existsSync(libRoot) || process.platform === 'win32') {
    return;
  }
  for (const name of fs.readdirSync(libRoot)) {
    if (/^(tcl|tk)/i.test(name) || /^lib(tcl|tk)/i.test(name)) {
      pruneTarget(path.join(libRoot, name), removed);
    }
  }
}

/**
 * Drop compile-time and installer leftovers that the packaged app never uses.
 * Must run after pip install + the import check.
 * @param {string} runtimeRoot
 */
function prunePackagingLeftovers(runtimeRoot) {
  const removed = [];
  pruneMsvcLibFiles(runtimeRoot, removed);

  const sitePackages = findSitePackages(runtimeRoot);
  if (sitePackages) {
    pruneTarget(path.join(sitePackages, 'torch', 'include'), removed);
    pruneSitePackage(sitePackages, 'pip', removed);
    pruneSitePackage(sitePackages, 'setuptools', removed);
    pruneSitePackage(sitePackages, 'hf_xet', removed);
    pruneTarget(path.join(sitePackages, 'pkg_resources'), removed);
    pruneTarget(path.join(sitePackages, '_distutils_hack'), removed);
    pruneTarget(path.join(sitePackages, 'distutils-precedence.pth'), removed);
    prunePackageTestSuites(sitePackages, removed);
  }

  const stdlib = findStdlibDir(runtimeRoot);
  if (stdlib) {
    pruneTarget(path.join(stdlib, 'ensurepip'), removed);
  }

  prunePipScripts(runtimeRoot, removed);
  pruneTclTk(runtimeRoot, removed);
  pruneUnusedTorchCudaExtras(runtimeRoot, removed);

  const bytes = removed.reduce((sum, item) => sum + item.bytes, 0);
  console.log(
    `Pruned packaging leftovers (${removed.length} paths, ${(bytes / (1024 * 1024)).toFixed(1)} MB)`
  );
}

function resolvePythonExe(runtimeRoot) {
  if (process.platform === 'win32') {
    const exe = path.join(runtimeRoot, 'python.exe');
    if (!fs.existsSync(exe)) {
      throw new Error(`Expected python.exe at ${exe}`);
    }
    return exe;
  }

  const candidates = [
    path.join(runtimeRoot, 'bin', 'python3'),
    path.join(runtimeRoot, 'bin', 'python'),
  ];
  for (const exe of candidates) {
    if (fs.existsSync(exe)) {
      try {
        fs.chmodSync(exe, 0o755);
      } catch {
        /* ignore */
      }
      return exe;
    }
  }
  throw new Error(`Expected bin/python3 under ${runtimeRoot}`);
}

function torchInstallArgs(torchVariant) {
  if (process.platform === 'darwin') {
    if (torchVariant !== 'cpu') {
      console.warn(
        `Note: CUDA variant "${torchVariant}" is not available on macOS; installing default PyPI torch/torchvision (MPS-capable).`
      );
    }
    return [
      '-m',
      'pip',
      'install',
      '--no-warn-script-location',
      `torch==${TORCH_VERSION}`,
      `torchvision==${TORCHVISION_VERSION}`,
    ];
  }

  const index =
    torchVariant === 'cpu'
      ? 'https://download.pytorch.org/whl/cpu'
      : `https://download.pytorch.org/whl/${torchVariant}`;

  return [
    '-m',
    'pip',
    'install',
    '--no-warn-script-location',
    `torch==${TORCH_VERSION}`,
    `torchvision==${TORCHVISION_VERSION}`,
    '--index-url',
    index,
  ];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const repoRoot = path.resolve(__dirname, '..');
  const outDir = args.outDir
    ? path.resolve(args.outDir)
    : path.join(repoRoot, 'vendor', 'python');
  const requirements = path.join(repoRoot, 'engines', 'huggingface', 'requirements.txt');
  const staging = path.join(repoRoot, 'vendor', '_python-staging');
  const triple = pbsTriple();
  const archiveName = `cpython-${args.pythonVersion}+${args.pbsTag}-${triple}-install_only_stripped.tar.gz`;
  const archiveUrl = `https://github.com/astral-sh/python-build-standalone/releases/download/${args.pbsTag}/${archiveName}`;
  const archivePath = path.join(staging, archiveName);

  console.log('=== Glaux Python runtime ===');
  console.log(`Platform: ${process.platform}/${process.arch} (${triple})`);
  console.log(`Python:   ${args.pythonVersion} (python-build-standalone ${args.pbsTag})`);
  console.log(`Torch:    ${args.torchVariant}`);
  console.log(`Output:   ${outDir}`);
  console.log('');

  if (!fs.existsSync(requirements)) {
    throw new Error(`Missing requirements file: ${requirements}`);
  }

  fs.mkdirSync(staging, { recursive: true });

  if (!fs.existsSync(archivePath)) {
    console.log(`Downloading ${archiveName} ...`);
    await downloadFile(archiveUrl, archivePath);
  } else {
    console.log(`Using cached archive: ${archivePath}`);
  }

  console.log('Extracting runtime...');
  const extractRoot = path.join(staging, 'extract');
  rmrf(extractRoot);
  fs.mkdirSync(extractRoot, { recursive: true });
  run('tar', ['-xzf', archivePath, '-C', extractRoot]);

  const extractedPython = path.join(extractRoot, 'python');
  if (!fs.existsSync(extractedPython)) {
    throw new Error(`Archive did not contain a top-level python/ directory: ${archiveName}`);
  }

  rmrf(outDir);
  fs.mkdirSync(path.dirname(outDir), { recursive: true });
  try {
    fs.renameSync(extractedPython, outDir);
  } catch {
    fs.cpSync(extractedPython, outDir, { recursive: true });
    rmrf(extractedPython);
  }

  const pythonExe = resolvePythonExe(outDir);
  console.log(`Python executable: ${pythonExe}`);

  console.log('Ensuring pip...');
  run(pythonExe, ['-m', 'ensurepip', '--upgrade']);

  console.log('Installing torch + torchvision...');
  run(pythonExe, torchInstallArgs(args.torchVariant));

  console.log('Installing remaining requirements...');
  run(pythonExe, ['-m', 'pip', 'install', '--no-warn-script-location', '-r', requirements]);

  if (process.platform !== 'darwin' && args.torchVariant.startsWith('cu')) {
    stageSharedCudaRuntime({ required: false });
    if (args.torchVariant === 'cu130') {
      shareTorchCuda13WithVendor(outDir);
    }
    installCudaSitecustomize(outDir);
  }

  console.log('Verifying imports...');
  run(
    pythonExe,
    [
      '-c',
      'import torch, transformers, PIL, librosa; print("ok", torch.__version__, "cuda", torch.version.cuda, transformers.__version__)',
    ],
    { env: withSharedCudaLibPath({ ...process.env }) }
  );

  console.log('Pruning packaging leftovers...');
  prunePackagingLeftovers(outDir);

  console.log('');
  console.log(`Done. Runtime ready at: ${outDir}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
