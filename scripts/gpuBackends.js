'use strict';

/**
 * Shared GPU backend helpers for native llama.cpp / transcribe.cpp builds
 * and packaging checks. One installer per OS ships every backend that OS
 * supports; missing NVIDIA/Vulkan drivers at runtime skip those modules.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const GPU_BACKENDS = ['cuda', 'vulkan', 'metal'];

/**
 * @param {string} [platform]
 * @returns {string[]}
 */
function expectedGpuBackends(platform = process.platform) {
  if (platform === 'darwin') {
    return ['metal'];
  }
  if (platform === 'win32' || platform === 'linux') {
    return ['cuda', 'vulkan'];
  }
  return [];
}

/**
 * @param {string} bin
 * @returns {string | null}
 */
function which(bin) {
  try {
    const cmd = process.platform === 'win32' ? 'where' : 'which';
    const out = execSync(`${cmd} ${bin}`, { encoding: 'utf8' }).trim().split(/\r?\n/)[0];
    return out || null;
  } catch {
    return null;
  }
}

/**
 * @param {string} fileName
 * @returns {boolean}
 */
function isNativeLibName(fileName) {
  if (process.platform === 'win32') {
    return /\.dll$/i.test(fileName);
  }
  if (process.platform === 'darwin') {
    return /\.dylib$/i.test(fileName) || /\.so$/i.test(fileName);
  }
  return /\.so(?:\.\d+)*$/i.test(fileName);
}

/**
 * @param {string} dest
 * @param {string} src
 */
function copyFile(src, dest) {
  if (path.resolve(src) === path.resolve(dest)) {
    return;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
  if (process.platform !== 'win32') {
    try {
      fs.chmodSync(dest, 0o755);
    } catch {
      /* ignore */
    }
  }
}

/**
 * @returns {string | null}
 */
function findCudaToolkitRoot() {
  const envRoots = [process.env.CUDA_PATH, process.env.CUDA_HOME, process.env.CUDA_ROOT];
  for (const root of envRoots) {
    if (root && fs.existsSync(root)) {
      return root;
    }
  }

  const nvcc = which('nvcc');
  if (nvcc) {
    const binDir = path.dirname(nvcc);
    const root = path.dirname(binDir);
    if (fs.existsSync(root)) {
      return root;
    }
  }

  if (process.platform === 'win32') {
    const toolkit = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA';
    if (fs.existsSync(toolkit)) {
      const versions = fs.readdirSync(toolkit).sort().reverse();
      for (const version of versions) {
        const root = path.join(toolkit, version);
        if (fs.existsSync(path.join(root, 'bin'))) {
          return root;
        }
      }
    }
  }

  if (process.platform === 'linux' && fs.existsSync('/usr/local/cuda')) {
    return '/usr/local/cuda';
  }

  return null;
}

/**
 * @returns {string | null}
 */
function findVulkanSdk() {
  if (process.env.VULKAN_SDK && fs.existsSync(process.env.VULKAN_SDK)) {
    return process.env.VULKAN_SDK;
  }

  if (process.platform === 'win32') {
    const root = 'C:\\VulkanSDK';
    if (fs.existsSync(root)) {
      const versions = fs.readdirSync(root).sort().reverse();
      for (const version of versions) {
        const sdk = path.join(root, version);
        const header = path.join(sdk, 'Include', 'vulkan', 'vulkan.h');
        const headerAlt = path.join(sdk, 'include', 'vulkan', 'vulkan.h');
        if (fs.existsSync(header) || fs.existsSync(headerAlt)) {
          return sdk;
        }
      }
    }
  }

  const headers = ['/usr/include/vulkan/vulkan.h', '/usr/local/include/vulkan/vulkan.h'];
  for (const header of headers) {
    if (fs.existsSync(header)) {
      return path.dirname(path.dirname(path.dirname(header)));
    }
  }

  return null;
}

/**
 * @returns {{
 *   nvcc: string | null,
 *   cudaRoot: string | null,
 *   vulkanSdk: string | null,
 *   hasCuda: boolean,
 *   hasVulkan: boolean,
 *   hasMetal: boolean,
 * }}
 */
function detectToolchains() {
  const nvcc = which('nvcc');
  const cudaRoot = findCudaToolkitRoot();
  const vulkanSdk = findVulkanSdk();
  const hasVulkanHeader =
    Boolean(vulkanSdk) ||
    fs.existsSync('/usr/include/vulkan/vulkan.h') ||
    fs.existsSync('/usr/local/include/vulkan/vulkan.h');
  return {
    nvcc,
    cudaRoot,
    vulkanSdk,
    hasCuda: Boolean(nvcc || cudaRoot),
    hasVulkan: Boolean(vulkanSdk || hasVulkanHeader || which('vulkaninfo')),
    hasMetal: process.platform === 'darwin',
  };
}

/**
 * @param {{ cpuOnly?: boolean, platform?: string }} [opts]
 * @returns {{ cpuOnly: boolean, cuda: boolean, vulkan: boolean, metal: boolean }}
 */
function resolveBuildBackends(opts = {}) {
  const cpuOnly = Boolean(opts.cpuOnly);
  if (cpuOnly) {
    return { cpuOnly: true, cuda: false, vulkan: false, metal: false };
  }

  const expected = expectedGpuBackends(opts.platform);
  const tools = detectToolchains();
  const missing = [];
  if (expected.includes('cuda') && !tools.hasCuda) {
    missing.push('CUDA Toolkit (nvcc on PATH, or CUDA_PATH)');
  }
  if (expected.includes('vulkan') && !tools.hasVulkan) {
    missing.push('Vulkan SDK (VULKAN_SDK, or libvulkan-dev headers)');
  }
  if (expected.includes('metal') && !tools.hasMetal) {
    missing.push('Apple Metal (macOS only)');
  }
  if (missing.length) {
    throw new Error(
      `Missing GPU build toolchains required to ship all backends: ${missing.join('; ')}.\n` +
        `Install them, or pass --cpu-only for a local CPU-only native build (cannot be packaged).`
    );
  }

  return {
    cpuOnly: false,
    cuda: expected.includes('cuda'),
    vulkan: expected.includes('vulkan'),
    metal: expected.includes('metal'),
  };
}

/**
 * @param {'ggml' | 'transcribe'} kind
 * @param {{ cuda: boolean, vulkan: boolean, metal: boolean }} backends
 * @returns {string[]}
 */
function cmakeGpuArgs(kind, backends) {
  const prefix = kind === 'transcribe' ? 'TRANSCRIBE' : 'GGML';
  const onOff = (on) => (on ? 'ON' : 'OFF');
  return [
    `-D${prefix}_CUDA=${onOff(backends.cuda)}`,
    `-D${prefix}_VULKAN=${onOff(backends.vulkan)}`,
    `-D${prefix}_METAL=${onOff(backends.metal)}`,
  ];
}

/**
 * @returns {string[]}
 */
function rpathCmakeArgs() {
  if (process.platform === 'win32') {
    return [];
  }
  // Linux ggml-cuda looks next to the binary and in the shared vendor/cuda folder.
  const rpath =
    process.platform === 'darwin' ? '@loader_path' : '$ORIGIN;$ORIGIN/../cuda';
  return [
    `-DCMAKE_BUILD_RPATH=${rpath}`,
    `-DCMAKE_INSTALL_RPATH=${rpath}`,
    '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
    '-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON',
  ];
}

/**
 * Quiet nvcc's unused-local warnings from ggml mmq templates. Each #177
 * reprints a long instantiation stack; with many .cu files and -j that looks
 * like thousands of errors per second and can hide a real failure.
 *
 * @param {{ cuda: boolean }} backends
 * @returns {string[]}
 */
function cudaQuietCmakeArgs(backends) {
  if (!backends.cuda) {
    return [];
  }
  // One token, no spaces: Windows spawnSync({shell:true}) concatenates args
  // unquoted. Do not use nvcc -w: on MSVC it is forwarded as cl /w and fights
  // CMake's /W1, printing D9025 for every .cu file. --diag-suppress is
  // cudafe-only (ggml mmq unused locals #177, topk 1e+300 #221, unused set #550).
  return ['-DCMAKE_CUDA_FLAGS=--diag-suppress=177,221,550'];
}

/**
 * Hide MSBuild's full nvcc command line for every .cu file. Errors still print.
 *
 * @returns {string[]}
 */
function cmakeBuildQuietArgs() {
  if (process.platform !== 'win32') {
    return [];
  }
  return ['--', '/v:minimal'];
}

/**
 * nvcc of ggml-cuda is RAM-heavy. Unbounded -j plus fat arch lists OOMs and
 * multiplies the template-warning flood.
 *
 * @param {number} requestedJobs
 * @param {{ cuda: boolean }} backends
 * @returns {number}
 */
function cudaBuildJobs(requestedJobs, backends) {
  const n = Math.max(1, requestedJobs);
  if (!backends.cuda) {
    return n;
  }
  return Math.min(n, 4);
}

const STAGED_LIB_NAME_RE = /^(lib)?(ggml|llama|mtmd|transcribe)/i;

/**
 * @param {string} buildDir
 * @param {string} builtBinaryPath
 * @returns {string[]}
 */
function collectSearchDirs(buildDir, builtBinaryPath) {
  const dirs = [];
  const add = (dir) => {
    if (dir && fs.existsSync(dir) && fs.statSync(dir).isDirectory() && !dirs.includes(dir)) {
      dirs.push(dir);
    }
  };
  add(path.dirname(builtBinaryPath));
  add(path.join(buildDir, 'bin'));
  add(path.join(buildDir, 'bin', 'Release'));
  add(path.join(buildDir, 'lib'));
  add(path.join(buildDir, 'lib', 'Release'));
  return dirs;
}

/**
 * Copy native libs from a single directory (siblings of the binary).
 * @param {string} fromDir
 * @param {string} outDir
 */
function stageLibsFromDir(fromDir, outDir) {
  if (!fs.existsSync(fromDir)) {
    return;
  }
  for (const name of fs.readdirSync(fromDir)) {
    if (!isNativeLibName(name) && !/\.metallib$/i.test(name)) {
      continue;
    }
    copyFile(path.join(fromDir, name), path.join(outDir, name));
  }
}

/**
 * Walk the build tree for ggml/llama/transcribe shared modules that did not
 * land next to the binary.
 * @param {string} dir
 * @param {string} outDir
 * @param {number} [depth]
 */
function walkStageNamedLibs(dir, outDir, depth = 0) {
  if (depth > 6 || !fs.existsSync(dir)) {
    return;
  }
  let ents;
  try {
    ents = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const ent of ents) {
    if (ent.name === 'CMakeFiles' || ent.name === '.git' || ent.name === 'e') {
      continue;
    }
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      walkStageNamedLibs(full, outDir, depth + 1);
      continue;
    }
    if ((isNativeLibName(ent.name) && STAGED_LIB_NAME_RE.test(ent.name)) || /\.metallib$/i.test(ent.name)) {
      copyFile(full, path.join(outDir, ent.name));
    }
  }
}

/**
 * Stage the main binary plus shared backend modules into outDir (flat).
 * @param {{ builtPath: string, buildDir: string, outDir: string, destName: string }} opts
 */
function stageNativeRuntime(opts) {
  const { builtPath, buildDir, outDir, destName } = opts;
  fs.mkdirSync(outDir, { recursive: true });
  copyFile(builtPath, path.join(outDir, destName));
  console.log(`Staged ${path.join(outDir, destName)}`);

  for (const dir of collectSearchDirs(buildDir, builtPath)) {
    stageLibsFromDir(dir, outDir);
  }
  walkStageNamedLibs(buildDir, outDir);
}

/**
 * @param {string} dir
 * @param {string} backend  cuda | vulkan | metal | cpu
 * @returns {string | null}
 */
function findBackendModule(dir, backend) {
  if (!fs.existsSync(dir)) {
    return null;
  }
  const needle = new RegExp(`^(lib)?ggml-${backend}(?:-[\\w.]+)?\\.(?:dll|so|dylib)(?:\\..*)?$`, 'i');
  const names = fs.readdirSync(dir);
  const match = names.find((name) => needle.test(name));
  return match ? path.join(dir, match) : null;
}

/**
 * @param {string} dir
 * @param {string[]} backends
 * @returns {string[]} missing backend ids
 */
function missingBackendModules(dir, backends) {
  return backends.filter((backend) => !findBackendModule(dir, backend));
}

const CUDA_WIN_PATTERNS = [
  /^cudart64_.*\.dll$/i,
  /^cublas64_.*\.dll$/i,
  /^cublasLt64_.*\.dll$/i,
  /^nvJitLink.*\.dll$/i,
];
const CUDA_POSIX_PATTERNS = [
  /^libcudart\.so/,
  /^libcublas\.so/,
  /^libcublasLt\.so/,
  /^libnvJitLink\.so/,
];

/**
 * Shared CUDA 13 redistributable directory (vendor/cuda → resources/cuda).
 * @param {string} [repoRoot]
 * @returns {string}
 */
function sharedCudaDir(repoRoot = path.join(__dirname, '..')) {
  return path.join(repoRoot, 'vendor', 'cuda');
}

/**
 * @returns {RegExp[]}
 */
function cudaRedistPatterns() {
  return process.platform === 'win32' ? CUDA_WIN_PATTERNS : CUDA_POSIX_PATTERNS;
}

/**
 * @param {string} name
 * @returns {boolean}
 */
function isCudaRedistributableName(name) {
  return cudaRedistPatterns().some((re) => re.test(name));
}

/**
 * Unversioned ELF SONAME links (libcudart.so → libcudart.so.13).
 * @param {string} name
 * @returns {boolean}
 */
function isUnversionedCudaRedistName(name) {
  return process.platform !== 'win32' && /^lib(cudart|cublasLt|cublas|nvJitLink)\.so$/.test(name);
}

/**
 * True for CUDA 13 redists (cudart64_13.dll, nvJitLink_130_0.dll, libcudart.so.13).
 * @param {string} name
 * @returns {boolean}
 */
function isCuda13RedistName(name) {
  return /(?:64_13|nvJitLink_13)/i.test(name) || /\.so\.13(\.|$)/.test(name);
}

/**
 * @param {string} name
 * @returns {boolean}
 */
function isShareableCuda13Name(name) {
  return isCudaRedistributableName(name) && (isCuda13RedistName(name) || isUnversionedCudaRedistName(name));
}

/**
 * @param {string} dir
 * @returns {string[]}
 */
function listCudaRedistributables(dir) {
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs.readdirSync(dir).filter((name) => isCudaRedistributableName(name));
}

/**
 * @param {string} dir
 */
function removeStagedCudaRedistributables(dir) {
  if (!fs.existsSync(dir)) {
    return;
  }
  for (const name of listCudaRedistributables(dir)) {
    fs.unlinkSync(path.join(dir, name));
  }
}

/**
 * @param {string[]} names
 * @returns {boolean}
 */
function hasCuda13CudartAndCublas(names) {
  const hasCudart13 = names.some((name) => /cudart/i.test(name) && isCuda13RedistName(name));
  const hasCublas13 = names.some((name) => /cublas(?!Lt)/i.test(name) && isCuda13RedistName(name));
  return hasCudart13 && hasCublas13;
}

/**
 * @param {string} dir
 * @returns {string[]} missing requirement labels
 */
function missingSharedCudaRedists(dir) {
  if (!fs.existsSync(dir)) {
    return ['directory'];
  }
  const names = listCudaRedistributables(dir);
  const missing = [];
  if (!names.some((name) => /cudart/i.test(name) && (isCuda13RedistName(name) || isUnversionedCudaRedistName(name)))) {
    missing.push('cudart (CUDA 13)');
  }
  if (!names.some((name) => /cublas(?!Lt)/i.test(name) && (isCuda13RedistName(name) || isUnversionedCudaRedistName(name)))) {
    missing.push('cublas (CUDA 13)');
  }
  return missing;
}

/**
 * @param {string} fromDir
 * @param {string} outDir
 * @param {{ overwrite?: boolean, cuda13Only?: boolean }} [opts]
 * @returns {string[]}
 */
function copyCudaRedistributablesFromDir(fromDir, outDir, opts = {}) {
  const overwrite = Boolean(opts.overwrite);
  const cuda13Only = opts.cuda13Only !== false;
  if (!fs.existsSync(fromDir)) {
    return [];
  }
  fs.mkdirSync(outDir, { recursive: true });
  const copied = [];
  for (const name of fs.readdirSync(fromDir)) {
    if (cuda13Only ? !isShareableCuda13Name(name) : !isCudaRedistributableName(name)) {
      continue;
    }
    const dest = path.join(outDir, name);
    if (!overwrite && fs.existsSync(dest)) {
      continue;
    }
    copyFile(path.join(fromDir, name), dest);
    copied.push(name);
  }
  return copied;
}

/**
 * Copy CUDA runtime redistributables (not the driver lib nvcuda/libcuda)
 * into the shared vendor/cuda folder (or an explicit outDir).
 * @param {string} [outDir]
 * @param {string | null} [cudaRoot]
 */
function copyCudaRedistributables(outDir = sharedCudaDir(), cudaRoot = findCudaToolkitRoot()) {
  if (!cudaRoot) {
    throw new Error('CUDA toolkit root not found; cannot stage cudart/cublas redistributables.');
  }

  const searchDirs =
    process.platform === 'win32'
      ? [path.join(cudaRoot, 'bin'), path.join(cudaRoot, 'bin', 'x64')]
      : [
          path.join(cudaRoot, 'lib64'),
          path.join(cudaRoot, 'lib'),
          path.join(cudaRoot, 'lib', 'x86_64-linux-gnu'),
        ];

  fs.mkdirSync(outDir, { recursive: true });
  removeStagedCudaRedistributables(outDir);

  const copied = [];
  for (const dir of searchDirs) {
    copied.push(...copyCudaRedistributablesFromDir(dir, outDir, { overwrite: true, cuda13Only: false }));
  }

  const hasCudart = copied.some((name) => /cudart/i.test(name));
  const hasCublas = copied.some((name) => /cublas(?!Lt)/i.test(name));
  if (!hasCudart || !hasCublas) {
    throw new Error(
      `Failed to stage CUDA redistributables from ${cudaRoot} (found: ${copied.join(', ') || 'none'}). ` +
        `Need cudart and cublas in the shared CUDA runtime folder.`
    );
  }
  if (!hasCuda13CudartAndCublas(copied)) {
    throw new Error(
      `Staged CUDA redistributables from ${cudaRoot} are not CUDA 13 ` +
        `(found: ${copied.join(', ')}). Install CUDA Toolkit 13.x so Torch cu130, ` +
        `llama.cpp, and transcribe.cpp can share one runtime.`
    );
  }
  console.log(`Staged shared CUDA 13 redistributables (${copied.length}): ${copied.join(', ')}`);
}

/**
 * Stage vendor/cuda from the build-machine toolkit.
 * @param {{ required?: boolean, outDir?: string }} [opts]
 * @returns {string | null} staged directory, or null when skipped
 */
function stageSharedCudaRuntime(opts = {}) {
  if (process.platform === 'darwin') {
    return null;
  }
  const outDir = opts.outDir || sharedCudaDir();
  const required = opts.required !== false;
  const cudaRoot = findCudaToolkitRoot();
  if (!cudaRoot) {
    if (required) {
      throw new Error('CUDA toolkit root not found; cannot stage cudart/cublas redistributables.');
    }
    return null;
  }
  copyCudaRedistributables(outDir, cudaRoot);
  return outDir;
}

/**
 * @param {string} pythonRoot
 * @returns {string | null}
 */
function findTorchLibDir(pythonRoot) {
  const candidates = [];
  if (process.platform === 'win32') {
    candidates.push(path.join(pythonRoot, 'Lib', 'site-packages', 'torch', 'lib'));
  } else {
    const libRoot = path.join(pythonRoot, 'lib');
    if (fs.existsSync(libRoot)) {
      for (const name of fs.readdirSync(libRoot)) {
        if (/^python\d/.test(name)) {
          candidates.push(path.join(libRoot, name, 'site-packages', 'torch', 'lib'));
        }
      }
    }
  }
  return candidates.find((dir) => fs.existsSync(dir)) || null;
}

/**
 * Copy CUDA 13 redists from torch/lib into vendor/cuda when missing, then
 * delete the overlapping copies so Torch loads the shared runtime.
 * @param {string} pythonRoot
 * @param {string} [cudaDir]
 * @returns {string[]} removed names
 */
function shareTorchCuda13WithVendor(pythonRoot, cudaDir = sharedCudaDir()) {
  const torchLib = findTorchLibDir(pythonRoot);
  if (!torchLib) {
    return [];
  }
  const names = fs.readdirSync(torchLib).filter((name) => isShareableCuda13Name(name));
  if (!names.length) {
    return [];
  }
  fs.mkdirSync(cudaDir, { recursive: true });
  const merged = copyCudaRedistributablesFromDir(torchLib, cudaDir, { overwrite: false, cuda13Only: true });
  if (merged.length) {
    console.log(`Filled shared CUDA runtime from Torch (${merged.length}): ${merged.join(', ')}`);
  }
  for (const name of names) {
    fs.unlinkSync(path.join(torchLib, name));
  }
  console.log(`Removed overlapping CUDA 13 redists from torch/lib (${names.length}): ${names.join(', ')}`);
  return names;
}

module.exports = {
  GPU_BACKENDS,
  expectedGpuBackends,
  which,
  isNativeLibName,
  copyFile,
  findCudaToolkitRoot,
  findVulkanSdk,
  detectToolchains,
  resolveBuildBackends,
  cmakeGpuArgs,
  rpathCmakeArgs,
  cudaQuietCmakeArgs,
  cmakeBuildQuietArgs,
  cudaBuildJobs,
  collectSearchDirs,
  stageNativeRuntime,
  findBackendModule,
  missingBackendModules,
  sharedCudaDir,
  isCudaRedistributableName,
  isCuda13RedistName,
  isShareableCuda13Name,
  listCudaRedistributables,
  removeStagedCudaRedistributables,
  missingSharedCudaRedists,
  copyCudaRedistributablesFromDir,
  copyCudaRedistributables,
  stageSharedCudaRuntime,
  findTorchLibDir,
  shareTorchCuda13WithVendor,
};
