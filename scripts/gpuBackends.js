'use strict';

/**
 * Shared GPU backend helpers for native llama.cpp / transcribe.cpp builds
 * and packaging checks. One installer per OS ships every backend that OS
 * supports; missing NVIDIA/Vulkan drivers at runtime skip those modules.
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { execSync, spawnSync } = require('child_process');

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
 * Linux vendor builds rewrite ELF RPATH / DT_NEEDED with patchelf.
 * @returns {string | null} patchelf path, or null off Linux
 */
function requirePatchelf() {
  if (process.platform !== 'linux') {
    return null;
  }
  const patchelf = which('patchelf');
  if (patchelf) {
    return patchelf;
  }
  throw new Error(
    'patchelf is required on Linux. Install it (e.g. sudo apt install patchelf) and retry.'
  );
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
 * Training, profiling, and installer-only CUDA bits that Glaux inference never
 * loads. Used both for file names and ELF DT_NEEDED entries. Must not match
 * cuDNN (`cudnn` is not `nccl`). NCCL, NVSHMEM, cuSPARSELt, and cuFile stay in
 * the dynamic table and are replaced with loader stubs (see scripts/cudaStubs.js)
 * because CPython opens torch with RTLD_NOW. Must not match Torch's `nvtx.py`
 * (`torch.cuda.nvtx`).
 *
 * @param {string} name
 * @returns {boolean}
 */
function isDroppedCudaDepName(name) {
  if (/\.(py|pyi|pyc|pyo|pth)$/i.test(name)) {
    return false;
  }
  return (
    /^libnvtx(\.|$)/i.test(name) ||
    /^nvtx\d/i.test(name) ||
    /nvToolsExt/i.test(name) ||
    /cusolverMg/i.test(name) ||
    /nvperf/i.test(name) ||
    /nvrtc.*\.alt/i.test(name) ||
    /^libpcsamplingutil/i.test(name) ||
    /libcheckpoint/i.test(name) ||
    /\.a$/i.test(name)
  );
}

/**
 * @param {string} dir
 * @param {(dir: string) => void} visit
 * @param {number} [depth]
 */
function walkDirs(dir, visit, depth = 0) {
  if (depth > 16 || !dir || !fs.existsSync(dir)) {
    return;
  }
  visit(dir);
  let ents;
  try {
    ents = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const ent of ents) {
    if (!ent.isDirectory() || ent.isSymbolicLink()) {
      continue;
    }
    if (ent.name === '__pycache__' || ent.name === 'CMakeFiles' || ent.name === '.git') {
      continue;
    }
    walkDirs(path.join(dir, ent.name), visit, depth + 1);
  }
}

/**
 * Copy a file, recreating relative symlinks instead of dereferencing them.
 * ELF SONAME links (libfoo.so → libfoo.so.1.2.3) must stay links or Linux
 * vendor trees triple in size.
 *
 * @param {string} src
 * @param {string} dest
 */
function copyFile(src, dest) {
  if (path.resolve(src) === path.resolve(dest)) {
    return;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  try {
    fs.lstatSync(dest);
    fs.rmSync(dest, { force: true });
  } catch {
    /* dest does not exist */
  }

  const srcStat = fs.lstatSync(src);
  if (srcStat.isSymbolicLink()) {
    let target = fs.readlinkSync(src);
    if (path.isAbsolute(target)) {
      target = path.relative(path.dirname(dest), target);
    }
    fs.symlinkSync(target, dest);
    return;
  }

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
 * Prefer the fully versioned ELF name as the canonical regular file
 * (libfoo.so.13.8.0.4 over libfoo.so.13 over libfoo.so).
 *
 * @param {string} name
 * @returns {number}
 */
function sonameRank(name) {
  const match = String(name).match(/\.so(?:\.(.+))?$/i);
  if (!match) {
    return String(name).length;
  }
  if (!match[1]) {
    return 0;
  }
  return match[1].split('.').length * 100 + String(name).length;
}

/**
 * @param {string} filePath
 * @returns {string}
 */
function fileMd5(filePath) {
  const hash = crypto.createHash('md5');
  const fd = fs.openSync(filePath, 'r');
  const buf = Buffer.alloc(8 * 1024 * 1024);
  try {
    let n;
    while ((n = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
      hash.update(n === buf.length ? buf : buf.subarray(0, n));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest('hex');
}

/**
 * Replace identical regular files in a directory with relative symlinks to one
 * canonical copy. No-op on Windows.
 *
 * @param {string} dir
 * @returns {number} number of files converted to symlinks
 */
function collapseDuplicateLibs(dir) {
  if (process.platform === 'win32' || !dir || !fs.existsSync(dir)) {
    return 0;
  }
  const files = [];
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    let st;
    try {
      st = fs.lstatSync(full);
    } catch {
      continue;
    }
    if (!st.isFile() || st.isSymbolicLink()) {
      continue;
    }
    if (!isNativeLibName(name)) {
      continue;
    }
    files.push({ name, full, size: st.size });
  }

  const bySize = new Map();
  for (const file of files) {
    const list = bySize.get(file.size) || [];
    list.push(file);
    bySize.set(file.size, list);
  }

  let collapsed = 0;
  for (const group of bySize.values()) {
    if (group.length < 2) {
      continue;
    }
    const byHash = new Map();
    for (const file of group) {
      const digest = fileMd5(file.full);
      const list = byHash.get(digest) || [];
      list.push(file);
      byHash.set(digest, list);
    }
    for (const identical of byHash.values()) {
      if (identical.length < 2) {
        continue;
      }
      identical.sort((a, b) => sonameRank(b.name) - sonameRank(a.name));
      const canonical = identical[0];
      for (const extra of identical.slice(1)) {
        fs.rmSync(extra.full, { force: true });
        fs.symlinkSync(canonical.name, extra.full);
        collapsed += 1;
      }
    }
  }
  return collapsed;
}

/**
 * Collapse identical native libs in dir and every subdirectory (Python
 * nvidia/<pkg>/lib, torch/lib). No-op on Windows.
 *
 * @param {string} dir
 * @returns {number}
 */
function collapseDuplicateLibsRecursive(dir) {
  let collapsed = 0;
  walkDirs(dir, (current) => {
    collapsed += collapseDuplicateLibs(current);
  });
  return collapsed;
}

/**
 * @param {string} file
 * @returns {string | null}
 */
function elfRunpath(file) {
  const patchelf = process.platform === 'linux' ? requirePatchelf() : which('patchelf');
  if (!patchelf) {
    return null;
  }
  const result = spawnSync(patchelf, ['--print-rpath', file], { encoding: 'utf8' });
  if (result.status !== 0) {
    return null;
  }
  return String(result.stdout || '').trim() || null;
}

/**
 * @param {string} file
 * @param {string} runpath
 */
function setElfRunpath(file, runpath) {
  if (!runpath) {
    return;
  }
  const patchelf = process.platform === 'linux' ? requirePatchelf() : which('patchelf');
  if (!patchelf) {
    return;
  }
  const result = spawnSync(patchelf, ['--set-rpath', runpath, file], { encoding: 'utf8' });
  if (result.status !== 0) {
    console.warn(`patchelf --set-rpath failed for ${path.basename(file)}: ${(result.stderr || '').trim()}`);
  }
}

/**
 * Strip debug/symbol tables from staged llama/transcribe/ffmpeg binaries.
 * Skips symlinks and CUDA redistributables. GNU strip can scramble `$ORIGIN`
 * in DT_RUNPATH, so the runpath is captured and restored via patchelf.
 *
 * @param {string} dir
 * @returns {number} files stripped
 */
function stripNativeBinaries(dir) {
  if (process.platform === 'win32' || !dir || !fs.existsSync(dir)) {
    return 0;
  }
  const stripBin = which('strip');
  if (!stripBin) {
    return 0;
  }
  const args = process.platform === 'darwin' ? ['-x'] : ['--strip-unneeded'];
  let stripped = 0;
  for (const name of fs.readdirSync(dir)) {
    if (isCudaRedistributableName(name) || /\.metallib$/i.test(name)) {
      continue;
    }
    const base = name.replace(/\.exe$/i, '');
    const strippable =
      isNativeLibName(name) || /^(llama-server|transcribe-cli|ffmpeg|ffprobe)$/.test(base);
    if (!strippable) {
      continue;
    }
    const full = path.join(dir, name);
    let st;
    try {
      st = fs.lstatSync(full);
    } catch {
      continue;
    }
    if (!st.isFile() || st.isSymbolicLink()) {
      continue;
    }
    const runpath = process.platform === 'linux' ? elfRunpath(full) : null;
    const result = spawnSync(stripBin, [...args, full], { encoding: 'utf8' });
    if (result.status === 0) {
      stripped += 1;
      if (runpath) {
        const restored = runpath.replace(/(?:\$\$|\d+)ORIGIN/g, '$ORIGIN');
        setElfRunpath(full, restored);
      }
    }
  }
  return stripped;
}

/**
 * Strip native libs/binaries in dir and every subdirectory. No-op on Windows.
 *
 * @param {string} dir
 * @returns {number}
 */
function stripNativeBinariesRecursive(dir) {
  let stripped = 0;
  walkDirs(dir, (current) => {
    stripped += stripNativeBinaries(current);
  });
  return stripped;
}

/**
 * Collapse identical SONAME copies and strip debug symbols. Same call on every
 * OS: both steps are no-ops on Windows (no ELF links / no GNU strip).
 *
 * @param {string} dir
 * @param {{ strip?: boolean }} [opts]
 * @returns {{ collapsed: number, stripped: number }}
 */
function finishStagedNativeDir(dir, opts = {}) {
  const strip = opts.strip !== false;
  if (!dir || !fs.existsSync(dir)) {
    return { collapsed: 0, stripped: 0 };
  }
  const collapsed = collapseDuplicateLibs(dir);
  const stripped = strip ? stripNativeBinaries(dir) : 0;
  if (collapsed) {
    console.log(`Collapsed ${collapsed} duplicate SONAME copy(ies) in ${dir}`);
  }
  if (stripped) {
    console.log(`Stripped ${stripped} native binary(ies) in ${dir}`);
  }
  return { collapsed, stripped };
}

/**
 * @param {string} file
 * @returns {string[]}
 */
function elfNeeded(file) {
  const patchelf = requirePatchelf();
  if (!patchelf) {
    return [];
  }
  const result = spawnSync(patchelf, ['--print-needed', file], { encoding: 'utf8' });
  if (result.status !== 0) {
    return [];
  }
  return String(result.stdout || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

/**
 * Drop DT_NEEDED entries for pruned CUDA extras so libtorch still loads after
 * NVTX / nvperf are deleted. Linux only. Requires patchelf.
 *
 * @param {string} dir
 * @returns {number} entries removed
 */
function removeDroppedElfNeeded(dir) {
  if (process.platform !== 'linux' || !dir || !fs.existsSync(dir)) {
    return 0;
  }
  const patchelf = requirePatchelf();

  let removed = 0;
  const visitFile = (full, name) => {
    if (!isNativeLibName(name) && !/^python3(\.\d+)?$/.test(name) && name !== 'python') {
      return;
    }
    let st;
    try {
      st = fs.lstatSync(full);
    } catch {
      return;
    }
    if (!st.isFile() || st.isSymbolicLink()) {
      return;
    }
    for (const needed of elfNeeded(full)) {
      if (!isDroppedCudaDepName(needed)) {
        continue;
      }
      const result = spawnSync(patchelf, ['--remove-needed', needed, full], { encoding: 'utf8' });
      if (result.status === 0) {
        removed += 1;
      } else {
        throw new Error(
          `patchelf --remove-needed ${needed} failed for ${full}: ${(result.stderr || '').trim()}`
        );
      }
    }
  };

  walkDirs(dir, (current) => {
    let ents;
    try {
      ents = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const ent of ents) {
      if (ent.isDirectory()) {
        continue;
      }
      visitFile(path.join(current, ent.name), ent.name);
    }
  });
  return removed;
}

function nvccBinName() {
  return process.platform === 'win32' ? 'nvcc.exe' : 'nvcc';
}

/**
 * @param {string | null} [cudaRoot]
 * @returns {string | null}
 */
function findNvcc(cudaRoot = findCudaToolkitRoot()) {
  const fromPath = which('nvcc');
  if (fromPath && fs.existsSync(fromPath)) {
    return fromPath;
  }
  if (!cudaRoot) {
    return null;
  }
  const candidates = [path.join(cudaRoot, 'bin', nvccBinName()), path.join(cudaRoot, 'bin', 'x64', nvccBinName())];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
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
 * Prepend a directory to the process path.
 * Linux and macOS use the case-sensitive `PATH` key. Windows stores the same
 * variable as `Path`; adding a second `PATH` key makes CreateProcess keep only
 * the new value and drops every other directory.
 *
 * @param {NodeJS.ProcessEnv} env
 * @param {string} dir
 */
function prependEnvPath(env, dir) {
  if (process.platform !== 'win32') {
    const current = env.PATH || '';
    const parts = current.split(path.delimiter).filter(Boolean);
    if (!parts.includes(dir)) {
      env.PATH = current ? `${dir}${path.delimiter}${current}` : dir;
    }
    return;
  }

  const keys = Object.keys(env).filter((key) => key.toLowerCase() === 'path');
  // Node copies the Windows process environment as `Path`. A later `PATH`
  // assignment is a second entry and hides the original, so keep `Path`.
  const key = keys.find((candidate) => candidate === 'Path') || keys[0] || 'Path';
  const current = env[key] || '';
  for (const extra of keys) {
    if (extra !== key) {
      delete env[extra];
    }
  }

  const parts = current.split(path.delimiter).filter(Boolean);
  const already = parts.some((part) => part.toLowerCase() === dir.toLowerCase());
  if (!already) {
    env[key] = current ? `${dir}${path.delimiter}${current}` : dir;
  }
}

/**
 * Prepend the toolkit bin dir and set CUDA_HOME / CUDACXX so cmake's
 * enable_language(CUDA) finds nvcc. Debian/Ubuntu CUDA packages install
 * nvcc under /usr/local/cuda/bin without putting that dir on PATH;
 * the Windows toolkit installer does add it.
 *
 * @param {NodeJS.ProcessEnv} [baseEnv]
 * @returns {NodeJS.ProcessEnv}
 */
function withCudaToolkitEnv(baseEnv = process.env) {
  const env = { ...baseEnv };
  const cudaRoot = findCudaToolkitRoot();
  const nvcc = findNvcc(cudaRoot);
  if (cudaRoot) {
    env.CUDA_PATH = env.CUDA_PATH || cudaRoot;
    env.CUDA_HOME = env.CUDA_HOME || cudaRoot;
    env.CUDA_ROOT = env.CUDA_ROOT || cudaRoot;
  }
  if (nvcc) {
    env.CUDACXX = env.CUDACXX || nvcc;
    prependEnvPath(env, path.dirname(nvcc));
  }
  return env;
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
 * @returns {string | null}
 */
function findGlslc() {
  const fromPath = which('glslc');
  if (fromPath) {
    return fromPath;
  }
  const sdk = process.env.VULKAN_SDK;
  if (!sdk) {
    return null;
  }
  const candidates = [
    path.join(sdk, 'bin', 'glslc'),
    path.join(sdk, 'Bin', 'glslc.exe'),
    path.join(sdk, 'bin', 'glslc.exe'),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

/**
 * llama.cpp's ggml-vulkan uses find_package(SPIRV-Headers CONFIG REQUIRED).
 * Debian puts this in spirv-headers (/usr/share/cmake/SPIRV-Headers); the
 * Windows Vulkan SDK ships it under VULKAN_SDK.
 *
 * @returns {string | null}
 */
function findSpirvHeadersConfig() {
  const prefixes = [];
  if (process.env.VULKAN_SDK) {
    prefixes.push(process.env.VULKAN_SDK);
  }
  if (process.platform === 'win32') {
    const sdk = findVulkanSdk();
    if (sdk && !prefixes.includes(sdk)) {
      prefixes.push(sdk);
    }
  }
  prefixes.push('/usr', '/usr/local');

  const candidates = [];
  for (const prefix of prefixes) {
    candidates.push(
      path.join(prefix, 'share', 'cmake', 'SPIRV-Headers', 'SPIRV-HeadersConfig.cmake'),
      path.join(prefix, 'lib', 'cmake', 'SPIRV-Headers', 'SPIRV-HeadersConfig.cmake')
    );
  }
  candidates.push(
    '/usr/lib/x86_64-linux-gnu/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake',
    '/usr/lib/aarch64-linux-gnu/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake'
  );
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
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
  const cudaRoot = findCudaToolkitRoot();
  const nvcc = findNvcc(cudaRoot);
  const vulkanSdk = findVulkanSdk();
  const hasVulkanHeader =
    fs.existsSync('/usr/include/vulkan/vulkan.h') ||
    fs.existsSync('/usr/local/include/vulkan/vulkan.h') ||
    Boolean(vulkanSdk);
  const glslc = findGlslc();
  const spirvHeaders = findSpirvHeadersConfig();
  const hasVulkan =
    process.platform === 'win32'
      ? Boolean(vulkanSdk)
      : Boolean(hasVulkanHeader && glslc && spirvHeaders);
  return {
    nvcc,
    cudaRoot,
    vulkanSdk,
    glslc,
    spirvHeaders,
    hasCuda: Boolean(nvcc),
    hasVulkan,
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
    missing.push('Vulkan SDK (VULKAN_SDK, or libvulkan-dev + glslc + spirv-headers)');
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
 * Point cmake at nvcc when it is not on PATH (typical Debian CUDA layout).
 * Skipped on Windows: spawnSync({shell:true}) concatenates unquoted args, and
 * "Program Files" in the nvcc path would split. PATH injection is enough there.
 *
 * @param {{ cuda: boolean }} backends
 * @returns {string[]}
 */
function cudaCompilerCmakeArgs(backends) {
  if (!backends.cuda || process.platform === 'win32') {
    return [];
  }
  const nvcc = findNvcc();
  if (!nvcc) {
    return [];
  }
  return [`-DCMAKE_CUDA_COMPILER=${nvcc}`];
}

/**
 * Turing (75) through Blackwell consumer (120). `-real` skips extra PTX per
 * arch; 120 without `-real` keeps one forward-compat PTX blob. Semicolons are
 * quoted on Windows because spawnSync({shell:true}) otherwise splits the list.
 */
const GGML_CUDA_ARCHITECTURES = '75-real;80-real;86-real;89-real;90-real;100-real;120';

/**
 * @param {string} name
 * @param {string} value
 * @returns {string}
 */
function cmakeD(name, value) {
  if (process.platform === 'win32' && /[;&^]/.test(value)) {
    return `-D${name}="${value}"`;
  }
  return `-D${name}=${value}`;
}

/**
 * @param {{ cuda: boolean }} backends
 * @returns {string[]}
 */
function cudaArchitectureCmakeArgs(backends) {
  if (!backends.cuda) {
    return [];
  }
  return [
    cmakeD('CMAKE_CUDA_ARCHITECTURES', GGML_CUDA_ARCHITECTURES),
    cmakeD('GGML_CUDA_ARCHITECTURES', GGML_CUDA_ARCHITECTURES),
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
  finishStagedNativeDir(outDir);
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
  const collapsed = collapseDuplicateLibs(outDir);
  console.log(
    `Staged shared CUDA 13 redistributables (${copied.length}): ${copied.join(', ')}` +
      (collapsed ? `; collapsed ${collapsed} SONAME copy(ies)` : '')
  );
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
function findSitePackages(pythonRoot) {
  if (process.platform === 'win32') {
    const dir = path.join(pythonRoot, 'Lib', 'site-packages');
    return fs.existsSync(dir) ? dir : null;
  }
  const libRoot = path.join(pythonRoot, 'lib');
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

/**
 * nvidia/{cu13,cublas,...}/lib (POSIX) or .../bin (Windows) next to Torch wheels.
 *
 * @param {string} pythonRoot
 * @returns {string[]}
 */
function findNvidiaCudaLibDirs(pythonRoot) {
  const site = findSitePackages(pythonRoot);
  if (!site) {
    return [];
  }
  const nvidia = path.join(site, 'nvidia');
  if (!fs.existsSync(nvidia)) {
    return [];
  }
  const dirs = [];
  for (const pkg of fs.readdirSync(nvidia)) {
    const pkgDir = path.join(nvidia, pkg);
    let st;
    try {
      st = fs.statSync(pkgDir);
    } catch {
      continue;
    }
    if (!st.isDirectory()) {
      continue;
    }
    for (const sub of ['lib', 'lib64', 'bin']) {
      const dir = path.join(pkgDir, sub);
      if (fs.existsSync(dir) && fs.statSync(dir).isDirectory()) {
        dirs.push(dir);
      }
    }
  }
  return dirs;
}

/**
 * @param {string} pythonRoot
 * @returns {string[]}
 */
function findShareableCudaLibDirs(pythonRoot) {
  const dirs = [];
  const torchLib = findTorchLibDir(pythonRoot);
  if (torchLib) {
    dirs.push(torchLib);
  }
  dirs.push(...findNvidiaCudaLibDirs(pythonRoot));
  return dirs;
}

/**
 * cublasLt must be matched before cublas.
 *
 * @param {string} name
 * @returns {string | null}
 */
function cudaLibFamily(name) {
  const match = String(name).match(/(cublasLt|cublas|cudart|nvJitLink)/i);
  return match ? match[1].toLowerCase() : null;
}

/**
 * @param {string} cudaDir
 * @param {string} name
 * @returns {string | null}
 */
function findSharedCudaMatch(cudaDir, name) {
  if (!fs.existsSync(cudaDir)) {
    return null;
  }
  const exact = path.join(cudaDir, name);
  if (fs.existsSync(exact)) {
    return exact;
  }
  const family = cudaLibFamily(name);
  if (!family) {
    return null;
  }
  const names = fs.readdirSync(cudaDir).filter((candidate) => cudaLibFamily(candidate) === family);
  if (!names.length) {
    return null;
  }
  names.sort((a, b) => sonameRank(b) - sonameRank(a));
  return path.join(cudaDir, names[0]);
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
 * Copy CUDA 13 redists from torch/lib and nvidia/cu13 into vendor/cuda when
 * missing, then drop the overlapping copies from the Python tree. POSIX keeps
 * relative symlinks at the original paths so the dynamic linker still resolves
 * them; Windows deletes them (PATH / add_dll_directory finds vendor/cuda).
 * Same four library families on both OS: cudart, cublas, cublasLt, nvJitLink.
 *
 * @param {string} pythonRoot
 * @param {string} [cudaDir]
 * @returns {string[]} replaced names
 */
function shareTorchCuda13WithVendor(pythonRoot, cudaDir = sharedCudaDir()) {
  const dirs = findShareableCudaLibDirs(pythonRoot);
  if (!dirs.length) {
    return [];
  }
  fs.mkdirSync(cudaDir, { recursive: true });

  const merged = [];
  for (const dir of dirs) {
    merged.push(
      ...copyCudaRedistributablesFromDir(dir, cudaDir, { overwrite: false, cuda13Only: true })
    );
  }
  if (merged.length) {
    console.log(`Filled shared CUDA runtime from Torch/NVIDIA wheels (${merged.length}): ${merged.join(', ')}`);
  }
  collapseDuplicateLibs(cudaDir);

  const replaced = [];
  for (const dir of dirs) {
    if (!fs.existsSync(dir)) {
      continue;
    }
    for (const name of fs.readdirSync(dir)) {
      if (!isShareableCuda13Name(name)) {
        continue;
      }
      const srcPath = path.join(dir, name);
      const cudaFile = findSharedCudaMatch(cudaDir, name);
      if (!cudaFile) {
        continue;
      }
      if (path.resolve(srcPath) === path.resolve(cudaFile)) {
        continue;
      }
      fs.rmSync(srcPath, { force: true });
      if (process.platform !== 'win32') {
        fs.symlinkSync(path.relative(dir, cudaFile), srcPath);
      }
      replaced.push(name);
    }
  }
  if (replaced.length) {
    console.log(
      `Replaced overlapping CUDA 13 redists with shared runtime (${replaced.length}): ${replaced.join(', ')}`
    );
  }
  return replaced;
}

/**
 * Filename of the ggml CUDA backend module for this OS.
 * @param {string} [platform]
 * @returns {string | null}
 */
function ggmlCudaBackendFileName(platform = process.platform) {
  if (platform === 'win32') {
    return 'ggml-cuda.dll';
  }
  if (platform === 'linux') {
    return 'libggml-cuda.so';
  }
  return null;
}

/**
 * Keep one ggml CUDA fatbin. llama.cpp's module is the canonical copy;
 * transcribe.cpp's directory gets a relative symlink (hard link if Windows
 * cannot create a symlink) so both CLIs still load `ggml-cuda` from their
 * own folder. Vulkan modules stay per engine: they compress to a few MB.
 * No-op when either side has not been built yet.
 *
 * @param {string} llamaDir vendor/llamacpp or resources/llamacpp
 * @param {string} transcribeDir vendor/transcribe or resources/transcribe
 * @returns {boolean}
 */
function shareGgmlCudaBackend(llamaDir, transcribeDir) {
  const name = ggmlCudaBackendFileName();
  if (!name || !llamaDir || !transcribeDir) {
    return false;
  }
  const source = path.join(llamaDir, name);
  const dest = path.join(transcribeDir, name);
  if (!fs.existsSync(transcribeDir) || !fs.existsSync(source)) {
    return false;
  }
  let sourceStat;
  try {
    sourceStat = fs.lstatSync(source);
  } catch {
    return false;
  }
  if (!sourceStat.isFile() || sourceStat.isSymbolicLink()) {
    return false;
  }

  const rel = path.relative(transcribeDir, source);
  try {
    if (fs.lstatSync(dest).isSymbolicLink() && fs.readlinkSync(dest) === rel) {
      return true;
    }
  } catch {
    /* dest absent */
  }
  fs.rmSync(dest, { force: true });
  try {
    fs.symlinkSync(rel, dest);
  } catch (err) {
    if (process.platform !== 'win32') {
      throw err;
    }
    fs.linkSync(source, dest);
  }
  console.log(`Shared ggml CUDA backend ${name} -> ${rel}`);
  return true;
}

module.exports = {
  GPU_BACKENDS,
  expectedGpuBackends,
  which,
  requirePatchelf,
  isNativeLibName,
  isDroppedCudaDepName,
  copyFile,
  collapseDuplicateLibs,
  collapseDuplicateLibsRecursive,
  stripNativeBinaries,
  stripNativeBinariesRecursive,
  finishStagedNativeDir,
  elfNeeded,
  removeDroppedElfNeeded,
  setElfRunpath,
  sonameRank,
  findCudaToolkitRoot,
  findNvcc,
  withCudaToolkitEnv,
  findVulkanSdk,
  findGlslc,
  findSpirvHeadersConfig,
  detectToolchains,
  resolveBuildBackends,
  cmakeGpuArgs,
  cudaCompilerCmakeArgs,
  cudaArchitectureCmakeArgs,
  GGML_CUDA_ARCHITECTURES,
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
  findSitePackages,
  findTorchLibDir,
  findNvidiaCudaLibDirs,
  shareTorchCuda13WithVendor,
  findSharedCudaMatch,
  ggmlCudaBackendFileName,
  shareGgmlCudaBackend,
};
