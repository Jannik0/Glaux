#!/usr/bin/env node
'use strict';

/**
 * Build a shared LGPL-minimal ffmpeg + ffprobe (plus dav1d) into vendor/ffmpeg.
 *
 * Usage:
 *   node scripts/build-ffmpeg.js
 *   node scripts/build-ffmpeg.js --force
 *   node scripts/build-ffmpeg.js --out-dir=vendor/ffmpeg
 *
 * Windows requires MSYS2 MinGW64 (gcc, nasm, meson, ninja, pkg-config).
 * macOS/Linux require the same tools on PATH (Homebrew / distro packages).
 * Linux also requires patchelf.
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const https = require('https');
const http = require('http');
const path = require('path');
const { pipeline } = require('stream/promises');
const { createWriteStream } = require('fs');
const os = require('os');
const { which, requirePatchelf, copyFile, finishStagedNativeDir } = require('./gpuBackends');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_OUT = path.join(ROOT, 'vendor', 'ffmpeg');
const WORK_ROOT = path.join(ROOT, 'deps', 'ffmpeg-glaux');

const FFMPEG_VERSION = '7.1.1';
const FFMPEG_TAG = `n${FFMPEG_VERSION}`;
const FFMPEG_URL = `https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_TAG}.tar.gz`;
const DAV1D_VERSION = '1.5.1';
const DAV1D_URL = `https://github.com/videolan/dav1d/archive/refs/tags/${DAV1D_VERSION}.tar.gz`;

const DEMUXERS = [
  'mov',
  'matroska',
  'avi',
  'wav',
  'mp3',
  'ogg',
  'flac',
  'aac',
  'mpegts',
];

const VIDEO_DECODERS = [
  'h264',
  'hevc',
  'vp8',
  'vp9',
  'mpeg4',
  'mpeg1video',
  'mpeg2video',
  'mjpeg',
  'libdav1d',
];

const AUDIO_DECODERS = [
  'aac',
  'aac_latm',
  'mp3float',
  'mp3',
  'opus',
  'vorbis',
  'flac',
  'mp2',
  'alac',
  'ac3',
  'eac3',
  'dca',
  'truehd',
  'mlp',
  'pcm_s16le',
  'pcm_s24le',
  'pcm_s32le',
  'pcm_f32le',
  'pcm_u8',
  'pcm_s16be',
  'pcm_s24be',
  'pcm_s32be',
  'pcm_f32be',
  'pcm_bluray',
  'pcm_dvd',
  'adpcm_ms',
  'adpcm_ima_wav',
  'pcm_mulaw',
  'pcm_alaw',
];

const PARSERS = [
  'h264',
  'hevc',
  'aac',
  'ac3',
  'dca',
  'mpegvideo',
  'mpeg4video',
  'mpegaudio',
  'opus',
  'vorbis',
  'vp8',
  'vp9',
  'flac',
  'av1',
  'mlp',
];

const BSFS = [
  'h264_mp4toannexb',
  'hevc_mp4toannexb',
  'aac_adtstoasc',
  'extract_extradata',
  'dca_core',
];

const FILTERS = ['fps', 'aresample', 'aformat', 'select', 'yadif', 'scale', 'format'];

function printHelp() {
  console.log(`Usage: node scripts/build-ffmpeg.js [options]

Options:
  --out-dir=<path>     Stage directory (default: vendor/ffmpeg)
  --jobs=<n>           Parallel build jobs (default: CPU count)
  --force              Rebuild even if ffmpeg/ffprobe are already staged
  --work-dir=<path>    Source/prefix scratch dir (default: deps/ffmpeg-glaux)
  -h, --help           Show this help

Requires nasm, meson, ninja, pkg-config, and a C compiler.
On Windows, run from a host with MSYS2 MinGW64 installed.
`);
}

function parseArgs(argv) {
  const opts = {
    outDir: DEFAULT_OUT,
    workDir: WORK_ROOT,
    jobs: Math.max(1, os.cpus().length || 4),
    force: false,
  };
  for (const arg of argv) {
    if (arg === '-h' || arg === '--help') {
      opts.help = true;
    } else if (arg === '--force') {
      opts.force = true;
    } else if (arg.startsWith('--out-dir=')) {
      opts.outDir = path.resolve(arg.slice('--out-dir='.length));
    } else if (arg.startsWith('--work-dir=')) {
      opts.workDir = path.resolve(arg.slice('--work-dir='.length));
    } else if (arg.startsWith('--jobs=')) {
      opts.jobs = Math.max(1, Number(arg.slice('--jobs='.length)) || opts.jobs);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return opts;
}

function ffmpegBinName() {
  return process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg';
}

function ffprobeBinName() {
  return process.platform === 'win32' ? 'ffprobe.exe' : 'ffprobe';
}

function alreadyStaged(outDir) {
  return (
    fs.existsSync(path.join(outDir, ffmpegBinName())) &&
    fs.existsSync(path.join(outDir, ffprobeBinName()))
  );
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = createWriteStream(dest);
    const follow = (currentUrl, redirects = 0) => {
      if (redirects > 8) {
        reject(new Error('Too many redirects'));
        return;
      }
      const getter = currentUrl.startsWith('https:') ? https.get : http.get;
      getter(currentUrl, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          follow(res.headers.location, redirects + 1);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`Download failed HTTP ${res.statusCode} for ${currentUrl}`));
          res.resume();
          return;
        }
        pipeline(res, file).then(resolve).catch(reject);
      }).on('error', reject);
    };
    follow(url);
  });
}

function toPosixPath(p) {
  const resolved = path.resolve(p);
  if (process.platform !== 'win32') {
    return resolved.replace(/\\/g, '/');
  }
  const m = resolved.match(/^([A-Za-z]):(.*)$/);
  if (!m) {
    return resolved.replace(/\\/g, '/');
  }
  return `/${m[1].toLowerCase()}${m[2].replace(/\\/g, '/')}`;
}

function findMsysBash() {
  const candidates = [
    process.env.MSYS2_BASH,
    process.env.MSYS2_PATH && path.join(process.env.MSYS2_PATH, 'usr', 'bin', 'bash.exe'),
    'C:\\msys64\\usr\\bin\\bash.exe',
    'D:\\msys64\\usr\\bin\\bash.exe',
    'C:\\msys32\\usr\\bin\\bash.exe',
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  return which('bash');
}

function runBash(script, options = {}) {
  const env = { ...process.env, ...(options.env || {}) };
  let bash;
  let args;
  if (process.platform === 'win32') {
    bash = findMsysBash();
    if (!bash) {
      throw new Error(
        'MSYS2 bash not found. Install MSYS2 MinGW64 (https://www.msys2.org/) and the packages: mingw-w64-x86_64-gcc nasm meson ninja pkg-config.'
      );
    }
    env.MSYSTEM = env.MSYSTEM || 'MINGW64';
    env.CHERE_INVOKING = '1';
    args = ['-lc', script];
  } else {
    bash = which('bash') || '/bin/bash';
    args = ['-lc', script];
  }
  console.log(`> bash -lc <build script>`);
  const result = spawnSync(bash, args, {
    stdio: 'inherit',
    env,
    cwd: options.cwd || process.cwd(),
  });
  if (result.status !== 0) {
    throw new Error(`FFmpeg build script failed (${result.status})`);
  }
}

function extractTarGz(archive, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  const tar = process.platform === 'win32' ? 'tar' : which('tar') || 'tar';
  const result = spawnSync(tar, ['-xzf', archive, '-C', destDir], { stdio: 'inherit' });
  if (result.status !== 0) {
    throw new Error(`Failed to extract ${archive}`);
  }
}

function findExtractedDir(parent, prefix) {
  const names = fs.readdirSync(parent);
  const match = names.find((n) => n.startsWith(prefix) && fs.statSync(path.join(parent, n)).isDirectory());
  if (!match) {
    throw new Error(`Extracted directory starting with ${prefix} not found in ${parent}`);
  }
  return path.join(parent, match);
}

function enableFlags(kind, names) {
  return names.map((n) => `--enable-${kind}=${n}`).join(' \\\n  ');
}

/**
 * Debian/Ubuntu meson defaults to lib/<triplet> (GNUInstallDirs). MinGW and
 * macOS use lib/. Include both so pkg-config and the linker find dav1d.
 */
function linuxMultiarchTriplet() {
  if (process.platform !== 'linux') {
    return null;
  }
  if (process.arch === 'x64') {
    return 'x86_64-linux-gnu';
  }
  if (process.arch === 'arm64') {
    return 'aarch64-linux-gnu';
  }
  return null;
}

function prefixLibSearchDirs(prefixPosix) {
  const dirs = [`${prefixPosix}/lib`, `${prefixPosix}/lib64`];
  const triplet = linuxMultiarchTriplet();
  if (triplet) {
    dirs.push(`${prefixPosix}/lib/${triplet}`);
  }
  return dirs;
}

function prefixPkgConfigPath(prefixPosix) {
  return prefixLibSearchDirs(prefixPosix)
    .map((dir) => `${dir}/pkgconfig`)
    .join(':');
}

function ffmpegConfigureFlags(prefixPosix, extraLdflags, { pic = false } = {}) {
  return [
    `--prefix=${prefixPosix}`,
    '--enable-shared',
    '--disable-static',
    '--disable-doc',
    '--disable-htmlpages',
    '--disable-manpages',
    '--disable-podpages',
    '--disable-txtpages',
    '--disable-ffplay',
    '--disable-network',
    '--disable-autodetect',
    '--disable-debug',
    pic ? '--enable-pic' : '',
    '--disable-everything',
    '--enable-avcodec',
    '--enable-avformat',
    '--enable-avutil',
    '--enable-avfilter',
    '--enable-swscale',
    '--enable-swresample',
    '--enable-ffmpeg',
    '--enable-ffprobe',
    '--enable-libdav1d',
    '--enable-protocol=file',
    '--enable-protocol=pipe',
    '--enable-protocol=cache',
    '--enable-encoder=pcm_s16le',
    '--enable-encoder=pcm_f32le',
    '--enable-encoder=rawvideo',
    '--enable-muxer=wav',
    '--enable-muxer=rawvideo',
    // Registered as pcm_* in configure; CLI names remain -f f32le / -f s16le.
    '--enable-muxer=pcm_f32le',
    '--enable-muxer=pcm_s16le',
    enableFlags('demuxer', DEMUXERS),
    enableFlags('decoder', [...VIDEO_DECODERS, ...AUDIO_DECODERS]),
    enableFlags('parser', PARSERS),
    enableFlags('bsf', BSFS),
    enableFlags('filter', FILTERS),
    extraLdflags,
  ]
    .filter(Boolean)
    .join(' \\\n  ');
}

function writeUnixBuildScript(opts) {
  const { workDir, prefixDir, ffmpegSrc, dav1dSrc, jobs } = opts;
  const prefixPosix = toPosixPath(prefixDir);
  const dav1dPosix = toPosixPath(dav1dSrc);
  const ffmpegPosix = toPosixPath(ffmpegSrc);
  const rpath =
    process.platform === 'darwin' ? '-Wl,-rpath,@loader_path' : '-Wl,-rpath,\\$ORIGIN';
  const libFlags = prefixLibSearchDirs(prefixPosix)
    .map((dir) => `-L${dir}`)
    .join(' ');
  const extraLdflags = `--extra-cflags=-I${prefixPosix}/include --extra-ldflags="${libFlags} ${rpath}"`;
  const pkgConfig = prefixPkgConfigPath(prefixPosix);

  const body = `#!/usr/bin/env bash
set -euo pipefail
export PKG_CONFIG_PATH="${pkgConfig}\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}"
export PATH="${prefixPosix}/bin:\${PATH}"

echo "==> Building dav1d ${DAV1D_VERSION}"
cd "${dav1dPosix}"
rm -rf build-glaux
meson setup build-glaux \\
  --prefix="${prefixPosix}" \\
  --libdir=lib \\
  --default-library=shared \\
  --buildtype=release \\
  -Denable_tools=false \\
  -Denable_tests=false
meson compile -C build-glaux -j ${jobs}
meson install -C build-glaux

echo "==> Building FFmpeg ${FFMPEG_VERSION}"
cd "${ffmpegPosix}"
./configure \\
  ${ffmpegConfigureFlags(prefixPosix, extraLdflags, { pic: true })}
make -j ${jobs}
make install
`;
  const scriptPath = path.join(workDir, 'build.sh');
  fs.writeFileSync(scriptPath, body.replace(/\r\n/g, '\n'));
  return scriptPath;
}

function writeWindowsBuildScript(opts) {
  const { workDir, prefixDir, ffmpegSrc, dav1dSrc, jobs } = opts;
  const prefixPosix = toPosixPath(prefixDir);
  const dav1dPosix = toPosixPath(dav1dSrc);
  const ffmpegPosix = toPosixPath(ffmpegSrc);
  const extraLdflags = `--extra-cflags=-I${prefixPosix}/include --extra-ldflags="-L${prefixPosix}/lib"`;
  const pkgConfig = `${prefixPosix}/lib/pkgconfig`;

  const body = `#!/usr/bin/env bash
set -euo pipefail
export PATH="/mingw64/bin:/usr/bin:\${PATH}"
export PKG_CONFIG_PATH="${pkgConfig}\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}"
export PATH="${prefixPosix}/bin:\${PATH}"

need() {
  command -v "\$1" >/dev/null 2>&1 || {
    echo "Missing \$1. In an MSYS2 MinGW64 shell: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-nasm mingw-w64-x86_64-meson mingw-w64-x86_64-ninja mingw-w64-x86_64-pkg-config" >&2
    exit 1
  }
}
need gcc
need nasm
need meson
need ninja
need pkg-config

echo "==> Building dav1d ${DAV1D_VERSION}"
cd "${dav1dPosix}"
rm -rf build-glaux
meson setup build-glaux \\
  --prefix="${prefixPosix}" \\
  --libdir=lib \\
  --default-library=shared \\
  --buildtype=release \\
  -Denable_tools=false \\
  -Denable_tests=false
meson compile -C build-glaux -j ${jobs}
meson install -C build-glaux

echo "==> Building FFmpeg ${FFMPEG_VERSION}"
cd "${ffmpegPosix}"
./configure \\
  ${ffmpegConfigureFlags(prefixPosix, extraLdflags)}
make -j ${jobs}
make install
`;
  const scriptPath = path.join(workDir, 'build.sh');
  fs.writeFileSync(scriptPath, body.replace(/\r\n/g, '\n'));
  return scriptPath;
}

function copyMatching(dir, predicate, destDir) {
  if (!fs.existsSync(dir)) {
    return;
  }
  for (const name of fs.readdirSync(dir)) {
    if (!predicate(name)) {
      continue;
    }
    const src = path.join(dir, name);
    if (!fs.statSync(src).isFile() && !fs.lstatSync(src).isSymbolicLink()) {
      continue;
    }
    copyFile(src, path.join(destDir, name));
  }
}

function isSharedLib(name) {
  if (process.platform === 'win32') {
    return /\.dll$/i.test(name);
  }
  if (process.platform === 'darwin') {
    return /\.dylib$/.test(name);
  }
  return /\.so(\.\d+)*$/.test(name);
}

/**
 * Bake $ORIGIN into DT_RUNPATH so ffmpeg loads sibling libav*.so without
 * LD_LIBRARY_PATH. The configure ldflag uses `\$ORIGIN` so bash does not
 * expand `$$` to the shell PID. patchelf is required on Linux when the baked
 * runpath is wrong.
 *
 * @param {string} outDir
 */
function fixLinuxFfmpegRunpaths(outDir) {
  if (process.platform !== 'linux') {
    return;
  }
  const patchelf = requirePatchelf();
  for (const name of fs.readdirSync(outDir)) {
    if (name !== 'ffmpeg' && name !== 'ffprobe' && !isSharedLib(name)) {
      continue;
    }
    const file = path.join(outDir, name);
    try {
      if (fs.lstatSync(file).isSymbolicLink()) {
        continue;
      }
    } catch {
      continue;
    }
    const result = spawnSync(patchelf, ['--set-rpath', '$ORIGIN', file], { encoding: 'utf8' });
    if (result.status !== 0) {
      console.warn(`patchelf --set-rpath failed for ${name}: ${(result.stderr || '').trim()}`);
    }
  }
}

function listPrefixLibDirs(prefixDir) {
  const dirs = [];
  const seen = new Set();
  const add = (dir) => {
    if (seen.has(dir) || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
      return;
    }
    seen.add(dir);
    dirs.push(dir);
  };
  add(path.join(prefixDir, 'lib'));
  add(path.join(prefixDir, 'lib64'));
  const libRoot = path.join(prefixDir, 'lib');
  if (fs.existsSync(libRoot) && fs.statSync(libRoot).isDirectory()) {
    for (const name of fs.readdirSync(libRoot)) {
      if (/-linux-gnu$/.test(name)) {
        add(path.join(libRoot, name));
      }
    }
  }
  return dirs;
}

function stagePrefixToVendor(prefixDir, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  const binDir = path.join(prefixDir, 'bin');

  const ffmpegSrc = path.join(binDir, ffmpegBinName());
  const ffprobeSrc = path.join(binDir, ffprobeBinName());
  if (!fs.existsSync(ffmpegSrc) || !fs.existsSync(ffprobeSrc)) {
    throw new Error(`ffmpeg/ffprobe missing under ${binDir} after install`);
  }
  copyFile(ffmpegSrc, path.join(outDir, ffmpegBinName()));
  copyFile(ffprobeSrc, path.join(outDir, ffprobeBinName()));
  if (process.platform !== 'win32') {
    fs.chmodSync(path.join(outDir, ffmpegBinName()), 0o755);
    fs.chmodSync(path.join(outDir, ffprobeBinName()), 0o755);
  }

  copyMatching(binDir, isSharedLib, outDir);
  for (const libDir of listPrefixLibDirs(prefixDir)) {
    copyMatching(libDir, isSharedLib, outDir);
  }
  finishStagedNativeDir(outDir);
  fixLinuxFfmpegRunpaths(outDir);
}

function copyLicenses(ffmpegSrc, dav1dSrc, outDir) {
  const candidates = [
    [path.join(ffmpegSrc, 'COPYING.LGPLv2.1'), 'COPYING.LGPLv2.1'],
    [path.join(ffmpegSrc, 'LICENSE.md'), 'FFMPEG.LICENSE.md'],
    [path.join(dav1dSrc, 'COPYING'), 'DAV1D.COPYING'],
  ];
  for (const [src, destName] of candidates) {
    if (fs.existsSync(src)) {
      copyFile(src, path.join(outDir, destName));
    }
  }
}

function copyMingwRuntime(outDir) {
  if (process.platform !== 'win32') {
    return;
  }
  const mingwBin = path.join(path.dirname(findMsysBash() || ''), '..', '..', 'mingw64', 'bin');
  const resolved = fs.existsSync(mingwBin)
    ? mingwBin
    : 'C:\\msys64\\mingw64\\bin';
  if (!fs.existsSync(resolved)) {
    return;
  }
  const runtimeDlls = [
    'libgcc_s_seh-1.dll',
    'libwinpthread-1.dll',
    'libstdc++-6.dll',
  ];
  for (const name of runtimeDlls) {
    const src = path.join(resolved, name);
    if (fs.existsSync(src)) {
      copyFile(src, path.join(outDir, name));
    }
  }
}

function fixMacLoaderPaths(outDir) {
  if (process.platform !== 'darwin') {
    return;
  }
  const installNameTool = which('install_name_tool');
  const otool = which('otool');
  if (!installNameTool || !otool) {
    console.warn('install_name_tool/otool not found; skipping macOS rpath rewrite.');
    return;
  }
  const entries = fs.readdirSync(outDir).filter((n) => /\.dylib$/.test(n) || n === 'ffmpeg' || n === 'ffprobe');
  for (const name of entries) {
    const file = path.join(outDir, name);
    if (/\.dylib$/.test(name)) {
      spawnSync(installNameTool, ['-id', `@loader_path/${name}`, file], { stdio: 'inherit' });
    }
    spawnSync(installNameTool, ['-add_rpath', '@loader_path', file], { stdio: 'ignore' });
    const listed = spawnSync(otool, ['-L', file], { encoding: 'utf8' });
    if (listed.status !== 0 || !listed.stdout) {
      continue;
    }
    for (const line of listed.stdout.split('\n').slice(1)) {
      const dep = line.trim().split(' ')[0];
      if (!dep || dep.startsWith('@') || dep.startsWith('/usr/lib') || dep.startsWith('/System/')) {
        continue;
      }
      const base = path.basename(dep);
      if (!fs.existsSync(path.join(outDir, base))) {
        continue;
      }
      spawnSync(installNameTool, ['-change', dep, `@loader_path/${base}`, file], { stdio: 'inherit' });
    }
  }
}

async function ensureArchive(url, dest, label) {
  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    console.log(`${label} archive already present: ${dest}`);
    return;
  }
  console.log(`Downloading ${label} from ${url}`);
  await download(url, dest);
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printHelp();
    return;
  }

  fs.mkdirSync(opts.outDir, { recursive: true });
  if (!opts.force && alreadyStaged(opts.outDir)) {
    console.log(`ffmpeg/ffprobe already staged in ${opts.outDir} (pass --force to rebuild).`);
    return;
  }

  if (process.platform !== 'win32') {
    for (const tool of ['nasm', 'meson', 'ninja']) {
      if (!which(tool)) {
        throw new Error(`Missing ${tool} on PATH. Install nasm, meson, ninja, and pkg-config, then retry.`);
      }
    }
    if (!which('pkg-config') && !which('pkgconf')) {
      throw new Error('Missing pkg-config on PATH. Install pkg-config, then retry.');
    }
  }
  requirePatchelf();

  const workDir = opts.workDir;
  const srcDir = path.join(workDir, 'src');
  const prefixDir = path.join(workDir, 'prefix');
  fs.mkdirSync(srcDir, { recursive: true });
  fs.rmSync(prefixDir, { recursive: true, force: true });
  fs.mkdirSync(prefixDir, { recursive: true });

  const ffmpegArchive = path.join(workDir, `ffmpeg-${FFMPEG_VERSION}.tar.gz`);
  const dav1dArchive = path.join(workDir, `dav1d-${DAV1D_VERSION}.tar.gz`);
  await ensureArchive(FFMPEG_URL, ffmpegArchive, 'FFmpeg');
  await ensureArchive(DAV1D_URL, dav1dArchive, 'dav1d');

  const ffmpegExtractRoot = path.join(srcDir, 'ffmpeg');
  const dav1dExtractRoot = path.join(srcDir, 'dav1d');
  fs.rmSync(ffmpegExtractRoot, { recursive: true, force: true });
  fs.rmSync(dav1dExtractRoot, { recursive: true, force: true });
  extractTarGz(ffmpegArchive, ffmpegExtractRoot);
  extractTarGz(dav1dArchive, dav1dExtractRoot);
  const ffmpegSrc = findExtractedDir(ffmpegExtractRoot, 'FFmpeg-');
  const dav1dSrc = findExtractedDir(dav1dExtractRoot, 'dav1d-');

  const scriptPath =
    process.platform === 'win32'
      ? writeWindowsBuildScript({ workDir, prefixDir, ffmpegSrc, dav1dSrc, jobs: opts.jobs })
      : writeUnixBuildScript({ workDir, prefixDir, ffmpegSrc, dav1dSrc, jobs: opts.jobs });

  const scriptPosix = toPosixPath(scriptPath);
  runBash(`bash "${scriptPosix}"`);

  fs.rmSync(opts.outDir, { recursive: true, force: true });
  fs.mkdirSync(opts.outDir, { recursive: true });
  stagePrefixToVendor(prefixDir, opts.outDir);
  copyLicenses(ffmpegSrc, dav1dSrc, opts.outDir);
  copyMingwRuntime(opts.outDir);
  fixMacLoaderPaths(opts.outDir);

  if (!alreadyStaged(opts.outDir)) {
    throw new Error(`Staging failed: ${ffmpegBinName()} / ${ffprobeBinName()} missing in ${opts.outDir}`);
  }
  console.log(`FFmpeg ${FFMPEG_VERSION} + dav1d ${DAV1D_VERSION} staged at ${opts.outDir}`);
}

main().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
