#!/usr/bin/env node
'use strict';

/**
 * Fail packaging if vendor/ffmpeg is missing ffmpeg, ffprobe, or shared libs.
 */

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const vendor = path.join(root, 'vendor', 'ffmpeg');
const ffmpegName = process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg';
const ffprobeName = process.platform === 'win32' ? 'ffprobe.exe' : 'ffprobe';
const ffmpegBin = path.join(vendor, ffmpegName);
const ffprobeBin = path.join(vendor, ffprobeName);

if (!fs.existsSync(ffmpegBin) || !fs.existsSync(ffprobeBin)) {
  console.error(
    `Missing ${ffmpegBin} and/or ${ffprobeBin}.\n` +
      `Run: npm run build:ffmpeg\n` +
      `(Requires nasm, meson, ninja, pkg-config, and a C compiler. On Windows, MSYS2 MinGW64.)`
  );
  process.exit(1);
}

function isAvLib(name) {
  if (process.platform === 'win32') {
    return /^avcodec.+\.dll$/i.test(name) || /^libdav1d.*\.dll$/i.test(name) || /^dav1d\.dll$/i.test(name);
  }
  if (process.platform === 'darwin') {
    return name.startsWith('libavcodec') && name.endsWith('.dylib');
  }
  return name.startsWith('libavcodec') && name.includes('.so');
}

const entries = fs.readdirSync(vendor);
const hasAvcodec = entries.some(isAvLib);
if (!hasAvcodec) {
  console.error(
    `Missing shared libavcodec in ${vendor}. Rebuild with:\n  npm run build:ffmpeg -- --force`
  );
  process.exit(1);
}

console.log(`Found ffmpeg at ${ffmpegBin}`);
console.log(`Found ffprobe at ${ffprobeBin}`);
console.log(`Found shared libavcodec in ${vendor}`);

if (process.platform === 'linux') {
  const { spawnSync } = require('child_process');
  const ldd = spawnSync('ldd', [ffmpegBin], {
    encoding: 'utf8',
    env: { ...process.env, LD_LIBRARY_PATH: vendor },
  });
  const missing = (ldd.stdout || '')
    .split('\n')
    .filter((line) => /libav|libsw|libdav1d/.test(line) && /not found/.test(line));
  if (ldd.status !== 0 || missing.length) {
    console.error(
      `Vendored ffmpeg cannot resolve libav* with LD_LIBRARY_PATH=${vendor}.\n` +
        (missing.join('\n') || ldd.stderr || 'ldd failed') +
        `\nRebuild with:\n  npm run build:ffmpeg -- --force`
    );
    process.exit(1);
  }
}
