'use strict';

/**
 * Locate the vendored ffmpeg/ffprobe tree (video probe/decode, ASR WAV prep).
 * Only vendor/ffmpeg (packaged as resources/ffmpeg) is used.
 */

const fs = require('fs');
const path = require('path');
const { findVendorBinary, getVendorRoot } = require('./runtimePaths');
const { withVendorLibPath } = require('./gpuRuntime');

function ffmpegBinName() {
  return process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg';
}

function ffprobeBinName() {
  return process.platform === 'win32' ? 'ffprobe.exe' : 'ffprobe';
}

/**
 * @returns {string} Absolute vendor/ffmpeg directory (may not exist yet).
 */
function ffmpegDir() {
  return getVendorRoot('ffmpeg');
}

/**
 * @returns {string | null} Absolute path to ffmpeg, or null if not found.
 */
function pickFfmpeg() {
  return findVendorBinary('ffmpeg', 'ffmpeg.exe', 'ffmpeg');
}

/**
 * @returns {string | null} Absolute path to ffprobe, or null if not found.
 */
function pickFfprobe() {
  return findVendorBinary('ffmpeg', 'ffprobe.exe', 'ffprobe');
}

/**
 * Prepend vendor/ffmpeg to PATH / loader path and set GLAUX_FFMPEG /
 * GLAUX_FFPROBE so child processes (llama-server, Python worker) find the
 * same binaries. No-ops when ffmpeg is not staged.
 *
 * @param {NodeJS.ProcessEnv} [baseEnv]
 * @param {string} [explicitDir]
 * @returns {NodeJS.ProcessEnv}
 */
function withFfmpegEnv(baseEnv = process.env, explicitDir) {
  const ffmpeg = pickFfmpeg();
  const dir = explicitDir ? path.resolve(explicitDir) : ffmpeg ? path.dirname(ffmpeg) : null;
  if (!dir) {
    return { ...baseEnv };
  }
  const env = withVendorLibPath(baseEnv, dir);
  const ffmpegPath = path.join(dir, ffmpegBinName());
  const ffprobePath = path.join(dir, ffprobeBinName());
  if (fs.existsSync(ffmpegPath)) {
    env.GLAUX_FFMPEG = ffmpegPath;
  }
  if (fs.existsSync(ffprobePath)) {
    env.GLAUX_FFPROBE = ffprobePath;
  }
  return env;
}

module.exports = {
  ffmpegBinName,
  ffprobeBinName,
  ffmpegDir,
  pickFfmpeg,
  pickFfprobe,
  withFfmpegEnv,
};
