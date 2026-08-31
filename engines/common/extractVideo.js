'use strict';

/**
 * Extract a video's audio track into a sibling mono PCM s16 WAV via vendor ffmpeg.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const fsp = require('fs').promises;
const path = require('path');
const { pickFfmpeg } = require('./ffmpeg');
const { VIDEO_EXTS, extOf } = require('./mediaKinds');

/**
 * @param {string} videoPath Absolute path to a video file.
 * @returns {Promise<string>} Absolute path to the sibling `.wav` file.
 */
async function extractVideoToWav(videoPath) {
  const abs = path.resolve(videoPath);
  try {
    await fsp.access(abs);
  } catch {
    throw new Error(`Video file not found: ${videoPath}`);
  }

  const ext = extOf(abs);
  if (!VIDEO_EXTS.has(ext)) {
    throw new Error(`Not a supported video file: ${videoPath}`);
  }

  const wavPath = abs.replace(/\.[^.\\/]+$/i, '.wav');
  try {
    await fsp.access(wavPath);
    return wavPath;
  } catch {
    /* need to extract */
  }

  const ffmpeg = pickFfmpeg();
  if (!ffmpeg) {
    throw new Error(
      'ffmpeg not found (needed to extract audio from video). ' +
        'Run npm run build:ffmpeg.'
    );
  }

  await new Promise((resolve, reject) => {
    // Match former PyAV behavior: mono PCM s16 at the source sample rate (no -ar).
    const args = ['-y', '-i', abs, '-vn', '-ac', '1', '-c:a', 'pcm_s16le', wavPath];
    const child = spawn(ffmpeg, args, { windowsHide: true });
    let err = '';
    child.stderr.on('data', (d) => {
      err += d.toString('utf8');
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0 && fs.existsSync(wavPath)) {
        resolve();
      } else {
        reject(new Error(err.trim() || `ffmpeg failed to extract audio (${code})`));
      }
    });
  });

  try {
    await fsp.access(wavPath);
  } catch {
    throw new Error('ffmpeg did not produce a wav file (video may have no audio track).');
  }
  return wavPath;
}

module.exports = {
  extractVideoToWav,
};
