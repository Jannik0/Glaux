'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { withFfmpegEnv, ffmpegBinName, ffprobeBinName, pickFfmpeg, pickFfprobe } = require('../engines/common/ffmpeg');
const { VIDEO_EXTS } = require('../engines/common/mediaKinds');

describe('withFfmpegEnv', () => {
  it('prepends an explicit ffmpeg dir and sets GLAUX_FFMPEG / GLAUX_FFPROBE', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-ffmpeg-'));
    try {
      fs.writeFileSync(path.join(dir, ffmpegBinName()), '');
      fs.writeFileSync(path.join(dir, ffprobeBinName()), '');
      const env = withFfmpegEnv({ PATH: 'rest' }, dir);
      assert.ok(env.PATH.startsWith(path.resolve(dir)));
      assert.match(env.PATH, /rest/);
      if (process.platform === 'linux') {
        assert.ok(env.LD_LIBRARY_PATH.startsWith(path.resolve(dir)));
      } else if (process.platform === 'darwin') {
        assert.ok(env.DYLD_LIBRARY_PATH.startsWith(path.resolve(dir)));
      }
      assert.equal(env.GLAUX_FFMPEG, path.join(path.resolve(dir), ffmpegBinName()));
      assert.equal(env.GLAUX_FFPROBE, path.join(path.resolve(dir), ffprobeBinName()));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('does not throw when ffmpeg is not staged and no explicit dir is given', () => {
    const env = withFfmpegEnv({ PATH: 'rest', GLAUX_FORCE_CPU: '0' });
    assert.ok(env.PATH.includes('rest'));
  });
});

describe('VIDEO_EXTS', () => {
  it('accepts MPEG-TS camcorder and broadcast extensions', () => {
    assert.equal(VIDEO_EXTS.has('mts'), true);
    assert.equal(VIDEO_EXTS.has('m2ts'), true);
    assert.equal(VIDEO_EXTS.has('ts'), true);
  });
});

describe('pickFfmpeg', () => {
  it('only resolves binaries under vendor/ffmpeg', () => {
    const ffmpeg = pickFfmpeg();
    if (ffmpeg) {
      assert.match(ffmpeg.replace(/\\/g, '/'), /\/ffmpeg\/ffmpeg(\.exe)?$/i);
      assert.doesNotMatch(ffmpeg, /llamacpp|transcribe/);
    }
    const ffprobe = pickFfprobe();
    if (ffprobe) {
      assert.match(ffprobe.replace(/\\/g, '/'), /\/ffmpeg\/ffprobe(\.exe)?$/i);
      assert.doesNotMatch(ffprobe, /llamacpp|transcribe/);
    }
  });
});
