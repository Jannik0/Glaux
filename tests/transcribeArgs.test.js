'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { buildTranscribeCliArgs } = require('../engines/transcribecpp/asr');

function argValue(args, flag) {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

describe('buildTranscribeCliArgs', () => {
  it('uses --backend auto by default', () => {
    const args = buildTranscribeCliArgs({
      modelPath: '/m.gguf',
      wavPath: '/clip.wav',
      forceCpu: false,
    });
    assert.equal(argValue(args, '--backend'), 'auto');
    assert.equal(args[args.length - 1], '/clip.wav');
  });

  it('uses --backend cpu when forced', () => {
    const args = buildTranscribeCliArgs({
      modelPath: '/m.gguf',
      wavPath: '/clip.wav',
      forceCpu: true,
    });
    assert.equal(argValue(args, '--backend'), 'cpu');
  });

  it('adds streaming flags when requested', () => {
    const args = buildTranscribeCliArgs({
      modelPath: '/m.gguf',
      wavPath: '/clip.wav',
      streaming: true,
      forceCpu: false,
    });
    assert.equal(argValue(args, '--stream-chunk-ms'), '1120');
    assert.equal(argValue(args, '--stream-att-right'), '13');
  });
});
