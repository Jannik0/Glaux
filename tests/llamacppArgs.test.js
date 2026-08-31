'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { buildLlamaServerArgs, parseLlamaCtxSize } = require('../engines/llamacpp/engine');

function argValue(args, flag) {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

describe('parseLlamaCtxSize', () => {
  it('treats unset, 0, and invalid values as omit -c', () => {
    assert.equal(parseLlamaCtxSize(undefined), null);
    assert.equal(parseLlamaCtxSize(''), null);
    assert.equal(parseLlamaCtxSize('0'), null);
    assert.equal(parseLlamaCtxSize('-1'), null);
    assert.equal(parseLlamaCtxSize('nope'), null);
  });

  it('accepts a positive explicit window', () => {
    assert.equal(parseLlamaCtxSize('8192'), 8192);
    assert.equal(parseLlamaCtxSize('4096.9'), 4096);
  });
});

describe('buildLlamaServerArgs', () => {
  const base = {
    modelPath: '/models/foo.gguf',
    port: 8080,
    mediaPathRoot: '/data',
  };

  it('omits CPU pins so llama.cpp auto-selects GPU', () => {
    const args = buildLlamaServerArgs({ ...base, forceCpu: false });
    assert.equal(argValue(args, '--device'), undefined);
    assert.equal(args.includes('-ngl'), false);
    assert.equal(args.includes('--no-mmproj-offload'), false);
    assert.equal(argValue(args, '-m'), base.modelPath);
    assert.equal(argValue(args, '--port'), '8080');
  });

  it('omits -c by default so --fit can shrink context', () => {
    const args = buildLlamaServerArgs({ ...base, forceCpu: false, ctxSize: 0 });
    assert.equal(args.includes('-c'), false);
    assert.equal(argValue(args, '--ctx-size'), undefined);
  });

  it('passes -c when an explicit window is set', () => {
    const args = buildLlamaServerArgs({ ...base, forceCpu: false, ctxSize: 8192 });
    assert.equal(argValue(args, '-c'), '8192');
  });

  it('forces CPU when requested', () => {
    const args = buildLlamaServerArgs({ ...base, forceCpu: true });
    assert.equal(argValue(args, '--device'), 'none');
    assert.equal(argValue(args, '-ngl'), '0');
    assert.equal(args.includes('--no-mmproj-offload'), true);
  });

  it('appends mmproj when provided', () => {
    const args = buildLlamaServerArgs({
      ...base,
      mmprojPath: '/models/mmproj.gguf',
      forceCpu: false,
    });
    assert.equal(argValue(args, '--mmproj'), '/models/mmproj.gguf');
  });
});
