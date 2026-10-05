'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { buildSdCliArgs } = require('../engines/stablediffusion/generate');

describe('buildSdCliArgs', () => {
  const base = {
    modelPath: '/models/sd.gguf',
    prompt: 'a red fox',
    outputPath: '/sessions/chat-1.png',
  };

  it('passes the prompt and output and leaves sampling flags off', () => {
    const args = buildSdCliArgs({ ...base, forceCpu: false });
    assert.deepEqual(args, [
      '-m',
      base.modelPath,
      '--mode',
      'img_gen',
      '-p',
      base.prompt,
      '-o',
      base.outputPath,
    ]);
    for (const flag of ['--steps', '--cfg-scale', '--width', '--height', '--seed', '--init-img', '--batch-count']) {
      assert.equal(args.includes(flag), false, flag);
    }
  });

  it('pins CPU only when forceCpu is set', () => {
    const cpu = buildSdCliArgs({ ...base, forceCpu: true });
    assert.equal(cpu.at(-2), '--backend');
    assert.equal(cpu.at(-1), 'cpu');
    const gpu = buildSdCliArgs({ ...base, forceCpu: false });
    assert.equal(gpu.includes('--backend'), false);
  });
});
