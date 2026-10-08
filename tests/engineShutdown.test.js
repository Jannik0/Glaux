'use strict';

const { describe, it, after } = require('node:test');
const assert = require('node:assert/strict');
const hf = require('../engines/huggingface/engine');
const llama = require('../engines/llamacpp/engine');
const transcribe = require('../engines/transcribecpp/engine');
const stableDiffusion = require('../engines/stablediffusioncpp/engine');
const engineManager = require('../engines/engineManager');

const engines = [hf, llama, transcribe, stableDiffusion];
const originals = engines.map((engine) => ({
  engine,
  chatStop: engine.chatStop,
  close: engine.close,
}));
const originalDestroy = hf.chatbotDestroyIfRunning;

function restore() {
  hf.chatbotDestroyIfRunning = originalDestroy;
  for (const saved of originals) {
    saved.engine.chatStop = saved.chatStop;
    saved.engine.close = saved.close;
  }
}

after(restore);

describe('engine shutdown', () => {
  it('unloads the Hugging Face pipeline before killing workers', async () => {
    const order = [];
    hf.chatbotDestroyIfRunning = async () => {
      order.push('destroy');
    };
    for (const engine of engines) {
      const label = engine === hf ? 'hf' : 'other';
      engine.chatStop = async () => {
        order.push(`${label}-stop`);
      };
      engine.close = async () => {
        order.push(`${label}-close`);
      };
    }

    await engineManager.shutdown();

    const destroyAt = order.indexOf('destroy');
    const firstClose = order.findIndex((entry) => entry.endsWith('-close'));
    assert.ok(order.includes('hf-stop'));
    assert.ok(destroyAt !== -1);
    assert.ok(firstClose > destroyAt);
  });

  it('unloads the Hugging Face pipeline before resetting the worker', async () => {
    const order = [];
    hf.chatbotDestroyIfRunning = async () => {
      order.push('destroy');
    };
    for (const engine of engines) {
      engine.chatStop = async () => {};
      engine.close = async () => {
        order.push('close');
      };
    }

    await engineManager.resetInferenceWorker();

    assert.equal(order[0], 'destroy');
    assert.ok(order.includes('close'));
  });

  it('does not start a Hugging Face worker when none is running', async () => {
    restore();
    await hf.chatbotDestroyIfRunning();
  });
});
