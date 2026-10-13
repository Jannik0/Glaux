'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { resolveEngineId } = require('../engines/common/resolveEngineId');

describe('resolveEngineId', () => {
  it('routes safetensors to huggingface', () => {
    assert.equal(resolveEngineId('huggingface', null), 'huggingface');
    assert.equal(
      resolveEngineId('huggingface', 'automatic-speech-recognition'),
      'huggingface'
    );
    assert.equal(resolveEngineId('huggingface', 'text-to-image'), 'huggingface');
  });

  it('routes ASR GGUF to transcribecpp', () => {
    assert.equal(
      resolveEngineId('llamacpp', 'automatic-speech-recognition'),
      'transcribecpp'
    );
  });

  it('routes chat GGUF to llamacpp', () => {
    assert.equal(resolveEngineId('llamacpp', 'text-generation'), 'llamacpp');
    assert.equal(resolveEngineId('llamacpp', null), 'llamacpp');
  });

  it('routes text-to-image GGUF to stablediffusioncpp', () => {
    assert.equal(resolveEngineId('llamacpp', 'text-to-image'), 'stablediffusioncpp');
  });

  it('routes image-to-image GGUF to stablediffusioncpp', () => {
    assert.equal(resolveEngineId('huggingface', 'image-to-image'), 'huggingface');
    assert.equal(resolveEngineId('llamacpp', 'image-to-image'), 'stablediffusioncpp');
  });

  it('returns null for unknown format', () => {
    assert.equal(resolveEngineId(null, null), null);
  });
});
