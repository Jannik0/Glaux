'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const {
  usesCacheAwareStreaming,
  languageFlagForModel,
  extractPartialFromLine,
  extractFinalTextFromLine,
} = require('../engines/transcribecpp/asr');

describe('transcribecpp asr helpers', () => {
  it('detects Nemotron / streaming model ids', () => {
    assert.equal(
      usesCacheAwareStreaming('handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf', 'x.gguf'),
      true
    );
    assert.equal(
      usesCacheAwareStreaming('org/whisper-large-v3-turbo-gguf', 'whisper-large-v3-turbo-Q8_0.gguf'),
      false
    );
  });

  it('omits language so the CLI auto-detects', () => {
    assert.equal(
      languageFlagForModel('handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf', ''),
      null
    );
    assert.equal(
      languageFlagForModel('handy-computer/nemotron-speech-streaming-en-0.6b-gguf', ''),
      null
    );
    assert.equal(languageFlagForModel('org/whisper-large-v3', ''), null);
  });

  it('parses streaming partial and final text lines', () => {
    assert.equal(
      extractPartialFromLine('  feed[ 0]: input=1120 ms buffered=1120 ms  partial="hello"'),
      'hello'
    );
    assert.equal(extractFinalTextFromLine('text: hello world'), 'hello world');
    assert.equal(extractFinalTextFromLine('text: (empty)'), '');
  });
});
