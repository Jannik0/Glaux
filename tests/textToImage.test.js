'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { assertTextToImagePrompt } = require('../engines/common/textToImage');

describe('assertTextToImagePrompt', () => {
  it('returns the trimmed prompt', () => {
    assert.equal(assertTextToImagePrompt('  a red fox  ', []), 'a red fox');
    assert.equal(assertTextToImagePrompt('a red fox', undefined), 'a red fox');
  });

  it('rejects attachments', () => {
    assert.throws(
      () => assertTextToImagePrompt('a red fox', [{ source: 'resources', relativePath: 'a.png' }]),
      /text prompt only/
    );
  });

  it('rejects a blank prompt', () => {
    assert.throws(() => assertTextToImagePrompt('   ', []), /requires a prompt/);
    assert.throws(() => assertTextToImagePrompt('', []), /requires a prompt/);
  });
});
