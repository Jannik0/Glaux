'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { assertImageToImagePrompt, assertTextToImagePrompt } = require('../engines/common/textToImage');

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

describe('assertImageToImagePrompt', () => {
  it('accepts a prompt with no image', () => {
    const result = assertImageToImagePrompt('  a red fox  ', []);
    assert.equal(result.prompt, 'a red fox');
    assert.equal(result.imageFile, null);
  });

  it('accepts one image', () => {
    const file = { source: 'resources', relativePath: 'photo.png' };
    const result = assertImageToImagePrompt('repaint this', [file]);
    assert.equal(result.prompt, 'repaint this');
    assert.equal(result.imageFile, file);
  });

  it('rejects a second image and non-image files', () => {
    assert.throws(
      () =>
        assertImageToImagePrompt('a', [
          { relativePath: 'a.png' },
          { relativePath: 'b.jpg' },
        ]),
      /at most one image/
    );
    assert.throws(
      () => assertImageToImagePrompt('a', [{ relativePath: 'note.wav' }]),
      /optional image/
    );
  });

  it('rejects a blank prompt', () => {
    assert.throws(() => assertImageToImagePrompt('  ', [{ relativePath: 'a.png' }]), /requires a prompt/);
  });
});
