'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { pipelineTagFromCard } = require('../engines/common/pipelineTag');

describe('pipelineTagFromCard', () => {
  it('uses pipeline_tag when the card declares one', () => {
    const card = [
      '---',
      'pipeline_tag: text-generation',
      'tags:',
      '- text-to-image',
      '- automatic-speech-recognition',
      '---',
      '',
    ].join('\n');
    assert.equal(pipelineTagFromCard(card), 'text-generation');
  });

  it('reads image-to-image from tags when pipeline_tag is absent', () => {
    const card = ['---', 'tags:', '- diffusers', '- image-to-image', '---', ''].join('\n');
    assert.equal(pipelineTagFromCard(card), 'image-to-image');
  });

  it('reads text-to-image from tags when pipeline_tag is absent', () => {
    const card = ['---', 'tags:', '- gguf', '- text-to-image', '---', ''].join('\n');
    assert.equal(pipelineTagFromCard(card), 'text-to-image');
  });

  it('reads automatic-speech-recognition from tags when pipeline_tag is absent', () => {
    const card = ['---', 'tags:', '- automatic-speech-recognition', '- whisper', '---', ''].join('\n');
    assert.equal(pipelineTagFromCard(card), 'automatic-speech-recognition');
  });

  it('uses the first task tag when the list contains both', () => {
    const asrFirst = ['---', 'tags:', '- automatic-speech-recognition', '- text-to-image', '---', ''].join('\n');
    const imageFirst = ['---', 'tags: [text-to-image, automatic-speech-recognition]', '---', ''].join('\n');
    assert.equal(pipelineTagFromCard(asrFirst), 'automatic-speech-recognition');
    assert.equal(pipelineTagFromCard(imageFirst), 'text-to-image');
  });

  it('reads a single tag written on the tags line', () => {
    const card = "---\ntags: 'text-to-image'\n---\n";
    assert.equal(pipelineTagFromCard(card), 'text-to-image');
  });

  it('falls through when pipeline_tag is empty', () => {
    const card = ['---', 'pipeline_tag:', 'tags:', '- text-to-image', '---', ''].join('\n');
    assert.equal(pipelineTagFromCard(card), 'text-to-image');
  });

  it('returns null when neither field names a routed task', () => {
    const card = ['---', 'tags:', '- gguf', '- image-classification', '---', ''].join('\n');
    assert.equal(pipelineTagFromCard(card), null);
  });
});
