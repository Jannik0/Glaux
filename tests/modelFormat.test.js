'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const {
  extractQuantKey,
  variantGroupKey,
  isGgufName,
  isMmprojName,
  classifyHubRepoFiles,
  buildGgufAllowPatterns,
} = require('../engines/common/modelFormat');

describe('modelFormat', () => {
  it('detects GGUF and mmproj names', () => {
    assert.equal(isGgufName('model-Q4_K_M.gguf'), true);
    assert.equal(isGgufName('model.safetensors'), false);
    assert.equal(isMmprojName('mmproj-f16.gguf'), true);
    assert.equal(isMmprojName('model-Q4_K_M.gguf'), false);
  });

  it('extracts quant keys from filenames', () => {
    assert.equal(extractQuantKey('org-model-Q4_K_M.gguf'), 'Q4_K_M');
    assert.equal(extractQuantKey('model-Q8_0-00001-of-00002.gguf'), 'Q8_0');
    assert.equal(extractQuantKey('model-F16.gguf'), 'F16');
  });

  it('groups variants by quant key', () => {
    assert.equal(variantGroupKey('a-Q4_K_M.gguf'), 'Q4_K_M');
    assert.equal(
      variantGroupKey('a-Q4_K_M-00001-of-00002.gguf'),
      'Q4_K_M'
    );
  });

  it('classifies safetensors repos as huggingface', () => {
    const classified = classifyHubRepoFiles([
      { path: 'model.safetensors', size: 100 },
      { path: 'README.md', size: 1 },
    ]);
    assert.equal(classified.kind, 'huggingface');
    assert.deepEqual(classified.variants, []);
  });

  it('classifies GGUF repos into variants', () => {
    const classified = classifyHubRepoFiles([
      { path: 'model-Q4_K_M.gguf', size: 50 },
      { path: 'model-Q8_0.gguf', size: 80 },
      { path: 'mmproj-f16.gguf', size: 10 },
      { path: 'README.md', size: 1 },
    ]);
    assert.equal(classified.kind, 'gguf');
    assert.ok(Array.isArray(classified.variants));
    assert.ok(classified.variants.length >= 2);
  });

  it('builds GGUF allow patterns including mmproj and README', () => {
    const allFiles = [
      'model-Q4_K_M.gguf',
      'model-Q8_0.gguf',
      'mmproj-f16.gguf',
      'README.md',
      'tokenizer.json',
    ];
    const variant = { key: 'Q4_K_M', files: ['model-Q4_K_M.gguf'], size: 50 };
    const patterns = buildGgufAllowPatterns(allFiles, variant);
    assert.ok(patterns.includes('model-Q4_K_M.gguf'));
    assert.ok(patterns.includes('mmproj-f16.gguf'));
    assert.ok(patterns.includes('README.md'));
    assert.ok(patterns.includes('tokenizer.json'));
    assert.ok(!patterns.includes('model-Q8_0.gguf'));
  });
});
