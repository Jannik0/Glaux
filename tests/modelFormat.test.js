'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const {
  extractQuantKey,
  variantGroupKey,
  isGgufName,
  isMmprojName,
  classifyHubRepoFiles,
  buildGgufAllowPatterns,
  directoryContainsModelWeights,
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

  it('builds GGUF allow patterns including mmproj, README, and companion weights', () => {
    const allFiles = [
      'model-Q4_K_M.gguf',
      'model-Q8_0.gguf',
      'mmproj-f16.gguf',
      'README.md',
      'tokenizer.json',
      'vae/diffusion_pytorch_model.safetensors',
      'text_encoder/model.safetensors',
      'transformer/diffusion_pytorch_model.safetensors',
    ];
    const variant = { key: 'Q4_K_M', files: ['model-Q4_K_M.gguf'], size: 50 };
    const patterns = buildGgufAllowPatterns(allFiles, variant);
    assert.ok(patterns.includes('model-Q4_K_M.gguf'));
    assert.ok(patterns.includes('mmproj-f16.gguf'));
    assert.ok(patterns.includes('README.md'));
    assert.ok(patterns.includes('tokenizer.json'));
    assert.ok(patterns.includes('vae/diffusion_pytorch_model.safetensors'));
    assert.ok(patterns.includes('text_encoder/model.safetensors'));
    assert.ok(!patterns.includes('model-Q8_0.gguf'));
    assert.ok(!patterns.includes('transformer/diffusion_pytorch_model.safetensors'));
  });

  it('keeps a GGUF repo that also ships a VAE on the GGUF path', () => {
    const classified = classifyHubRepoFiles([
      { path: 'model-Q4_K_M.gguf', size: 50 },
      { path: 'vae/diffusion_pytorch_model.safetensors', size: 10 },
      { path: 'README.md', size: 1 },
    ]);
    assert.equal(classified.kind, 'gguf');
    assert.ok(classified.variants.length >= 1);
  });

  it('treats a VAE and text encoder folder as weights only when a model file is present', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-companions-'));
    const base = path.join(root, 'org', 'image');
    fs.mkdirSync(path.join(base, 'vae'), { recursive: true });
    fs.mkdirSync(path.join(base, 'text_encoder'), { recursive: true });
    fs.writeFileSync(path.join(base, 'vae', 'config.json'), '{}');
    fs.writeFileSync(path.join(base, 'vae', 'diffusion_pytorch_model.safetensors'), '');
    fs.writeFileSync(path.join(base, 'text_encoder', 'config.json'), '{}');
    fs.writeFileSync(path.join(base, 'text_encoder', 'model.safetensors'), '');
    assert.equal(await directoryContainsModelWeights(base), false);

    fs.mkdirSync(path.join(base, 'transformer'), { recursive: true });
    fs.writeFileSync(path.join(base, 'transformer', 'diffusion_pytorch_model.safetensors'), '');
    assert.equal(await directoryContainsModelWeights(base), true);
  });

  it('keeps a diffusers repo on the safetensors path when model_index.json is present', () => {
    const classified = classifyHubRepoFiles([
      { path: 'model_index.json', size: 1 },
      { path: 'transformer/diffusion_pytorch_model.safetensors', size: 100 },
      { path: 'extra-Q4_K_M.gguf', size: 10 },
    ]);
    assert.equal(classified.kind, 'huggingface');
    assert.deepEqual(classified.variants, []);
  });
});
