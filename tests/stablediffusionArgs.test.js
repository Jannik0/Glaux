'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { buildSdCliArgs, formatSdCliError } = require('../engines/stablediffusion/generate');
const {
  findLocalComponents,
  readBaseModelId,
  missingBaseCompanions,
  resolveRunComponents,
} = require('../engines/stablediffusion/weights');

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

  it('passes whichever companion weights the model directory has', () => {
    const args = buildSdCliArgs({
      ...base,
      components: {
        vae: '/models/vae.safetensors',
        clipL: '/models/clip_l.safetensors',
        t5xxl: '/models/t5xxl.safetensors',
        llm: '/models/text_encoder',
      },
      forceCpu: false,
    });
    assert.deepEqual(args.slice(0, 10), [
      '--diffusion-model',
      base.modelPath,
      '--vae',
      '/models/vae.safetensors',
      '--clip_l',
      '/models/clip_l.safetensors',
      '--t5xxl',
      '/models/t5xxl.safetensors',
      '--llm',
      '/models/text_encoder',
    ]);
    assert.equal(args.includes('-m'), false);
  });

  it('loads a standalone diffusion file with --diffusion-model', () => {
    const args = buildSdCliArgs({ ...base, diffusionOnly: true, forceCpu: false });
    assert.equal(args[0], '--diffusion-model');
    assert.equal(args[1], base.modelPath);
    assert.equal(args.includes('-m'), false);
  });

  it('passes one init image and leaves strength at the CLI default', () => {
    const args = buildSdCliArgs({
      ...base,
      initImage: '/sessions/source.png',
      forceCpu: false,
    });
    assert.equal(args.includes('--init-img'), true);
    assert.equal(args[args.indexOf('--init-img') + 1], '/sessions/source.png');
    assert.equal(args.includes('--strength'), false);
  });

  it('pins CPU only when forceCpu is set', () => {
    const cpu = buildSdCliArgs({ ...base, forceCpu: true });
    assert.equal(cpu.at(-2), '--backend');
    assert.equal(cpu.at(-1), 'cpu');
    const gpu = buildSdCliArgs({ ...base, forceCpu: false });
    assert.equal(gpu.includes('--backend'), false);
  });
});

describe('formatSdCliError', () => {
  it('keeps the validation failure and drops CUDA init logs', () => {
    const stderr = [
      'ggml_cuda_init: found 1 CUDA devices',
      'load_backend: loaded CUDA backend',
      '[ERROR  ] model_manager.cpp:757  - VAE tensor \'x\' not in model metadata',
      '[ERROR  ] diffusion_engine.cpp:1270 - model metadata validation failed',
    ].join('\n');
    assert.equal(formatSdCliError(stderr, '', 1), 'model metadata validation failed');
  });
});

describe('readBaseModelId', () => {
  it('reads a scalar or the first list entry from the model card', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-card-'));
    fs.writeFileSync(
      path.join(root, 'README.md'),
      '---\nbase_model:\n- black-forest-labs/FLUX.2-klein-4B\ntags:\n- gguf\n---\n'
    );
    assert.equal(readBaseModelId(root), 'black-forest-labs/FLUX.2-klein-4B');
    fs.writeFileSync(
      path.join(root, 'README.md'),
      '---\nbase_model: org/name\npipeline_tag: text-to-image\n---\n'
    );
    assert.equal(readBaseModelId(root), 'org/name');
  });
});

describe('resolveRunComponents', () => {
  it('uses the base repo folder named on the model card', () => {
    const cache = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-cache-'));
    const modelRoot = path.join(cache, 'leejet', 'image-gguf');
    const baseRoot = path.join(cache, 'org', 'image');
    fs.mkdirSync(path.join(modelRoot, 'vae'), { recursive: true });
    fs.writeFileSync(path.join(modelRoot, 'vae', 'diffusion_pytorch_model.safetensors'), '');
    fs.writeFileSync(
      path.join(modelRoot, 'README.md'),
      '---\nbase_model:\n- org/image\npipeline_tag: text-to-image\n---\n'
    );
    fs.mkdirSync(path.join(baseRoot, 'vae'), { recursive: true });
    fs.mkdirSync(path.join(baseRoot, 'text_encoder'), { recursive: true });
    fs.writeFileSync(path.join(baseRoot, 'vae', 'diffusion_pytorch_model.safetensors'), '');
    fs.writeFileSync(path.join(baseRoot, 'text_encoder', 'model.safetensors'), '');
    fs.writeFileSync(
      path.join(baseRoot, 'text_encoder', 'config.json'),
      JSON.stringify({ architectures: ['Qwen3ForCausalLM'], model_type: 'qwen3' })
    );

    const found = resolveRunComponents(modelRoot, cache);
    assert.equal(found.vae, path.join(baseRoot, 'vae', 'diffusion_pytorch_model.safetensors'));
    assert.equal(found.llm, path.join(baseRoot, 'text_encoder', 'model.safetensors'));
    assert.notEqual(found.vae, path.join(modelRoot, 'vae', 'diffusion_pytorch_model.safetensors'));
    assert.equal(missingBaseCompanions(modelRoot, cache), null);
  });

  it('names the base repo when its VAE and text encoder are gone', () => {
    const cache = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-cache-'));
    const modelRoot = path.join(cache, 'leejet', 'image-gguf');
    fs.mkdirSync(path.join(modelRoot, 'vae'), { recursive: true });
    fs.writeFileSync(path.join(modelRoot, 'vae', 'diffusion_pytorch_model.safetensors'), '');
    fs.writeFileSync(path.join(modelRoot, 'README.md'), '---\nbase_model: org/image\n---\n');
    assert.equal(missingBaseCompanions(modelRoot, cache), 'org/image');
    assert.throws(
      () => resolveRunComponents(modelRoot, cache),
      /VAE and text encoder from org\/image/
    );
  });

  it('uses files in the GGUF directory when the card names no base repo', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-local-'));
    fs.mkdirSync(path.join(root, 'vae'), { recursive: true });
    fs.writeFileSync(path.join(root, 'vae', 'diffusion_pytorch_model.safetensors'), '');
    fs.writeFileSync(path.join(root, 'llm.gguf'), '');
    const found = resolveRunComponents(root, path.join(root, 'cache'));
    assert.equal(found.vae, path.join(root, 'vae', 'diffusion_pytorch_model.safetensors'));
    assert.equal(found.llm, path.join(root, 'llm.gguf'));
    assert.equal(missingBaseCompanions(root, path.join(root, 'cache')), null);
  });
});

describe('findLocalComponents', () => {
  it('reads companion weights from one model directory', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-'));
    const modelRoot = path.join(root, 'org', 'image-gguf');
    const otherRoot = path.join(root, 'org', 'image');
    fs.mkdirSync(path.join(modelRoot, 'vae'), { recursive: true });
    fs.mkdirSync(path.join(modelRoot, 'text_encoder'), { recursive: true });
    fs.mkdirSync(path.join(modelRoot, 'text_encoder_2'), { recursive: true });
    fs.writeFileSync(path.join(modelRoot, 'vae', 'diffusion_pytorch_model.safetensors'), '');
    fs.writeFileSync(path.join(modelRoot, 'text_encoder', 'model.safetensors.index.json'), '{}');
    fs.writeFileSync(
      path.join(modelRoot, 'text_encoder', 'config.json'),
      JSON.stringify({ architectures: ['CLIPTextModel'], model_type: 'clip' })
    );
    fs.writeFileSync(path.join(modelRoot, 'text_encoder', 'model.safetensors'), '');
    fs.writeFileSync(
      path.join(modelRoot, 'text_encoder_2', 'config.json'),
      JSON.stringify({ architectures: ['T5EncoderModel'], model_type: 't5' })
    );
    fs.writeFileSync(path.join(modelRoot, 'text_encoder_2', 'model.safetensors'), '');
    fs.writeFileSync(path.join(modelRoot, 'clip_g.safetensors'), '');
    fs.mkdirSync(path.join(otherRoot, 'vae'), { recursive: true });
    fs.writeFileSync(path.join(otherRoot, 'vae', 'diffusion_pytorch_model.safetensors'), '');
    fs.writeFileSync(
      path.join(modelRoot, 'README.md'),
      '---\nbase_model: org/image\npipeline_tag: text-to-image\n---\n'
    );

    const found = findLocalComponents(modelRoot);
    assert.equal(found.vae, path.join(modelRoot, 'vae', 'diffusion_pytorch_model.safetensors'));
    assert.equal(found.clipL, path.join(modelRoot, 'text_encoder'));
    assert.equal(found.clipG, path.join(modelRoot, 'clip_g.safetensors'));
    assert.equal(found.t5xxl, path.join(modelRoot, 'text_encoder_2', 'model.safetensors'));
    assert.equal(found.llm, null);
    assert.notEqual(found.vae, path.join(otherRoot, 'vae', 'diffusion_pytorch_model.safetensors'));
  });
});
