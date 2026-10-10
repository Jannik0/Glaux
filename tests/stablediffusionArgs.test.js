'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const {
  buildSdCliArgs,
  fitInitCanvas,
  formatSdCliError,
  imageSizeFromBuffer,
} = require('../engines/stablediffusioncpp/generate');
const {
  componentPath,
  findLocalComponents,
  readBaseModelId,
  missingBaseCompanions,
  resolveRunComponents,
} = require('../engines/stablediffusioncpp/weights');

describe('buildSdCliArgs', () => {
  const base = {
    modelPath: '/models/sd.gguf',
    prompt: 'a red fox',
    outputPath: '/sessions/chat-1.png',
  };

  it('passes a random seed and leaves the other sampling flags off', () => {
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
      '--seed',
      '-1',
    ]);
    for (const flag of ['--steps', '--cfg-scale', '--width', '--height', '--init-img', '--vae-tiling', '--batch-count']) {
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

  it('fits an init image to the prompt-only canvas and leaves strength at the CLI default', () => {
    const unknown = buildSdCliArgs({
      ...base,
      initImage: '/sessions/source.png',
      forceCpu: false,
    });
    assert.equal(unknown[unknown.indexOf('--init-img') + 1], '/sessions/source.png');
    assert.equal(unknown[unknown.indexOf('--width') + 1], '512');
    assert.equal(unknown[unknown.indexOf('--height') + 1], '512');
    assert.equal(unknown.includes('--vae-tiling'), true);
    assert.equal(unknown.includes('--strength'), false);

    const photo = buildSdCliArgs({
      ...base,
      initImage: '/sessions/source.png',
      width: 1920,
      height: 1080,
      forceCpu: false,
    });
    assert.equal(photo[photo.indexOf('--width') + 1], '512');
    assert.equal(photo[photo.indexOf('--height') + 1], '256');
    assert.equal(photo.includes('--ref-image'), false);
  });

  it('passes a text-to-image attachment as a reference image', () => {
    const args = buildSdCliArgs({
      ...base,
      referenceImage: '/sessions/source.png',
      width: 1920,
      height: 1080,
      forceCpu: false,
    });
    assert.equal(args[args.indexOf('--ref-image') + 1], '/sessions/source.png');
    assert.equal(args.includes('--init-img'), false);
    assert.equal(args.includes('--strength'), false);
    assert.equal(args[args.indexOf('--width') + 1], '512');
    assert.equal(args[args.indexOf('--height') + 1], '256');
    assert.equal(args.includes('--vae-tiling'), true);
  });

  it('passes a Z-Image attachment as an init image', () => {
    const named = buildSdCliArgs({
      ...base,
      modelPath: '/models/leejet/Z-Image-Turbo-GGUF/z-image-turbo-Q4_K.gguf',
      referenceImage: '/sessions/source.png',
      forceCpu: false,
    });
    assert.equal(named.includes('--init-img'), true);
    assert.equal(named.includes('--ref-image'), false);
    assert.equal(named[named.indexOf('--strength') + 1], '0.4');
    assert.equal(
      named[named.indexOf('--extra-sample-args') + 1],
      'strength_as_noise_level=true,force_first_sigma=true'
    );

    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-zimage-'));
    const modelPath = path.join(root, 'weights.gguf');
    fs.writeFileSync(modelPath, Buffer.from('model.diffusion_model.cap_embedder.0.weight'));
    const scanned = buildSdCliArgs({
      ...base,
      modelPath,
      referenceImage: '/sessions/source.png',
      forceCpu: false,
    });
    assert.equal(scanned[scanned.indexOf('--init-img') + 1], '/sessions/source.png');
    assert.equal(scanned.includes('--ref-image'), false);
  });

  it('pins CPU only when forceCpu is set', () => {
    const cpu = buildSdCliArgs({ ...base, forceCpu: true });
    assert.equal(cpu.at(-2), '--backend');
    assert.equal(cpu.at(-1), 'cpu');
    const gpu = buildSdCliArgs({ ...base, forceCpu: false });
    assert.equal(gpu.includes('--backend'), false);
  });
});

describe('fitInitCanvas', () => {
  it('keeps a square default and scales a wide photo down to a 64-pixel grid', () => {
    assert.deepEqual(fitInitCanvas(512, 512), { width: 512, height: 512 });
    assert.deepEqual(fitInitCanvas(1920, 1080), { width: 512, height: 256 });
    assert.deepEqual(fitInitCanvas(400, 300), { width: 384, height: 256 });
    assert.deepEqual(fitInitCanvas(0, 1080), { width: 512, height: 512 });
  });
});

describe('imageSizeFromBuffer', () => {
  it('reads a PNG header', () => {
    const buf = Buffer.alloc(24);
    buf[0] = 0x89;
    buf.write('PNG', 1, 'ascii');
    buf.writeUInt32BE(1920, 16);
    buf.writeUInt32BE(1080, 20);
    assert.deepEqual(imageSizeFromBuffer(buf), { width: 1920, height: 1080 });
  });

  it('reads a JPEG frame header', () => {
    const buf = Buffer.from([
      0xff, 0xd8,
      0xff, 0xe0, 0x00, 0x04, 0x00, 0x00,
      0xff, 0xc0, 0x00, 0x0b, 0x08,
      0x04, 0x38,
      0x07, 0x80,
    ]);
    assert.deepEqual(imageSizeFromBuffer(buf), { width: 1920, height: 1080 });
  });

  it('reads a progressive JPEG frame and swaps axes for Exif orientation 6', () => {
    const progressive = Buffer.from([
      0xff, 0xd8,
      0xff, 0xc2, 0x00, 0x0b, 0x08,
      0x00, 0x10,
      0x00, 0x20,
    ]);
    assert.deepEqual(imageSizeFromBuffer(progressive), { width: 32, height: 16 });

    const exif = Buffer.from([
      0xff, 0xd8,
      0xff, 0xe1, 0x00, 0x22,
      0x45, 0x78, 0x69, 0x66, 0x00, 0x00,
      0x49, 0x49, 0x2a, 0x00,
      0x08, 0x00, 0x00, 0x00,
      0x01, 0x00,
      0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0xff, 0xc0, 0x00, 0x0b, 0x08,
      0x00, 0x10,
      0x00, 0x20,
    ]);
    assert.deepEqual(imageSizeFromBuffer(exif), { width: 16, height: 32 });
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

  it('keeps a Z-Image assertion that sits under the CUDA init log', () => {
    const stderr = [
      'ggml_cuda_init: found 1 CUDA devices (Total VRAM: 6143 MiB)',
      'load_backend: loaded CUDA backend from ggml-cuda.dll',
      'ggml_vulkan: Found 2 Vulkan devices:',
      'ggml_vulkan: 0 = NVIDIA GeForce RTX 3060 Laptop GPU',
      'GGML_ASSERT(txt->ne[1] + img->ne[1] == pe->ne[3]) failed',
    ].join('\n');
    assert.equal(
      formatSdCliError(stderr, '', 3),
      'GGML_ASSERT(txt->ne[1] + img->ne[1] == pe->ne[3]) failed'
    );
  });
});

describe('componentPath', () => {
  it('returns a single weight file, a sharded directory, or null', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-component-'));
    assert.equal(componentPath(path.join(root, 'missing')), null);
    const single = path.join(root, 'vae');
    fs.mkdirSync(single);
    fs.writeFileSync(path.join(single, 'diffusion_pytorch_model.safetensors'), '');
    assert.equal(componentPath(single), path.join(single, 'diffusion_pytorch_model.safetensors'));
    const sharded = path.join(root, 'text_encoder');
    fs.mkdirSync(sharded);
    fs.writeFileSync(path.join(sharded, 'model-00001-of-00002.safetensors'), '');
    fs.writeFileSync(path.join(sharded, 'model-00002-of-00002.safetensors'), '');
    fs.writeFileSync(path.join(sharded, 'model.safetensors.index.json'), '');
    assert.equal(componentPath(sharded), sharded);
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
