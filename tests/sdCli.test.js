'use strict';

const { describe, it, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { runSdCli } = require('../engines/stablediffusioncpp/cli');
const generateApi = require('../engines/stablediffusioncpp/generate');
const engine = require('../engines/stablediffusioncpp/engine');

describe('runSdCli', () => {
  let root;

  after(async () => {
    if (root) {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it('kills an aborted child and escalates to SIGKILL', { timeout: 5000 }, async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-sdcli-'));
    const script = path.join(root, 'ignore-term.js');
    await fs.writeFile(
      script,
      [
        "process.on('SIGTERM', () => { process.stderr.write('term'); });",
        'setInterval(() => {}, 1000);',
        '',
      ].join('\n')
    );
    const abort = new AbortController();
    const pending = runSdCli([script], {
      bin: process.execPath,
      signal: abort.signal,
      killGraceMs: 40,
    });
    abort.abort();
    const result = await pending;
    assert.equal(result.code, null);
  });

  it('keeps only the tail of stdout and stderr', async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-sdcli-tail-'));
    const script = path.join(root, 'spill.js');
    await fs.writeFile(
      script,
      [
        'process.stdout.write(Buffer.alloc(80, 0x41));',
        'process.stdout.write(Buffer.alloc(80, 0x42));',
        'process.stderr.write(Buffer.alloc(80, 0x43));',
        'process.stderr.write(Buffer.alloc(80, 0x44));',
        '',
      ].join('\n')
    );
    const result = await runSdCli([script], {
      bin: process.execPath,
      outputTailBytes: 30,
    });
    assert.equal(result.stdout, 'B'.repeat(30));
    assert.equal(result.stderr, 'D'.repeat(30));
  });
});

describe('stablediffusioncpp abort handling', () => {
  const originalGenerate = generateApi.generateImage;
  let root;

  after(async () => {
    generateApi.generateImage = originalGenerate;
    delete process.env.GLAUX_SD_CLI;
    try {
      await engine.close({ final: false });
    } catch {
      /* not loaded */
    }
    if (root) {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  async function load() {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-sd-abort-'));
    const cache = path.join(root, 'Models');
    const model = path.join(cache, 'org', 'sd');
    await fs.mkdir(model, { recursive: true });
    await fs.writeFile(path.join(model, 'model.gguf'), 'gguf');
    await fs.writeFile(
      path.join(model, 'README.md'),
      '---\npipeline_tag: text-to-image\n---\n'
    );
    process.env.GLAUX_SD_CLI = process.execPath;
    await engine.configure({ modelsCacheDir: cache });
    await engine.chatbotCreate('org/sd');
  }

  it('reports an sd-cli abort string as an error unless the user aborted', async () => {
    await load();
    generateApi.generateImage = async () => {
      throw new Error('GGML_ASSERT Aborted');
    };
    await assert.rejects(
      () =>
        engine.runChat('org/sd', false, 'a red fox', {
          outputPath: path.join(root, 'out.png'),
        }),
      /Aborted/
    );

    generateApi.generateImage = async (_model, _prompt, _output, opts) => {
      await engine.chatStop();
      throw new Error('Aborted');
    };
    const cancelled = await engine.runChat('org/sd', false, 'a red fox', {
      outputPath: path.join(root, 'out.png'),
    });
    assert.equal(cancelled, '');
  });
});
