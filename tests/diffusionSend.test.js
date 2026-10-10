'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const engineManager = require('../engines/engineManager');
const contextManager = require('../engines/contextManager');
const stableDiffusion = require('../engines/stablediffusioncpp/engine');
const { STOP_MARKER } = require('../engines/common/stopMarker');

describe('sendPrompt diffusion', () => {
  const originalRun = stableDiffusion.runChat;
  let root;
  let sessions;
  let outside;

  before(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-diffusion-send-'));
    sessions = path.join(root, 'Sessions');
    const cache = path.join(root, 'Models');
    const model = path.join(cache, 'org', 'sd');
    await fs.mkdir(sessions);
    await fs.mkdir(model, { recursive: true });
    await fs.writeFile(path.join(model, 'model.gguf'), 'gguf');
    await fs.writeFile(
      path.join(model, 'README.md'),
      '---\npipeline_tag: text-to-image\n---\n'
    );
    outside = path.join(root, 'outside.png');
    process.env.GLAUX_SD_CLI = process.execPath;
    await engineManager.initialize({
      modelId: 'org/sd',
      modelsCacheDir: cache,
      sessionsRoot: sessions,
      resourcesRoot: path.join(root, 'Resources'),
      outputsRoot: path.join(root, 'Outputs'),
    });
  });

  after(async () => {
    stableDiffusion.runChat = originalRun;
    delete process.env.GLAUX_SD_CLI;
    contextManager.clear();
    await engineManager.shutdown({ hfDestroyTimeoutMs: 20 });
    if (root) {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it('rolls back the user turn and deletes only a session sidecar on error', async () => {
    contextManager.clear();
    const outputPath = path.join(sessions, 'chat-1.png');
    await fs.writeFile(outputPath, 'png');
    await fs.writeFile(outside, 'keep');
    stableDiffusion.runChat = async () => {
      throw new Error('sd failed');
    };
    await assert.rejects(
      () => engineManager.sendPrompt('a red fox', { outputPath }),
      /sd failed/
    );
    assert.deepEqual(await engineManager.contextSnapshot(), []);
    await assert.rejects(fs.stat(outputPath));
    assert.equal(engineManager.getStatus().phase, 'idle');

    await fs.writeFile(outside, 'keep');
    await assert.rejects(
      () => engineManager.sendPrompt('a red fox', { outputPath: outside }),
      /sd failed/
    );
    assert.equal(await fs.readFile(outside, 'utf8'), 'keep');
    assert.deepEqual(await engineManager.contextSnapshot(), []);
  });

  it('discards the image on cancel', async () => {
    contextManager.clear();
    const outputPath = path.join(sessions, 'run-1.png');
    await fs.writeFile(outputPath, 'one');
    let release;
    let markStarted;
    const started = new Promise((resolve) => {
      markStarted = resolve;
    });
    stableDiffusion.runChat = () =>
      new Promise((resolve) => {
        markStarted();
        release = () => resolve({ text: '', imagePaths: [outputPath] });
      });
    const pending = engineManager.sendPrompt('a red fox', { outputPath });
    await started;
    engineManager.cancelGeneration();
    release();
    assert.equal(await pending, STOP_MARKER);
    await assert.rejects(fs.stat(outputPath));
    assert.equal(engineManager.getStatus().phase, 'idle');
    const snap = await engineManager.contextSnapshot();
    assert.equal(snap.at(-1).role, 'assistant');
    assert.match(JSON.stringify(snap.at(-1).content), /\[STOP\]/);
  });

  it('does not let an older diffusion run clear a newer run flag', async () => {
    contextManager.clear();
    const firstPath = path.join(sessions, 'older.png');
    const secondPath = path.join(sessions, 'newer.png');
    await fs.writeFile(firstPath, 'old');
    let releaseFirst;
    let markFirstStarted;
    const firstStarted = new Promise((resolve) => {
      markFirstStarted = resolve;
    });
    stableDiffusion.runChat = () =>
      new Promise((resolve) => {
        markFirstStarted();
        releaseFirst = () => resolve({ text: '', imagePaths: [firstPath] });
      });
    const first = engineManager.sendPrompt('a red fox', { outputPath: firstPath });
    await firstStarted;
    engineManager.cancelGeneration();

    let releaseSecond;
    let markSecondStarted;
    const secondStarted = new Promise((resolve) => {
      markSecondStarted = resolve;
    });
    stableDiffusion.runChat = () =>
      new Promise((resolve) => {
        markSecondStarted();
        releaseSecond = () => resolve({ text: '', imagePaths: [secondPath] });
      });
    const second = engineManager.sendPrompt('a blue fox', { outputPath: secondPath });
    await secondStarted;
    assert.equal(engineManager.getStatus().phase, 'generating');
    releaseFirst();
    await first;
    assert.equal(engineManager.getStatus().phase, 'generating');

    await fs.writeFile(secondPath, 'new');
    releaseSecond();
    const done = await second;
    assert.deepEqual(done.imagePaths, [secondPath]);
    assert.equal(engineManager.getStatus().phase, 'idle');
    assert.equal(await fs.readFile(secondPath, 'utf8'), 'new');
  });
});
