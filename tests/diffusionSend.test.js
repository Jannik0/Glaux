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
const state = require('../src/main/state');
const { allocateSessionImagePath } = require('../src/main/sessionImages');
const { cancelStreamRequest, handleEngineCancel } = require('../src/main/domains/engineBridge');

function contextShape(messages) {
  return messages.map((msg) => {
    const text = (msg.content || [])
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join('\n');
    const image = (msg.content || []).find((part) => part.type === 'image');
    return {
      role: msg.role,
      text,
      image: image ? image.relativePath : null,
    };
  });
}

describe('sendPrompt diffusion', { concurrency: 1 }, () => {
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
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
    ]);
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
    await fs.writeFile(secondPath, 'new');
    releaseFirst();
    assert.equal(await first, STOP_MARKER);
    assert.equal(engineManager.getStatus().phase, 'generating');
    assert.equal(await fs.readFile(secondPath, 'utf8'), 'new');
    await assert.rejects(fs.stat(firstPath));
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
    ]);

    releaseSecond();
    const done = await second;
    assert.deepEqual(done.imagePaths, [secondPath]);
    assert.equal(engineManager.getStatus().phase, 'idle');
    assert.equal(await fs.readFile(secondPath, 'utf8'), 'new');
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
      { role: 'assistant', text: '', image: 'newer.png' },
    ]);
  });

  it('does not roll back the newer user turn when an older run returns no image', async () => {
    contextManager.clear();
    const firstPath = path.join(sessions, 'empty-older.png');
    const secondPath = path.join(sessions, 'empty-newer.png');
    await fs.writeFile(firstPath, 'old');
    let releaseFirst;
    let markFirstStarted;
    const firstStarted = new Promise((resolve) => {
      markFirstStarted = resolve;
    });
    stableDiffusion.runChat = () =>
      new Promise((resolve, reject) => {
        markFirstStarted();
        releaseFirst = () => reject(new Error('Image generation did not return an image.'));
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
    releaseFirst();
    assert.equal(await first, STOP_MARKER);
    await assert.rejects(fs.stat(firstPath));
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
    ]);

    await fs.writeFile(secondPath, 'new');
    releaseSecond();
    await second;
    assert.equal(await fs.readFile(secondPath, 'utf8'), 'new');
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
      { role: 'assistant', text: '', image: 'empty-newer.png' },
    ]);
  });

  it('inserts a stop marker when an older run settles empty after a newer prompt', async () => {
    contextManager.clear();
    const firstPath = path.join(sessions, 'blank-older.png');
    const secondPath = path.join(sessions, 'blank-newer.png');
    await fs.writeFile(firstPath, 'old');
    let releaseFirst;
    let markFirstStarted;
    const firstStarted = new Promise((resolve) => {
      markFirstStarted = resolve;
    });
    stableDiffusion.runChat = () =>
      new Promise((resolve) => {
        markFirstStarted();
        releaseFirst = () => resolve({ text: '', imagePaths: [] });
      });
    const first = engineManager.sendPrompt('a red fox', { outputPath: firstPath });
    await firstStarted;
    engineManager.cancelGeneration();

    let releaseSecond;
    const secondStarted = new Promise((resolve) => {
      stableDiffusion.runChat = () =>
        new Promise((resolveImage) => {
          resolve();
          releaseSecond = () => resolveImage({ text: '', imagePaths: [secondPath] });
        });
    });
    const second = engineManager.sendPrompt('a blue fox', { outputPath: secondPath });
    await secondStarted;
    releaseFirst();
    assert.equal(await first, STOP_MARKER);
    await assert.rejects(fs.stat(firstPath));
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
    ]);
    await fs.writeFile(secondPath, 'new');
    releaseSecond();
    await second;
    assert.equal(await fs.readFile(secondPath, 'utf8'), 'new');
  });

  it('inserts the older stop marker when that run settles after the newer one finished', async () => {
    contextManager.clear();
    const firstPath = path.join(sessions, 'late-older.png');
    const secondPath = path.join(sessions, 'late-newer.png');
    await fs.writeFile(firstPath, 'old');
    await fs.writeFile(secondPath, 'new');
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
    stableDiffusion.runChat = async () => ({ text: '', imagePaths: [secondPath] });
    const second = await engineManager.sendPrompt('a blue fox', { outputPath: secondPath });
    assert.deepEqual(second.imagePaths, [secondPath]);
    releaseFirst();
    assert.equal(await first, STOP_MARKER);
    await assert.rejects(fs.stat(firstPath));
    assert.equal(await fs.readFile(secondPath, 'utf8'), 'new');
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
      { role: 'assistant', text: '', image: 'late-newer.png' },
    ]);
  });

  it('does not let a late cancel from an old request stop a newer run', async () => {
    contextManager.clear();
    const firstPath = path.join(sessions, 'scope-older.png');
    const secondPath = path.join(sessions, 'scope-newer.png');
    await fs.writeFile(firstPath, 'old');
    await fs.writeFile(secondPath, 'new');
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
    const first = engineManager.sendPrompt('a red fox', {
      outputPath: firstPath,
      requestId: 'req-old',
      senderId: 4,
    });
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
    const second = engineManager.sendPrompt('a blue fox', {
      outputPath: secondPath,
      requestId: 'req-new',
      senderId: 4,
    });
    await secondStarted;
    state.activeStreamRequests.set('req-old', { canceled: false });
    cancelStreamRequest('req-old', { id: 4 });
    assert.equal(state.activeStreamRequests.get('req-old').canceled, true);
    assert.equal(engineManager.getStatus().phase, 'generating');
    releaseSecond();
    const done = await second;
    assert.deepEqual(done.imagePaths, [secondPath]);
    assert.equal(await fs.readFile(secondPath, 'utf8'), 'new');
    releaseFirst();
    assert.equal(await first, STOP_MARKER);
    await assert.rejects(fs.stat(firstPath));
    state.activeStreamRequests.delete('req-old');
  });

  it('cancels a stream request only when that request owns the active run', async () => {
    contextManager.clear();
    const outputPath = path.join(sessions, 'scope-sender.png');
    await fs.writeFile(outputPath, 'png');
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
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath,
      requestId: 'req-live',
      senderId: 4,
    });
    await started;
    state.activeStreamRequests.set('req-live', { canceled: false });
    cancelStreamRequest('req-other', { id: 4 });
    assert.equal(state.activeStreamRequests.get('req-live').canceled, false);
    assert.equal(engineManager.getStatus().phase, 'generating');
    cancelStreamRequest('req-live', { id: 4 });
    assert.equal(state.activeStreamRequests.get('req-live').canceled, true);
    assert.equal(engineManager.getStatus().phase, 'idle');
    release();
    assert.equal(await pending, STOP_MARKER);
    await assert.rejects(fs.stat(outputPath));
    state.activeStreamRequests.delete('req-live');
  });

  it('engine:cancel still cancels the active run', async () => {
    contextManager.clear();
    const outputPath = path.join(sessions, 'engine-cancel.png');
    await fs.writeFile(outputPath, 'png');
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
    const pending = engineManager.sendPrompt('a red fox', { outputPath, requestId: 'req-cancel' });
    await started;
    const result = await handleEngineCancel();
    assert.equal(result.ok, true);
    assert.equal(engineManager.getStatus().phase, 'idle');
    release();
    assert.equal(await pending, STOP_MARKER);
    await assert.rejects(fs.stat(outputPath));
  });

  it('does not rewrite a finished diffusion turn when that request is cancelled', async () => {
    contextManager.clear();
    const allocated = await allocateSessionImagePath(sessions, 'finished-turn.json');
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'painted', imagePaths: [opts.outputPath] };
    };
    const done = await engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
      requestId: 'req-img',
      senderId: 4,
    });
    assert.deepEqual(done.imagePaths, [allocated.absolutePath]);
    const before = await engineManager.contextSnapshot().then(contextShape);
    assert.deepEqual(before, [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: 'painted', image: allocated.relativePath },
    ]);
    let stops = 0;
    const originalStop = stableDiffusion.chatStop;
    stableDiffusion.chatStop = () => {
      stops += 1;
      return Promise.resolve();
    };
    engineManager.cancelGeneration();
    engineManager.cancelRequestGeneration({ requestId: 'req-img', senderId: 4, senderFallback: true });
    const idle = await handleEngineCancel();
    stableDiffusion.chatStop = originalStop;
    assert.equal(idle.ok, true);
    assert.equal(stops, 0);
    assert.equal(engineManager.getStatus().phase, 'idle');
    assert.equal(engineManager.activeRunStopSealed(), false);
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), before);
    assert.equal(await fs.readFile(allocated.absolutePath, 'utf8'), 'png');
  });

  it('passes alternating history to the newer diffusion run and discards only the older reserved sidecar', async () => {
    contextManager.clear();
    const first = await allocateSessionImagePath(sessions, 'reserved-turn.json');
    const second = await allocateSessionImagePath(sessions, 'reserved-turn.json');
    assert.notEqual(first.relativePath, second.relativePath);
    let releaseFirst;
    let markFirstStarted;
    const firstStarted = new Promise((resolve) => {
      markFirstStarted = resolve;
    });
    stableDiffusion.runChat = (_model, _thinking, _prompt, opts) =>
      new Promise((resolve) => {
        markFirstStarted();
        releaseFirst = async () => {
          await fs.writeFile(opts.outputPath, 'old');
          resolve({ text: '', imagePaths: [opts.outputPath] });
        };
      });
    const older = engineManager.sendPrompt('a red fox', {
      outputPath: first.absolutePath,
      reservationId: first.reservationId,
    });
    await firstStarted;
    engineManager.cancelGeneration();

    let secondMessages;
    let secondContext;
    let releaseSecond;
    let markSecondStarted;
    const secondStarted = new Promise((resolve) => {
      markSecondStarted = resolve;
    });
    stableDiffusion.runChat = (_model, _thinking, _prompt, opts) =>
      new Promise((resolve) => {
        secondMessages = opts.messages;
        secondContext = contextManager.snapshot();
        markSecondStarted();
        releaseSecond = async () => {
          await fs.writeFile(opts.outputPath, 'new');
          resolve({ text: '', imagePaths: [opts.outputPath] });
        };
      });
    const newer = engineManager.sendPrompt('a blue fox', {
      outputPath: second.absolutePath,
      reservationId: second.reservationId,
    });
    await secondStarted;
    assert.deepEqual(secondMessages, []);
    assert.deepEqual(contextShape(secondContext), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
    ]);
    await releaseFirst();
    assert.equal(await older, STOP_MARKER);
    await assert.rejects(fs.stat(first.absolutePath));
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
    ]);
    await releaseSecond();
    const done = await newer;
    assert.deepEqual(done.imagePaths, [second.absolutePath]);
    assert.equal(await fs.readFile(second.absolutePath, 'utf8'), 'new');
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
      { role: 'user', text: 'a blue fox', image: null },
      { role: 'assistant', text: '', image: second.relativePath },
    ]);
  });
});
