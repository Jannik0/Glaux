'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const engineManager = require('../engines/engineManager');
const contextManager = require('../engines/contextManager');
const stableDiffusion = require('../engines/stablediffusioncpp/engine');
const state = require('../src/main/state');
const { deliverSendMessage, handleEngineSendMessage } = require('../src/main/domains/engineBridge');
const { allocateSessionImagePath, releaseSessionImagePath } = require('../src/main/sessionImages');
const {
  clearPendingSessionFilename,
  getPendingSessionFilename,
  persistActiveSession,
  reserveActiveSessionFilename,
} = require('../src/main/domains/sessions');

describe('engine:sendMessage session persist', { concurrency: 1 }, () => {
  const originalRun = stableDiffusion.runChat;
  let root;
  let sessions;

  before(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-send-message-'));
    sessions = path.join(root, 'Sessions');
    const cache = path.join(root, 'Models');
    const model = path.join(cache, 'org', 'sd');
    await fs.mkdir(sessions);
    await fs.mkdir(model, { recursive: true });
    await fs.writeFile(path.join(model, 'model.gguf'), 'gguf');
    await fs.writeFile(path.join(model, 'README.md'), '---\npipeline_tag: text-to-image\n---\n');
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
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    await engineManager.shutdown({ hfDestroyTimeoutMs: 20 });
    if (root) {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  function persistDeps() {
    return {
      contextSnapshot: () => engineManager.contextSnapshot(),
      writeSessionFile: async (name, messages) => {
        await fs.writeFile(path.join(sessions, name), `${JSON.stringify(messages, null, 2)}\n`);
      },
    };
  }

  it('writes the session file and keeps the sidecar after a successful turn', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    const filename = reserveActiveSessionFilename();
    const allocated = await allocateSessionImagePath(sessions, filename);
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'done', imagePaths: [opts.outputPath] };
    };

    const result = await deliverSendMessage('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
      persist: persistDeps(),
    });

    assert.equal(result.response, 'done');
    assert.equal(state.activeSessionFilename, filename);
    assert.equal(getPendingSessionFilename(), null);
    assert.equal(await fs.readFile(allocated.absolutePath, 'utf8'), 'png');
    const saved = JSON.parse(await fs.readFile(path.join(sessions, filename), 'utf8'));
    const image = saved
      .flatMap((msg) => (Array.isArray(msg.content) ? msg.content : []))
      .find((part) => part.type === 'image');
    assert.equal(image.relativePath, allocated.relativePath);
    assert.equal(image.source, 'sessions');
  });

  it('clears the pending name and leaves no sidecar when generation fails', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    const filename = reserveActiveSessionFilename();
    const allocated = await allocateSessionImagePath(sessions, filename);
    const jsonPath = path.join(sessions, filename);
    let jsonBefore = null;
    try {
      jsonBefore = await fs.stat(jsonPath);
    } catch {
      jsonBefore = null;
    }
    await fs.writeFile(allocated.absolutePath, 'png');
    stableDiffusion.runChat = async () => {
      throw new Error('sd failed');
    };

    await assert.rejects(
      () =>
        deliverSendMessage('a red fox', {
          outputPath: allocated.absolutePath,
          reservationId: allocated.reservationId,
          persist: persistDeps(),
        }),
      /sd failed/
    );

    assert.equal(state.activeSessionFilename, null);
    assert.equal(getPendingSessionFilename(), null);
    await assert.rejects(fs.stat(allocated.absolutePath));
    const reused = await allocateSessionImagePath(sessions, filename);
    assert.equal(reused.relativePath, allocated.relativePath);
    releaseSessionImagePath(reused.absolutePath, reused.reservationId);
    if (jsonBefore) {
      const jsonAfter = await fs.stat(jsonPath);
      assert.equal(jsonAfter.mtimeMs, jsonBefore.mtimeMs);
    } else {
      await assert.rejects(fs.stat(jsonPath));
    }
  });

  it('allocates the sidecar through textToImageOutputPath', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      assert.equal(opts.outputPath.startsWith(sessions), true);
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'allocated', imagePaths: [opts.outputPath] };
    };

    const result = await deliverSendMessage('a painted fox', { persist: persistDeps() });
    assert.equal(result.response, 'allocated');
    assert.equal(result.images.length, 1);
    assert.match(result.images[0].relativePath, /-\d+\.png$/);
    assert.equal(await fs.readFile(path.join(sessions, result.images[0].relativePath), 'utf8'), 'png');
    assert.equal(state.activeSessionFilename.endsWith('.json'), true);
    assert.equal(getPendingSessionFilename(), null);
  });

  it('keeps the sidecar when persisting the session fails so the next save can record it', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    const outputs = path.join(root, 'Outputs');
    await fs.mkdir(outputs, { recursive: true });
    const keep = path.join(outputs, 'keep.png');
    await fs.writeFile(keep, 'keep');
    let sidecar;
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      sidecar = opts.outputPath;
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'done', imagePaths: [opts.outputPath] };
    };

    const result = await deliverSendMessage('a red fox', {
      persist: {
        contextSnapshot: () => engineManager.contextSnapshot(),
        writeSessionFile: async () => {
          throw new Error('disk full');
        },
      },
    });

    assert.equal(result.response, 'done');
    assert.equal(state.activeSessionFilename, null);
    assert.equal(getPendingSessionFilename(), null);
    assert.equal(await fs.readFile(sidecar, 'utf8'), 'png');
    assert.equal(await fs.readFile(keep, 'utf8'), 'keep');
    const live = await engineManager.contextSnapshot();
    const liveImage = live
      .flatMap((msg) => (Array.isArray(msg.content) ? msg.content : []))
      .find((part) => part.type === 'image');
    assert.equal(liveImage.relativePath, result.images[0].relativePath);
    assert.equal(liveImage.source, 'sessions');

    let saved;
    await persistActiveSession({
      contextSnapshot: () => engineManager.contextSnapshot(),
      writeSessionFile: async (name, messages) => {
        saved = messages;
        await fs.writeFile(path.join(sessions, name), `${JSON.stringify(messages)}\n`);
      },
    });
    const savedImage = saved
      .flatMap((msg) => (Array.isArray(msg.content) ? msg.content : []))
      .find((part) => part.type === 'image');
    assert.equal(savedImage.relativePath, liveImage.relativePath);
    assert.equal(await fs.readFile(path.join(sessions, savedImage.relativePath), 'utf8'), 'png');

    await fs.writeFile(keep, 'orig');
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'done', imagePaths: [opts.outputPath] };
    };
    await deliverSendMessage('a red fox', {
      outputPath: keep,
      persist: {
        contextSnapshot: () => engineManager.contextSnapshot(),
        writeSessionFile: async () => {
          throw new Error('disk full');
        },
      },
    });
    assert.equal(await fs.readFile(keep, 'utf8'), 'png');
  });

  it('wraps engine:sendMessage and refuses to run before bootstrap', async () => {
    contextManager.clear();
    state.engineBootstrapped = false;
    const denied = await handleEngineSendMessage('a red fox');
    assert.equal(denied.ok, false);
    assert.equal(denied.errorInfo.code, 'E_NOT_READY');
    assert.deepEqual(await engineManager.contextSnapshot(), []);

    state.engineBootstrapped = true;
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'from-ipc', imagePaths: [opts.outputPath] };
    };
    const allowed = await handleEngineSendMessage('a painted fox');
    assert.equal(allowed.ok, true);
    assert.equal(allowed.response, 'from-ipc');
    assert.equal(allowed.images.length, 1);
    state.engineBootstrapped = false;
  });
});
