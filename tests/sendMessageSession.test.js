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
const { deliverSendMessage } = require('../src/main/domains/engineBridge');
const { allocateSessionImagePath } = require('../src/main/sessionImages');
const {
  clearPendingSessionFilename,
  getPendingSessionFilename,
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
          persist: persistDeps(),
        }),
      /sd failed/
    );

    assert.equal(state.activeSessionFilename, null);
    assert.equal(getPendingSessionFilename(), null);
    await assert.rejects(fs.stat(allocated.absolutePath));
    if (jsonBefore) {
      const jsonAfter = await fs.stat(jsonPath);
      assert.equal(jsonAfter.mtimeMs, jsonBefore.mtimeMs);
    } else {
      await assert.rejects(fs.stat(jsonPath));
    }
  });
});
