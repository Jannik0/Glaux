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
const {
  handleEngineStreamStart,
  handleEngineCancel,
} = require('../src/main/domains/engineBridge');
const {
  clearPendingSessionFilename,
  getPendingSessionFilename,
  reserveActiveSessionFilename,
} = require('../src/main/domains/sessions');

function fakeSender(initialId) {
  let destroyed = false;
  let onDestroyed = null;
  let idReads = 0;
  return {
    events: [],
    get id() {
      idReads += 1;
      if (destroyed) {
        throw new Error('read sender.id after destruction');
      }
      return initialId;
    },
    get idReads() {
      return idReads;
    },
    isDestroyed() {
      return destroyed;
    },
    send(_channel, payload) {
      this.events.push(payload);
    },
    once(event, fn) {
      if (event === 'destroyed') {
        onDestroyed = fn;
      }
    },
    removeListener() {
      onDestroyed = null;
    },
    destroy() {
      destroyed = true;
      if (onDestroyed) {
        onDestroyed();
      }
    },
  };
}

describe('engine stream handler', { concurrency: 1 }, () => {
  const originalRun = stableDiffusion.runChat;
  let root;
  let sessions;

  before(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-stream-'));
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
    state.engineBootstrapped = true;
  });

  after(async () => {
    stableDiffusion.runChat = originalRun;
    delete process.env.GLAUX_SD_CLI;
    contextManager.clear();
    state.activeSessionFilename = null;
    state.engineBootstrapped = false;
    state.activeStreamRequests.clear();
    clearPendingSessionFilename();
    await engineManager.shutdown({ hfDestroyTimeoutMs: 20 });
    if (root) {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it('releases a reserved sidecar when sendPrompt rejects before the run starts', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    const filename = reserveActiveSessionFilename();
    const sender = fakeSender(4);
    const turn = handleEngineStreamStart(
      { sender },
      { requestId: 'bad-attachment', message: 'a red fox', files: [{ source: 'resources', relativePath: 'notes.txt' }] }
    );
    await turn;
    const error = sender.events.find((event) => event.type === 'error');
    assert.ok(error);
    assert.match(error.errorInfo.message, /optional image|text prompt/i);
    const again = await allocateSessionImagePath(sessions, filename);
    assert.equal(again.relativePath, `${filename.replace(/\.json$/i, '')}-1.png`);
    const { releaseSessionImagePath } = require('../src/main/sessionImages');
    releaseSessionImagePath(again.absolutePath, again.reservationId);
    assert.equal(getPendingSessionFilename(), null);
  });

  it('does not cancel a running generation when a busy stream request fails', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    const blocker = await allocateSessionImagePath(sessions, 'blocker.json');
    let release;
    let markStarted;
    const started = new Promise((resolve) => {
      markStarted = resolve;
    });
    stableDiffusion.runChat = (_model, _thinking, _prompt, opts) =>
      new Promise((resolve) => {
        markStarted();
        release = async () => {
          await fs.writeFile(opts.outputPath, 'kept');
          resolve({ text: '', imagePaths: [opts.outputPath] });
        };
      });
    const running = engineManager.sendPrompt('a red fox', {
      outputPath: blocker.absolutePath,
      reservationId: blocker.reservationId,
      requestId: 'running',
      senderId: 4,
    });
    await started;

    const filename = reserveActiveSessionFilename();
    const sender = fakeSender(4);
    await handleEngineStreamStart({ sender }, { requestId: 'busy-one', message: 'a blue fox' });
    const error = sender.events.find((event) => event.type === 'error');
    assert.equal(error.errorInfo.code, 'E_BUSY');
    assert.equal(engineManager.getStatus().phase, 'generating');
    const freed = await allocateSessionImagePath(sessions, filename);
    assert.equal(freed.relativePath, `${filename.replace(/\.json$/i, '')}-1.png`);
    const { releaseSessionImagePath } = require('../src/main/sessionImages');
    releaseSessionImagePath(freed.absolutePath, freed.reservationId);

    await release();
    const done = await running;
    assert.deepEqual(done.imagePaths, [blocker.absolutePath]);
    assert.equal(await fs.readFile(blocker.absolutePath, 'utf8'), 'kept');
  });

  it('captures the sender id before the webContents is destroyed', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    let release;
    let markStarted;
    const started = new Promise((resolve) => {
      markStarted = resolve;
    });
    stableDiffusion.runChat = (_model, _thinking, _prompt, opts) =>
      new Promise((resolve) => {
        markStarted();
        release = () => resolve({ text: '', imagePaths: [opts.outputPath] });
      });
    const sender = fakeSender(4);
    const readsBefore = sender.idReads;
    const turn = handleEngineStreamStart({ sender }, { requestId: 'destroy-me', message: 'a red fox' });
    await started;
    assert.equal(engineManager.getStatus().phase, 'generating');
    assert.doesNotThrow(() => sender.destroy());
    assert.equal(sender.idReads, readsBefore + 1);
    assert.equal(engineManager.getStatus().phase, 'idle');
    release();
    await turn;
    const canceled = sender.events.find((event) => event.type === 'canceled' || event.type === 'error');
    assert.equal(canceled, undefined);
  });

  it('keeps the sidecar after a stream persist failure and still reports it as done', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'done', imagePaths: [opts.outputPath] };
    };
    const sender = fakeSender(4);
    await handleEngineStreamStart({ sender }, { requestId: 'persist-fail', message: 'a red fox' });
    const done = sender.events.find((event) => event.type === 'done');
    assert.ok(done);
    assert.equal(done.images.length, 1);
    const sidecar = path.join(sessions, done.images[0].relativePath);
    assert.equal(await fs.readFile(sidecar, 'utf8'), 'png');
    const live = await engineManager.contextSnapshot();
    const image = live
      .flatMap((msg) => (Array.isArray(msg.content) ? msg.content : []))
      .find((part) => part.type === 'image');
    assert.equal(image.relativePath, done.images[0].relativePath);
    assert.equal(getPendingSessionFilename(), null);
  });

  it('engine:cancel stops the active stream run', async () => {
    contextManager.clear();
    let release;
    let markStarted;
    const started = new Promise((resolve) => {
      markStarted = resolve;
    });
    stableDiffusion.runChat = () =>
      new Promise((resolve) => {
        markStarted();
        release = () => resolve({ text: '', imagePaths: [] });
      });
    const sender = fakeSender(8);
    const turn = handleEngineStreamStart({ sender }, { requestId: 'stop-button', message: 'a red fox' });
    await started;
    const result = await handleEngineCancel();
    assert.equal(result.ok, true);
    assert.equal(engineManager.getStatus().phase, 'idle');
    release();
    await turn;
    const done = sender.events.find((event) => event.type === 'done');
    assert.deepEqual(
      {
        requestId: done.requestId,
        type: done.type,
        response: done.response,
        images: done.images,
      },
      {
        requestId: 'stop-button',
        type: 'done',
        response: STOP_MARKER,
        images: [],
      }
    );
    const snap = await engineManager.contextSnapshot();
    assert.equal(
      snap.some(
        (msg) =>
          msg.role === 'assistant' &&
          msg.content.some((part) => part.type === 'text' && part.text === STOP_MARKER)
      ),
      true
    );
  });

  it('engine:cancel while idle leaves the finished turn unchanged', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'kept', imagePaths: [opts.outputPath] };
    };
    const sender = fakeSender(8);
    await handleEngineStreamStart({ sender }, { requestId: 'idle-after', message: 'a red fox' });
    const before = await engineManager.contextSnapshot();
    assert.equal(before.length, 2);
    const phase = engineManager.getStatus().phase;
    let stops = 0;
    const originalStop = stableDiffusion.chatStop;
    stableDiffusion.chatStop = () => {
      stops += 1;
      return Promise.resolve();
    };
    const result = await handleEngineCancel();
    stableDiffusion.chatStop = originalStop;
    assert.equal(result.ok, true);
    assert.equal(stops, 0);
    assert.equal(engineManager.getStatus().phase, phase);
    assert.deepEqual(await engineManager.contextSnapshot(), before);
  });

  it('an owned generation error removes only its user turn and does not seal a stop', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    stableDiffusion.runChat = async (_model, _thinking, _prompt, opts) => {
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'kept', imagePaths: [opts.outputPath] };
    };
    const keptSender = fakeSender(4);
    await handleEngineStreamStart({ sender: keptSender }, { requestId: 'kept-turn', message: 'a red fox' });
    const kept = await engineManager.contextSnapshot();
    assert.equal(kept.length, 2);

    stableDiffusion.runChat = async () => {
      throw new Error('sd failed');
    };
    const sender = fakeSender(4);
    await handleEngineStreamStart({ sender }, { requestId: 'owned-error', message: 'a blue fox' });
    const error = sender.events.find((event) => event.type === 'error');
    assert.ok(error);
    assert.match(error.errorInfo.message, /sd failed/);
    const after = await engineManager.contextSnapshot();
    assert.deepEqual(after, kept);
    engineManager.cancelRequestGeneration({ requestId: 'owned-error' });
    engineManager.cancelGeneration();
    assert.deepEqual(await engineManager.contextSnapshot(), kept);
  });
});
