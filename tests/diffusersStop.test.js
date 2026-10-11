'use strict';

const { describe, it, before, after, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const engineManager = require('../engines/engineManager');

const recoveryInstalls = [];
const installRecovery = engineManager.setDiffusersKillRecovery;
engineManager.setDiffusersKillRecovery = (fn) => {
  recoveryInstalls.push(fn);
  installRecovery(fn);
};

const contextManager = require('../engines/contextManager');
const hf = require('../engines/huggingface/engine');
const { STOP_MARKER } = require('../engines/common/stopMarker');
const state = require('../src/main/state');
const {
  allocateSessionImagePath,
  releaseSessionImagePath,
} = require('../src/main/sessionImages');
const { clearPendingSessionFilename } = require('../src/main/domains/sessions');
const {
  cancelStreamRequest,
  handleEngineStreamStart,
} = require('../src/main/domains/engineBridge');

const WORKER_PID = 4321;

const original = {
  chatbotCreate: hf.chatbotCreate,
  chatbotDestroy: hf.chatbotDestroy,
  chatbotDestroyIfRunning: hf.chatbotDestroyIfRunning,
  runChat: hf.runChat,
  close: hf.close,
  chatStop: hf.chatStop,
  workerPid: hf.workerPid,
};

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(predicate, ms = 1000) {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > ms) {
      throw new Error('timed out waiting');
    }
    await delay(5);
  }
}

async function racePrompt(pending, ms = 1000) {
  let timer;
  const timeout = new Promise((resolve) => {
    timer = setTimeout(() => resolve({ kind: 'timeout' }), ms);
  });
  try {
    return await Promise.race([
      pending.then((value) => ({ kind: 'prompt', value })),
      timeout,
    ]);
  } finally {
    clearTimeout(timer);
  }
}

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

function fakeSender(id) {
  return {
    id,
    events: [],
    send(_channel, payload) {
      this.events.push(payload);
    },
    once() {},
    removeListener() {},
    isDestroyed() {
      return false;
    },
  };
}

describe('Diffusers stop grace', { concurrency: 1 }, () => {
  let root;
  let sessions;
  let cache;
  let closes;
  let stops;
  let exitSettled;
  let exitResolve;
  let resolveRun;
  let rejectRun;

  before(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-diffusers-stop-'));
    sessions = path.join(root, 'Sessions');
    cache = path.join(root, 'Models');
    const model = path.join(cache, 'org', 'diffusers');
    await fs.mkdir(sessions, { recursive: true });
    await fs.mkdir(model, { recursive: true });
    await fs.writeFile(path.join(model, 'model.safetensors'), 'weights');
    await fs.writeFile(
      path.join(model, 'README.md'),
      '---\npipeline_tag: text-to-image\n---\n'
    );
    hf.chatbotCreate = async () => {};
    hf.chatbotDestroy = async () => {};
    hf.chatbotDestroyIfRunning = async () => {};
    await engineManager.initialize({
      modelId: 'org/diffusers',
      modelsCacheDir: cache,
      sessionsRoot: sessions,
      resourcesRoot: path.join(root, 'Resources'),
      outputsRoot: path.join(root, 'Outputs'),
    });
    state.engineBootstrapped = true;
    const status = engineManager.getStatus();
    assert.equal(status.engineId, 'huggingface');
    assert.equal(status.pipelineTag, 'text-to-image');
  });

  beforeEach(() => {
    contextManager.clear();
    state.activeSessionFilename = null;
    state.activeStreamRequests.clear();
    clearPendingSessionFilename();
    closes = [];
    stops = 0;
    exitSettled = false;
    exitResolve = null;
    resolveRun = null;
    rejectRun = null;
    engineManager.setDiffusersStopGraceMs(40);
    engineManager.setDiffusersKillRecovery(async () => {});
    hf.workerPid = () => WORKER_PID;
    hf.chatStop = () => {
      stops += 1;
      return Promise.resolve();
    };
    hf.runChat = () =>
      new Promise((resolve, reject) => {
        resolveRun = resolve;
        rejectRun = reject;
      });
    hf.close = (options = {}) => {
      const final = options.final !== false;
      closes.push({ final });
      if (!final) {
        const reject = rejectRun;
        rejectRun = null;
        resolveRun = null;
        if (typeof reject === 'function') {
          reject(new Error('Engine closed'));
        }
        return new Promise((resolve) => {
          exitResolve = () => {
            exitSettled = true;
            resolve();
          };
        });
      }
      return Promise.resolve();
    };
  });

  after(async () => {
    hf.chatbotCreate = original.chatbotCreate;
    hf.chatbotDestroy = original.chatbotDestroy;
    hf.chatbotDestroyIfRunning = original.chatbotDestroyIfRunning;
    hf.runChat = original.runChat;
    hf.close = original.close;
    hf.chatStop = original.chatStop;
    hf.workerPid = original.workerPid;
    engineManager.setDiffusersStopGraceMs(engineManager.DIFFUSERS_STOP_GRACE_MS);
    engineManager.setDiffusersKillRecovery(async () => {});
    contextManager.clear();
    state.engineBootstrapped = false;
    state.activeSessionFilename = null;
    state.activeStreamRequests.clear();
    clearPendingSessionFilename();
    await engineManager.shutdown({ hfDestroyTimeoutMs: 20 });
    if (root) {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  async function plantOffload() {
    const folder = path.join('.glaux-t2i-offload', `pid-${WORKER_PID}`);
    const dirs = [
      path.join(cache, folder),
      path.join(cache, 'org', folder),
      path.join('/var/tmp', folder),
    ];
    for (const dir of dirs) {
      await fs.mkdir(dir, { recursive: true });
      await fs.writeFile(path.join(dir, 'weight.bin'), 'w');
    }
    return dirs;
  }

  function finishRunChat() {
    hf.runChat = async (_model, _thinking, _prompt, opts) => {
      await fs.writeFile(opts.outputPath, 'png');
      return { text: 'painted', imagePaths: [opts.outputPath] };
    };
  }

  it('keeps the production stop grace at 5000 ms', () => {
    assert.equal(engineManager.DIFFUSERS_STOP_GRACE_MS, 5000);
    engineManager.setDiffusersStopGraceMs(40);
    assert.equal(engineManager.DIFFUSERS_STOP_GRACE_MS, 5000);
    assert.equal(typeof recoveryInstalls[0], 'function');
  });

  it('kills a hung worker after the grace and settles before process exit', async () => {
    const allocated = await allocateSessionImagePath(sessions, 'kill.json');
    await fs.writeFile(allocated.absolutePath, 'png');
    const offloadDirs = await plantOffload();
    let reloaded = false;
    engineManager.setDiffusersKillRecovery(() => {
      reloaded = true;
    });

    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    assert.equal(engineManager.getStatus().phase, 'generating');
    engineManager.cancelGeneration();

    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    assert.equal(raced.value, STOP_MARKER);
    assert.equal(exitSettled, false);
    assert.equal(typeof exitResolve, 'function');
    await waitFor(() => reloaded);
    assert.deepEqual(closes, [{ final: false }]);
    assert.equal(exitSettled, false);
    await assert.rejects(fs.stat(allocated.absolutePath));
    for (const dir of offloadDirs) {
      await assert.rejects(fs.stat(dir));
    }
    const again = await allocateSessionImagePath(sessions, 'kill.json');
    assert.equal(again.relativePath, allocated.relativePath);
    releaseSessionImagePath(again.absolutePath, again.reservationId);
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
    ]);
  });

  it('does not kill the worker when the cooperative stop settles within the grace', async () => {
    const allocated = await allocateSessionImagePath(sessions, 'coop.json');
    await fs.writeFile(allocated.absolutePath, 'png');
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    resolveRun({ text: 'late pixels', imagePaths: [allocated.absolutePath] });
    assert.equal(await pending, STOP_MARKER);
    await delay(80);
    assert.equal(closes.length, 0);
    assert.equal(stops, 1);
    assert.equal(exitSettled, false);

    const next = await allocateSessionImagePath(sessions, 'coop-next.json');
    finishRunChat();
    const done = await engineManager.sendPrompt('a blue fox', {
      outputPath: next.absolutePath,
      reservationId: next.reservationId,
    });
    assert.deepEqual(done.imagePaths, [next.absolutePath]);
    assert.equal(closes.length, 0);
    assert.equal(await fs.readFile(next.absolutePath, 'utf8'), 'png');
  });

  it('ignores a repeated Stop and closes the worker once', async () => {
    const allocated = await allocateSessionImagePath(sessions, 'repeat.json');
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    engineManager.cancelGeneration();
    assert.equal(stops, 1);
    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    assert.equal(raced.value, STOP_MARKER);
    await delay(80);
    assert.deepEqual(closes, [{ final: false }]);
    assert.equal(stops, 1);
    assert.equal(exitSettled, false);
  });

  it('rejects a new prompt during the grace window and during reload', async () => {
    const allocated = await allocateSessionImagePath(sessions, 'reload.json');
    await fs.writeFile(allocated.absolutePath, 'png');
    let releaseReload = () => {};
    let entered = false;
    const enteredReload = new Promise((resolve) => {
      engineManager.setDiffusersKillRecovery(() => {
        entered = true;
        resolve();
        return new Promise((done) => {
          releaseReload = done;
        });
      });
    });

    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    const duringGrace = engineManager.contextSnapshot();
    await assert.rejects(
      engineManager.sendPrompt('a green fox', {
        outputPath: allocated.absolutePath,
        reservationId: allocated.reservationId,
      }),
      /Engine is busy\./
    );
    assert.deepEqual(await engineManager.contextSnapshot(), await duringGrace);

    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    await enteredReload;
    assert.equal(entered, true);
    await assert.rejects(
      engineManager.sendPrompt('a yellow fox', {
        outputPath: allocated.absolutePath,
        reservationId: allocated.reservationId,
      }),
      /Engine is busy\./
    );

    releaseReload();
    const next = await allocateSessionImagePath(sessions, 'reload-next.json');
    finishRunChat();
    const start = Date.now();
    let done;
    while (!done) {
      try {
        done = await engineManager.sendPrompt('a blue fox', {
          outputPath: next.absolutePath,
          reservationId: next.reservationId,
        });
      } catch (err) {
        if (!/Engine is busy\./.test(err.message) || Date.now() - start > 1000) {
          throw err;
        }
        await delay(5);
      }
    }
    assert.deepEqual(done.imagePaths, [next.absolutePath]);
    assert.equal(await fs.readFile(next.absolutePath, 'utf8'), 'png');
  });

  it('ignores a cancel that arrives after the run has finished', async () => {
    const allocated = await allocateSessionImagePath(sessions, 'late.json');
    finishRunChat();
    await engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    const before = await engineManager.contextSnapshot();
    assert.equal(before.length, 2);
    engineManager.cancelGeneration();
    await delay(80);
    assert.equal(stops, 0);
    assert.equal(closes.length, 0);
    assert.deepEqual(await engineManager.contextSnapshot(), before);
    assert.equal(engineManager.getStatus().phase, 'idle');
  });

  it('emits canceled for a stopped stream before the worker exits', async () => {
    const sender = fakeSender(3);
    const turn = handleEngineStreamStart(
      { sender },
      { requestId: 'diffusers-stream', message: 'a red fox' }
    );
    await waitFor(() => engineManager.getStatus().phase === 'generating');
    cancelStreamRequest('diffusers-stream', sender);
    await waitFor(() => sender.events.some((event) => event.type === 'canceled'), 1000);
    assert.equal(exitSettled, false);
    assert.deepEqual(closes, [{ final: false }]);
    await turn;
    assert.equal(exitSettled, false);
    const canceled = sender.events.find((event) => event.type === 'canceled');
    assert.equal(canceled.requestId, 'diffusers-stream');
    assert.equal(
      sender.events.some((event) => event.type === 'done' || event.type === 'error'),
      false
    );
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'a red fox', image: null },
      { role: 'assistant', text: STOP_MARKER, image: null },
    ]);
  });

  it('returns from shutdown during reload without waiting for worker exit', async () => {
    const allocated = await allocateSessionImagePath(sessions, 'quit.json');
    let sawShutdown = false;
    let releaseEntered;
    const entered = new Promise((resolve) => {
      releaseEntered = resolve;
    });
    engineManager.setDiffusersKillRecovery(async () => {
      releaseEntered();
      const start = Date.now();
      while (!engineManager.isShuttingDown() && Date.now() - start < 2000) {
        await delay(5);
      }
      sawShutdown = engineManager.isShuttingDown();
    });

    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    await entered;
    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    assert.equal(exitSettled, false);

    const outcome = await Promise.race([
      engineManager.shutdown({ hfDestroyTimeoutMs: 20 }).then(() => 'done'),
      delay(1000).then(() => 'hung'),
    ]);
    assert.equal(outcome, 'done');
    assert.equal(exitSettled, false);
    assert.equal(engineManager.getStatus().phase, 'idle');
    await waitFor(() => sawShutdown);
  });
});
