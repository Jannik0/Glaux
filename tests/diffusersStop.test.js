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
  requestWorkerReload,
} = require('../src/main/domains/engineBridge');
const { buildEngineStatusForRenderer } = require('../src/main/domains/engineCore');

const WORKER_PID = 4321;

const original = {
  chatbotCreate: hf.chatbotCreate,
  chatbotDestroy: hf.chatbotDestroy,
  chatbotDestroyIfRunning: hf.chatbotDestroyIfRunning,
  runChat: hf.runChat,
  close: hf.close,
  chatStop: hf.chatStop,
  workerPid: hf.workerPid,
  downloadModel: hf.downloadModel,
  cancelDownloadModel: hf.cancelDownloadModel,
  chatbotSupportsThinking: hf.chatbotSupportsThinking,
  chatbotHasChatTemplate: hf.chatbotHasChatTemplate,
};

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** True when *promise* is still pending after one event-loop turn. A timer that is already due can fire during that turn. */
async function assertStillPending(promise) {
  await new Promise((resolve) => setImmediate(resolve));
  let settled = false;
  const mark = () => {
    settled = true;
  };
  promise.then(mark, mark);
  await Promise.resolve();
  assert.equal(settled, false);
}

/** Keep the event loop alive while an unref'd stop timer settles `promise`. */
async function awaitAlive(promise, ms = 15000) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('timed out')), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

async function waitFor(predicate, ms = 15000) {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > ms) {
      throw new Error('timed out waiting');
    }
    await delay(5);
  }
}

async function racePrompt(pending, ms = 15000) {
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

function referencedTimeouts() {
  if (typeof process._getActiveHandles !== 'function') {
    return [];
  }
  return process._getActiveHandles().filter((handle) => {
    if (!handle || typeof handle.hasRef !== 'function' || !handle.hasRef()) {
      return false;
    }
    return Boolean(handle.constructor && handle.constructor.name === 'Timeout');
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
  let rejectDownload;
  let offloadBases;

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
    offloadBases = [
      path.join(root, 'offload', 'cache'),
      path.join(root, 'offload', 'parent'),
      path.join(root, 'offload', 'tmp'),
    ];
    engineManager.setDiffusersOffloadBaseDirs(offloadBases);
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
    engineManager.setDiffusersWorkerExitMs(25);
    engineManager.setDiffusersOffloadBaseDirs(offloadBases);
    engineManager.setDiffusersKillRecovery(async () => {});
    rejectDownload = null;
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
    hf.downloadModel = async () => {
      throw new Error('unexpected download');
    };
    hf.cancelDownloadModel = async () => {};
    hf.chatbotSupportsThinking = async () => false;
    hf.chatbotHasChatTemplate = async () => false;
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
        if (typeof rejectDownload === 'function') {
          const rejectDl = rejectDownload;
          rejectDownload = null;
          rejectDl(new Error('Engine closed'));
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
    hf.downloadModel = original.downloadModel;
    hf.cancelDownloadModel = original.cancelDownloadModel;
    hf.chatbotSupportsThinking = original.chatbotSupportsThinking;
    hf.chatbotHasChatTemplate = original.chatbotHasChatTemplate;
    engineManager.setDiffusersOffloadBaseDirs(null);
    engineManager.setDiffusersWorkerExitMs(engineManager.DIFFUSERS_WORKER_EXIT_MS);
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
    const dirs = offloadBases.map((base) => path.join(base, folder));
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
    assert.equal(engineManager.DIFFUSERS_WORKER_EXIT_MS, 5000);
    engineManager.setDiffusersStopGraceMs(40);
    engineManager.setDiffusersWorkerExitMs(25);
    assert.equal(engineManager.DIFFUSERS_STOP_GRACE_MS, 5000);
    assert.equal(engineManager.DIFFUSERS_WORKER_EXIT_MS, 5000);
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
    engineManager.setDiffusersStopGraceMs(60_000);
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    resolveRun({ text: 'late pixels', imagePaths: [allocated.absolutePath] });
    assert.equal(await pending, STOP_MARKER);
    await engineManager.expireDiffusersStopGrace();
    assert.equal(closes.length, 0);
    assert.equal(stops, 1);
    assert.equal(exitSettled, false);
    assert.equal(engineManager.getStatus().phase, 'idle');

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
    assert.deepEqual(closes, [{ final: false }]);
    assert.equal(stops, 1);
    assert.equal(exitSettled, false);
    // The close that settled the prompt is only the start of the kill. The
    // worker-exit timer is unref'd, so wait until reload finishes before the
    // next test sends a prompt.
    await waitFor(() => engineManager.getStatus().phase === 'idle');
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

    engineManager.setDiffusersStopGraceMs(60_000);
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

    await awaitAlive(engineManager.expireDiffusersStopGrace());
    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    await awaitAlive(enteredReload);
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
        if (!/Engine is busy\./.test(err.message) || Date.now() - start > 15000) {
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
    assert.equal(stops, 0);
    assert.equal(closes.length, 0);
    // A grace armed by this cancel would kill on expire. A finished run must not arm one.
    await engineManager.expireDiffusersStopGrace();
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
    await waitFor(() => sender.events.some((event) => event.type === 'canceled'));
    assert.equal(exitSettled, false);
    assert.deepEqual(closes, [{ final: false }]);
    await turn;
    assert.equal(exitSettled, false);
    await waitFor(() => engineManager.getStatus().phase !== 'loading');
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

  function modelInit(modelId, extra = {}) {
    return {
      modelId,
      modelsCacheDir: cache,
      sessionsRoot: sessions,
      resourcesRoot: path.join(root, 'Resources'),
      outputsRoot: path.join(root, 'Outputs'),
      ...extra,
    };
  }

  it('rejects model changes during the stop and skips the kill if the model changed', async () => {
    const otherDir = path.join(cache, 'org', 'other');
    await fs.mkdir(otherDir, { recursive: true });
    await fs.writeFile(path.join(otherDir, 'model.safetensors'), 'weights');
    await fs.writeFile(path.join(otherDir, 'README.md'), '---\npipeline_tag: text-generation\n---\n');
    const allocated = await allocateSessionImagePath(sessions, 'switch.json');
    // Longer than any reinitialize in this test. The grace used to be 400 ms
    // and then the test slept 500 ms. Under CPU load reinitialize itself
    // exceeded 400 ms, the timer closed the worker while the model was still
    // org/diffusers, and closes.length was 1. Expire after the id has changed.
    engineManager.setDiffusersStopGraceMs(60_000);
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    assert.equal(engineManager.getStatus().phase, 'loading');
    await assert.rejects(engineManager.ejectModel(), /Engine is busy\./);
    await assert.rejects(engineManager.reinitialize(modelInit('org/diffusers')), /Engine is busy\./);
    await assert.rejects(engineManager.initialize(modelInit('org/diffusers')), /Engine is busy\./);
    assert.equal(engineManager.getStatus().modelId, 'org/diffusers');
    assert.equal(engineManager.getStatus().engineId, 'huggingface');

    await engineManager.reinitialize(
      modelInit('org/other', { allowDuringDiffusersStop: true })
    );
    assert.equal(engineManager.getStatus().modelId, 'org/other');
    await engineManager.expireDiffusersStopGrace();
    assert.equal(closes.length, 0);
    assert.equal(engineManager.getStatus().phase, 'idle');
    resolveRun({ text: '', imagePaths: [allocated.absolutePath] });
    assert.equal(await pending, STOP_MARKER);
    await engineManager.reinitialize(modelInit('org/diffusers'));
    assert.equal(engineManager.getStatus().modelId, 'org/diffusers');
    assert.equal(engineManager.getStatus().pipelineTag, 'text-to-image');
  });

  it('reports loading for the whole stop and the reload-model progress events', async () => {
    const events = [];
    let releaseReload = () => {};
    let markEntered;
    const entered = new Promise((resolve) => {
      markEntered = resolve;
    });
    const originalCreate = hf.chatbotCreate;
    hf.chatbotCreate = async (_modelId, options) => {
      if (options && typeof options.onProgress === 'function') {
        options.onProgress({ loaded: 1, total: 2, percent: 50 });
      }
    };
    engineManager.setDiffusersKillRecovery(async () => {
      markEntered();
      await new Promise((resolve) => {
        releaseReload = resolve;
      });
      await engineManager.reinitialize(
        modelInit('org/diffusers', {
          allowDuringDiffusersStop: true,
          onProgress: (info) => events.push(info),
        })
      );
    });
    try {
      const allocated = await allocateSessionImagePath(sessions, 'loading.json');
      const pending = engineManager.sendPrompt('a red fox', {
        outputPath: allocated.absolutePath,
        reservationId: allocated.reservationId,
      });
      engineManager.cancelGeneration();
      assert.equal(engineManager.getStatus().phase, 'loading');
      assert.equal(engineManager.getStatus().phase === 'idle', false);
      await awaitAlive(entered);
      assert.equal(engineManager.getStatus().phase, 'loading');
      releaseReload();
      const raced = await racePrompt(pending);
      assert.equal(raced.kind, 'prompt');
      await waitFor(() => engineManager.getStatus().phase === 'idle');
      assert.equal(
        events.some((info) => info.phase === 'loading' && info.status === 'starting'),
        true
      );
      assert.equal(
        events.some((info) => info.phase === 'loading' && info.status === 'progress'),
        true
      );
      assert.equal(
        events.some((info) => info.phase === 'loading' && info.status === 'complete'),
        true
      );
    } finally {
      hf.chatbotCreate = originalCreate;
      releaseReload();
    }
  });

  it('removes offload only after a late worker exit', async () => {
    // The close promise is what ends the wait. A short exit timer can fire
    // during the file reads under CPU load and delete the offload first.
    engineManager.setDiffusersWorkerExitMs(60_000);
    const allocated = await allocateSessionImagePath(sessions, 'late-exit.json');
    const offloadDirs = await plantOffload();
    let reloaded = false;
    engineManager.setDiffusersKillRecovery(() => {
      reloaded = true;
    });
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    assert.equal(raced.value, STOP_MARKER);
    assert.equal(exitSettled, false);
    assert.equal(reloaded, false);
    for (const dir of offloadDirs) {
      assert.equal(await fs.readFile(path.join(dir, 'weight.bin'), 'utf8'), 'w');
    }
    exitResolve();
    await waitFor(() => reloaded);
    assert.equal(exitSettled, true);
    for (const dir of offloadDirs) {
      await assert.rejects(fs.stat(dir));
    }
  });

  it('retries offload removal once when the first delete fails', async () => {
    const base = path.join(root, 'retry-offload');
    engineManager.setDiffusersOffloadBaseDirs([base]);
    const dir = path.join(base, '.glaux-t2i-offload', `pid-${WORKER_PID}`);
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir, 'weight.bin'), 'w');
    const fsPromises = require('fs').promises;
    const originalRm = fsPromises.rm;
    let failures = 0;
    fsPromises.rm = async function rmOnce(target, opts) {
      if (path.resolve(String(target)) === path.resolve(dir) && failures < 1) {
        failures += 1;
        const err = new Error('busy');
        err.code = 'EBUSY';
        throw err;
      }
      return originalRm.call(fsPromises, target, opts);
    };
    try {
      const allocated = await allocateSessionImagePath(sessions, 'retry.json');
      let reloaded = false;
      engineManager.setDiffusersKillRecovery(() => {
        reloaded = true;
      });
      const pending = engineManager.sendPrompt('a red fox', {
        outputPath: allocated.absolutePath,
        reservationId: allocated.reservationId,
      });
      engineManager.cancelGeneration();
      const raced = await racePrompt(pending);
      assert.equal(raced.kind, 'prompt');
      await waitFor(() => reloaded);
      assert.equal(failures, 1);
      await assert.rejects(fs.stat(dir));
    } finally {
      fsPromises.rm = originalRm;
      engineManager.setDiffusersOffloadBaseDirs(offloadBases);
    }
  });

  it('builds offload paths from the armed model and skips /var/tmp on Windows', () => {
    engineManager.setDiffusersOffloadBaseDirs(null);
    try {
      const linux = engineManager.killedOffloadPaths(WORKER_PID, 'org/diffusers', 'linux');
      const win = engineManager.killedOffloadPaths(WORKER_PID, 'org/diffusers', 'win32');
      const other = engineManager.killedOffloadPaths(WORKER_PID, 'otherorg/model', 'linux');
      const folder = path.join('.glaux-t2i-offload', `pid-${WORKER_PID}`);
      assert.ok(linux.some((dir) => dir === path.resolve('/var/tmp', folder)));
      assert.equal(
        win.some((dir) => dir.startsWith(path.resolve('/var/tmp'))),
        false
      );
      assert.ok(linux.some((dir) => dir === path.resolve(cache, 'org', folder)));
      assert.ok(other.some((dir) => dir === path.resolve(cache, 'otherorg', folder)));
      assert.equal(other.some((dir) => dir === path.resolve(cache, 'org', folder)), false);
      assert.deepEqual(engineManager.killedOffloadPaths(0, 'org/diffusers', 'linux'), []);
      assert.deepEqual(engineManager.killedOffloadPaths(1.5, 'org/diffusers', 'linux'), []);
      assert.equal(
        linux.some((dir) => dir.startsWith('/var/tmp') && !dir.includes('.glaux-t2i-offload')),
        false
      );
    } finally {
      engineManager.setDiffusersOffloadBaseDirs(offloadBases);
    }
  });

  it('joins the stop reload and the stream-timeout reload into one', async () => {
    const orig = engineManager.joinModelReload.bind(engineManager);
    const seen = [];
    engineManager.joinModelReload = (fn) => {
      seen.push(fn);
      return Promise.resolve();
    };
    try {
      await recoveryInstalls[0]();
      await requestWorkerReload('Failed to reinitialize inference engine after stream timeout:');
      assert.equal(seen.length, 2);
    } finally {
      engineManager.joinModelReload = orig;
    }
    let runs = 0;
    let release;
    const hold = new Promise((resolve) => {
      release = resolve;
    });
    const first = orig(async () => {
      runs += 1;
      await hold;
    });
    const second = orig(async () => {
      runs += 1;
    });
    await Promise.resolve();
    assert.equal(runs, 1);
    release();
    await Promise.all([first, second]);
    assert.equal(runs, 1);
  });

  function trackDownload() {
    let calls = 0;
    hf.downloadModel = (_modelId, options) => {
      calls += 1;
      if (calls === 1) {
        return new Promise((_resolve, reject) => {
          rejectDownload = () => reject(new Error('Engine closed'));
        });
      }
      if (options && typeof options.onProgress === 'function') {
        options.onProgress({ status: 'progress', loaded: 3, total: 4, file: 'download' });
      }
      return Promise.resolve();
    };
    return {
      calls: () => calls,
    };
  }

  async function startParkedDownload(modelId) {
    const tracked = trackDownload();
    const events = [];
    const download = engineManager.downloadModel({
      modelId,
      modelsCacheDir: cache,
      onProgress: (info) => events.push(info),
    });
    await waitFor(() => tracked.calls() === 1);
    return { tracked, events, download };
  }

  it('resumes a download aborted by the kill after reload', async () => {
    const { tracked, events, download } = await startParkedDownload('org/side');
    let releaseReload = () => {};
    let markEntered;
    const entered = new Promise((resolve) => {
      markEntered = resolve;
    });
    engineManager.setDiffusersKillRecovery(() => {
      markEntered();
      return new Promise((resolve) => {
        releaseReload = resolve;
      });
    });
    const allocated = await allocateSessionImagePath(sessions, 'resume.json');
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    await awaitAlive(entered);
    assert.equal(tracked.calls(), 1);
    releaseReload();
    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    await download;
    assert.equal(tracked.calls(), 2);
    assert.equal(
      events.some((info) => info.phase === 'download' && info.status === 'starting'),
      true
    );
    assert.equal(
      events.some((info) => info.phase === 'download' && info.status === 'complete'),
      true
    );
    assert.equal(
      events.some((info) => info.status === 'error'),
      false
    );
  });

  it('does not resume a download the user cancelled during reload', async () => {
    const { tracked, download } = await startParkedDownload('org/cancel-me');
    let releaseReload = () => {};
    let markEntered;
    const entered = new Promise((resolve) => {
      markEntered = resolve;
    });
    engineManager.setDiffusersKillRecovery(() => {
      markEntered();
      return new Promise((resolve) => {
        releaseReload = resolve;
      });
    });
    try {
      const allocated = await allocateSessionImagePath(sessions, 'cancel-dl.json');
      const pending = engineManager.sendPrompt('a red fox', {
        outputPath: allocated.absolutePath,
        reservationId: allocated.reservationId,
      });
      engineManager.cancelGeneration();
      await awaitAlive(entered);
      const cancelled = download.then(
        () => null,
        (err) => err
      );
      await engineManager.cancelModelDownload('org/cancel-me', { modelsCacheDir: cache });
      const err = await cancelled;
      assert.equal(err.code, 'DOWNLOAD_CANCELLED');
      releaseReload();
      await racePrompt(pending);
      await waitFor(() => engineManager.getStatus().phase === 'idle');
      assert.equal(tracked.calls(), 1);
    } finally {
      releaseReload();
    }
  });

  it('does not resume a download when reload fails or the model was deleted', async () => {
    const failed = await startParkedDownload('org/fail-dl');
    engineManager.setDiffusersKillRecovery(async () => ({ ok: false }));
    const failedPath = await allocateSessionImagePath(sessions, 'fail-dl.json');
    const failedPrompt = engineManager.sendPrompt('a red fox', {
      outputPath: failedPath.absolutePath,
      reservationId: failedPath.reservationId,
    });
    engineManager.cancelGeneration();
    await awaitAlive(assert.rejects(failed.download, /Engine closed/));
    const failedRace = await racePrompt(failedPrompt);
    assert.equal(failedRace.kind, 'prompt');
    assert.equal(failed.tracked.calls(), 1);

    const deletedDir = path.join(cache, 'org', 'deleted-dl');
    await fs.mkdir(deletedDir, { recursive: true });
    const deleted = await startParkedDownload('org/deleted-dl');
    let releaseReload = () => {};
    let markEntered;
    const entered = new Promise((resolve) => {
      markEntered = resolve;
    });
    engineManager.setDiffusersKillRecovery(() => {
      markEntered();
      return new Promise((resolve) => {
        releaseReload = resolve;
      });
    });
    const deletedPath = await allocateSessionImagePath(sessions, 'deleted-dl.json');
    const deletedPrompt = engineManager.sendPrompt('a blue fox', {
      outputPath: deletedPath.absolutePath,
      reservationId: deletedPath.reservationId,
    });
    engineManager.cancelGeneration();
    await awaitAlive(entered);
    await fs.rm(deletedDir, { recursive: true, force: true });
    releaseReload();
    await assert.rejects(deleted.download, /Engine closed/);
    const deletedRace = await racePrompt(deletedPrompt);
    assert.equal(deletedRace.kind, 'prompt');
    assert.equal(deleted.tracked.calls(), 1);
  });

  it('clears the active download rpc id when the worker is closed', async () => {
    hf.setActiveDownloadRpcId(9);
    assert.equal(hf.activeDownloadRpcId(), 9);
    await original.close({ final: false });
    assert.equal(hf.activeDownloadRpcId(), null);
  });

  async function beginHungResume(modelId) {
    let calls = 0;
    const events = [];
    let resumeResolve = null;
    let resumeReject = null;
    hf.downloadModel = () => {
      calls += 1;
      return new Promise((resolve, reject) => {
        if (calls === 1) {
          rejectDownload = () => reject(new Error('Engine closed'));
          return;
        }
        resumeResolve = resolve;
        resumeReject = reject;
        rejectDownload = () => reject(new Error('Engine closed'));
      });
    };
    const download = engineManager.downloadModel({
      modelId,
      modelsCacheDir: cache,
      onProgress: (info) => events.push(info),
    });
    await waitFor(() => calls === 1);
    let markEntered;
    const entered = new Promise((resolve) => {
      markEntered = resolve;
    });
    engineManager.setDiffusersKillRecovery(() => {
      markEntered();
    });
    const allocated = await allocateSessionImagePath(sessions, `${modelId.split('/').pop()}.json`);
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    await awaitAlive(entered);
    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    assert.equal(raced.value, STOP_MARKER);
    await waitFor(() => engineManager.getStatus().phase === 'idle' && calls >= 2);
    return {
      calls: () => calls,
      events,
      download,
      finish: () => resumeResolve(),
      failResume: (err) => resumeReject(err),
    };
  }

  it('drops the busy gate while a resumed download is still running', async () => {
    const unhandled = [];
    const onUnhandled = (err) => unhandled.push(err);
    process.on('unhandledRejection', onUnhandled);
    const logged = [];
    const origError = console.error;
    console.error = (...args) => {
      logged.push(args.map((part) => String(part && part.message ? part.message : part)).join(' '));
    };
    try {
      const hung = await beginHungResume('org/resume-hang');
      const rendered = await buildEngineStatusForRenderer();
      assert.equal(rendered.phase, 'idle');
      assert.equal(rendered.ready, true);
      assert.equal(rendered.modelId, 'org/diffusers');
      let settled = false;
      hung.download.then(
        () => {
          settled = true;
        },
        () => {
          settled = true;
        }
      );

      await engineManager.reinitialize(modelInit('org/diffusers'));
      assert.equal(engineManager.getStatus().modelId, 'org/diffusers');
      assert.equal(engineManager.getStatus().phase, 'idle');
      await assertStillPending(hung.download);
      assert.equal(settled, false);

      hung.finish();
      await hung.download;
      assert.equal(
        hung.events.some((info) => info.phase === 'download' && info.status === 'complete'),
        true
      );
      assert.equal(
        hung.events.some((info) => info.status === 'error'),
        false
      );

      const hungCancel = await beginHungResume('org/resume-cancel');
      const cancelIds = [];
      let releaseCancelRpc = () => {};
      hf.cancelDownloadModel = (id) => {
        cancelIds.push(id);
        return new Promise((resolve) => {
          releaseCancelRpc = resolve;
        });
      };
      await engineManager.ejectModel();
      assert.equal(engineManager.getStatus().modelId, null);
      assert.equal(engineManager.getStatus().phase, 'idle');
      let cancelSettled = false;
      const cancelled = hungCancel.download.then(
        () => {
          cancelSettled = true;
          return null;
        },
        (err) => {
          cancelSettled = true;
          return err;
        }
      );
      const cancelCall = engineManager.cancelModelDownload('org/resume-cancel', {
        modelsCacheDir: cache,
      });
      await assertStillPending(cancelled);
      assert.equal(cancelSettled, false);
      assert.deepEqual(cancelIds, ['org/resume-cancel']);
      releaseCancelRpc();
      await cancelCall;
      await assertStillPending(hungCancel.download);
      assert.equal(cancelSettled, false);
      hungCancel.failResume(new Error('Engine closed'));
      const err = await cancelled;
      assert.equal(err.code, 'DOWNLOAD_CANCELLED');
      assert.equal(
        hungCancel.events.some((info) => info.status === 'error'),
        false
      );
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(unhandled.length, 0);
      assert.equal(
        logged.some((line) => /Failed to resume downloads/.test(line)),
        false
      );

      hf.cancelDownloadModel = async () => {};
      await engineManager.initialize(modelInit('org/diffusers'));
    } finally {
      console.error = origError;
      process.removeListener('unhandledRejection', onUnhandled);
    }
  });

  it('resumes a download across two hung stops and completes once', async () => {
    let calls = 0;
    let releaseFinal = () => {};
    const events = [];
    hf.downloadModel = () => {
      calls += 1;
      return new Promise((resolve, reject) => {
        if (calls < 3) {
          rejectDownload = () => reject(new Error('Engine closed'));
          return;
        }
        releaseFinal = resolve;
      });
    };
    const download = engineManager.downloadModel({
      modelId: 'org/two-stops',
      modelsCacheDir: cache,
      onProgress: (info) => events.push(info),
    });
    const settled = download.then(
      () => 'ok',
      (err) => err
    );
    await waitFor(() => calls === 1);

    async function hungStop(name) {
      const allocated = await allocateSessionImagePath(sessions, `${name}.json`);
      const pending = engineManager.sendPrompt('a red fox', {
        outputPath: allocated.absolutePath,
        reservationId: allocated.reservationId,
      });
      engineManager.cancelGeneration();
      const raced = await racePrompt(pending);
      assert.equal(raced.kind, 'prompt');
      assert.equal(raced.value, STOP_MARKER);
    }

    await hungStop('two-stops-1');
    await waitFor(() => calls === 2 && engineManager.getStatus().phase === 'idle');
    await assertStillPending(settled);

    await hungStop('two-stops-2');
    await waitFor(() => calls === 3 && engineManager.getStatus().phase === 'idle');
    await assertStillPending(settled);
    releaseFinal();
    assert.equal(await settled, 'ok');
    assert.equal(calls, 3);
    assert.equal(
      events.some((info) => info.phase === 'download' && info.status === 'complete'),
      true
    );
    assert.equal(
      events.some((info) => info.status === 'error'),
      false
    );
  });

  it('parks a new download started during a second stop while a resume is active', async () => {
    const inflight = [];
    hf.downloadModel = (modelId) =>
      new Promise((resolve, reject) => {
        const slot = { modelId, resolve, reject, done: false };
        inflight.push(slot);
        rejectDownload = () => {
          for (const item of inflight) {
            if (!item.done) {
              item.done = true;
              item.reject(new Error('Engine closed'));
            }
          }
        };
      });
    const events = { a: [], b: [] };
    const downloadA = engineManager.downloadModel({
      modelId: 'org/resume-a',
      modelsCacheDir: cache,
      onProgress: (info) => events.a.push(info),
    });
    const settledA = downloadA.then(
      () => 'ok',
      (err) => err
    );
    await waitFor(() => inflight.length === 1);
    const first = await allocateSessionImagePath(sessions, 'second-grace-1.json');
    const firstPrompt = engineManager.sendPrompt('a red fox', {
      outputPath: first.absolutePath,
      reservationId: first.reservationId,
    });
    engineManager.cancelGeneration();
    const firstRace = await racePrompt(firstPrompt);
    assert.equal(firstRace.kind, 'prompt');
    await waitFor(() => inflight.length === 2 && engineManager.getStatus().phase === 'idle');

    engineManager.setDiffusersStopGraceMs(60_000);
    const second = await allocateSessionImagePath(sessions, 'second-grace-2.json');
    const secondPrompt = engineManager.sendPrompt('a blue fox', {
      outputPath: second.absolutePath,
      reservationId: second.reservationId,
    });
    engineManager.cancelGeneration();
    assert.equal(engineManager.getStatus().phase, 'loading');
    assert.equal(closes.length, 1);
    const downloadB = engineManager.downloadModel({
      modelId: 'org/during-second',
      modelsCacheDir: cache,
      onProgress: (info) => events.b.push(info),
    });
    const settledB = downloadB.then(
      () => 'ok',
      (err) => err
    );
    await waitFor(() => inflight.length === 3);
    assert.equal(closes.length, 1);
    await assertStillPending(settledB);
    await awaitAlive(engineManager.expireDiffusersStopGrace());

    await waitFor(
      () => engineManager.getStatus().phase === 'idle' && inflight.filter((item) => item.modelId === 'org/resume-a').length === 3
    );
    const thirdA = inflight.filter((item) => item.modelId === 'org/resume-a')[2];
    thirdA.done = true;
    thirdA.resolve();
    await waitFor(
      () => inflight.filter((item) => item.modelId === 'org/during-second').length === 2
    );
    const secondB = inflight.filter((item) => item.modelId === 'org/during-second')[1];
    secondB.done = true;
    secondB.resolve();
    assert.equal(await settledA, 'ok');
    assert.equal(await settledB, 'ok');
    assert.equal(
      events.a.some((info) => info.status === 'error') || events.b.some((info) => info.status === 'error'),
      false
    );
    assert.equal(
      events.b.some((info) => info.phase === 'download' && info.status === 'complete'),
      true
    );
  });

  it('rejects parked downloads when shutdown arrives during the worker exit wait', async () => {
    await delay(0);
    const timersBefore = referencedTimeouts().length;
    engineManager.setDiffusersStopGraceMs(60_000);
    engineManager.setDiffusersWorkerExitMs(60_000);
    let calls = 0;
    let reloaded = false;
    hf.downloadModel = () => {
      calls += 1;
      return new Promise((_resolve, reject) => {
        rejectDownload = () => reject(new Error('Engine closed'));
      });
    };
    engineManager.setDiffusersKillRecovery(() => {
      reloaded = true;
    });
    const events = [];
    const download = engineManager.downloadModel({
      modelId: 'org/quit-during-exit',
      modelsCacheDir: cache,
      onProgress: (info) => events.push(info),
    });
    const settled = download.then(
      () => ({ kind: 'resolved' }),
      (err) => ({ kind: 'rejected', err })
    );
    await waitFor(() => calls === 1);
    const offloadDirs = await plantOffload();
    const allocated = await allocateSessionImagePath(sessions, 'quit-exit.json');
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    const killing = engineManager.expireDiffusersStopGrace();
    await waitFor(() => closes.length === 1);
    await assertStillPending(settled);
    for (const dir of offloadDirs) {
      assert.equal(await fs.readFile(path.join(dir, 'weight.bin'), 'utf8'), 'w');
    }
    assert.equal(exitSettled, false);

    const outcome = await Promise.race([
      engineManager.shutdown({ hfDestroyTimeoutMs: 20 }).then(() => 'done'),
      delay(15000).then(() => 'hung'),
    ]);
    assert.equal(outcome, 'done');
    assert.equal(exitSettled, false);
    assert.equal(reloaded, false);
    assert.equal(engineManager.getStatus().phase, 'idle');
    assert.equal(engineManager.getStatus().modelId, null);
    const result = await settled;
    assert.equal(result.kind, 'rejected');
    assert.match(result.err.message, /Engine closed/);
    assert.equal(calls, 1);
    assert.equal(
      events.some((info) => info.status === 'error'),
      false
    );
    for (const dir of offloadDirs) {
      assert.equal(await fs.readFile(path.join(dir, 'weight.bin'), 'utf8'), 'w');
    }

    // Shutdown returns while killHungDiffusersRun is still inside the exit
    // wait. The offload delete happens after that wait. Resolving the close
    // promise ends it. Leaving the 60s timer armed would let the kill reload
    // after initialize() clears shuttingDown.
    exitResolve();
    await awaitAlive(killing);
    for (const dir of offloadDirs) {
      await assert.rejects(fs.stat(dir));
    }
    assert.equal(reloaded, false);
    assert.equal(engineManager.getStatus().phase, 'idle');
    await waitFor(() => referencedTimeouts().length <= timersBefore);

    await engineManager.initialize(modelInit('org/diffusers'));
    state.engineBootstrapped = true;
    assert.equal(engineManager.getStatus().modelId, 'org/diffusers');
    assert.equal(engineManager.getStatus().phase, 'idle');
  });

  it('parks a download started during the grace and resumes it after reload', async () => {
    // The download must start while the grace is still armed. A 400 ms grace
    // could fire during downloadModel under CPU load and close the worker
    // before this asserts closes.length === 0. Expire once the download is parked.
    engineManager.setDiffusersStopGraceMs(60_000);
    let calls = 0;
    hf.downloadModel = (_modelId, options) => {
      calls += 1;
      if (calls === 1) {
        return new Promise((_resolve, reject) => {
          rejectDownload = () => reject(new Error('Engine closed'));
        });
      }
      if (options && typeof options.onProgress === 'function') {
        options.onProgress({ status: 'progress', loaded: 1, total: 2, file: 'grace' });
      }
      return Promise.resolve();
    };
    const allocated = await allocateSessionImagePath(sessions, 'during-grace.json');
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    assert.equal(engineManager.getStatus().phase, 'loading');
    const events = [];
    const download = engineManager.downloadModel({
      modelId: 'org/during-grace',
      modelsCacheDir: cache,
      onProgress: (info) => events.push(info),
    });
    await waitFor(() => calls === 1);
    assert.equal(closes.length, 0);
    await awaitAlive(engineManager.expireDiffusersStopGrace());
    await awaitAlive(download, 15000);
    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    assert.equal(calls, 2);
    assert.equal(
      events.some((info) => info.phase === 'download' && info.status === 'complete'),
      true
    );
    assert.equal(
      events.some((info) => info.status === 'error'),
      false
    );
    await waitFor(() => engineManager.getStatus().phase === 'idle');
  });

  it('parks a kill-error download started during reload and resumes it', async () => {
    let releaseReload = () => {};
    let markEntered;
    const entered = new Promise((resolve) => {
      markEntered = resolve;
    });
    engineManager.setDiffusersKillRecovery(() => {
      markEntered();
      return new Promise((resolve) => {
        releaseReload = resolve;
      });
    });
    const allocated = await allocateSessionImagePath(sessions, 'during-reload.json');
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    await awaitAlive(entered);
    assert.equal(engineManager.getStatus().phase, 'loading');
    let calls = 0;
    hf.downloadModel = (_modelId, options) => {
      calls += 1;
      if (calls === 1) {
        return Promise.reject(new Error('Engine closed'));
      }
      if (options && typeof options.onProgress === 'function') {
        options.onProgress({ status: 'progress', loaded: 2, total: 2, file: 'reload' });
      }
      return Promise.resolve();
    };
    // Parking awaits a stat of the model directory. Releasing the reload
    // before that stat returns surfaces the kill error instead of resuming.
    const originalStat = fs.stat;
    let parkStatDone = false;
    fs.stat = async function statUntilParked(target, ...rest) {
      try {
        return await originalStat.call(fs, target, ...rest);
      } finally {
        if (String(target).includes('during-reload')) parkStatDone = true;
      }
    };
    const events = [];
    try {
      const download = engineManager.downloadModel({
        modelId: 'org/during-reload',
        modelsCacheDir: cache,
        onProgress: (info) => events.push(info),
      });
      const settled = download.then(
        () => 'resolved',
        (err) => err
      );
      await waitFor(() => parkStatDone);
      await assertStillPending(settled);
      assert.equal(calls, 1);
      releaseReload();
      assert.equal(await awaitAlive(settled, 15000), 'resolved');
      const raced = await racePrompt(pending);
      assert.equal(raced.kind, 'prompt');
      assert.equal(calls, 2);
      assert.equal(
        events.some((info) => info.phase === 'download' && info.status === 'complete'),
        true
      );
      assert.equal(
        events.some((info) => info.status === 'error'),
        false
      );
    } finally {
      fs.stat = originalStat;
    }
  });

  it('rejects a download that is still parking when the stop procedure ends', async () => {
    let releaseReload = () => {};
    let markEntered;
    const entered = new Promise((resolve) => {
      markEntered = resolve;
    });
    engineManager.setDiffusersKillRecovery(() => {
      markEntered();
      return new Promise((resolve) => {
        releaseReload = resolve;
      });
    });
    const allocated = await allocateSessionImagePath(sessions, 'late-park.json');
    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    await awaitAlive(entered);
    const fsPromises = require('fs').promises;
    const originalStat = fsPromises.stat;
    let releaseStat = () => {};
    let markStat;
    const statHeld = new Promise((resolve) => {
      markStat = resolve;
    });
    fsPromises.stat = async (target, ...rest) => {
      if (String(target).includes('race-dl')) {
        markStat();
        await new Promise((resolve) => {
          releaseStat = resolve;
        });
      }
      return originalStat.call(fsPromises, target, ...rest);
    };
    let calls = 0;
    hf.downloadModel = () => {
      calls += 1;
      return Promise.reject(new Error('Engine closed'));
    };
    try {
      const events = [];
      const download = engineManager.downloadModel({
        modelId: 'org/race-dl',
        modelsCacheDir: cache,
        onProgress: (info) => events.push(info),
      });
      const settled = download.then(
        () => null,
        (err) => err
      );
      await awaitAlive(statHeld);
      releaseReload();
      await waitFor(() => engineManager.getStatus().phase === 'idle');
      assert.equal(calls, 1);
      releaseStat();
      const err = await settled;
      assert.match(err.message, /Engine closed/);
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(calls, 1);
      assert.equal(
        events.some((info) => info.status === 'error'),
        false
      );
      const raced = await racePrompt(pending);
      assert.equal(raced.kind, 'prompt');
    } finally {
      fsPromises.stat = originalStat;
      releaseStat();
      releaseReload();
    }
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
      while (!engineManager.isShuttingDown() && Date.now() - start < 15000) {
        await delay(5);
      }
      sawShutdown = engineManager.isShuttingDown();
    });

    const pending = engineManager.sendPrompt('a red fox', {
      outputPath: allocated.absolutePath,
      reservationId: allocated.reservationId,
    });
    engineManager.cancelGeneration();
    await awaitAlive(entered);
    const raced = await racePrompt(pending);
    assert.equal(raced.kind, 'prompt');
    assert.equal(exitSettled, false);

    const outcome = await Promise.race([
      engineManager.shutdown({ hfDestroyTimeoutMs: 20 }).then(() => 'done'),
      delay(15000).then(() => 'hung'),
    ]);
    assert.equal(outcome, 'done');
    assert.equal(exitSettled, false);
    assert.equal(engineManager.getStatus().phase, 'idle');
    await waitFor(() => sawShutdown);
  });
});
