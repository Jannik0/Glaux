'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const engineManager = require('../engines/engineManager');
const { MAX_TRACKED_REQUEST_RUNS } = engineManager;
const contextManager = require('../engines/contextManager');
const llama = require('../engines/llamacpp/engine');
const { STOP_MARKER, withStopMarker } = require('../engines/common/stopMarker');
const state = require('../src/main/state');
const { deliverSendMessage, handleEngineSendMessage } = require('../src/main/domains/engineBridge');
const { clearPendingSessionFilename, getPendingSessionFilename } = require('../src/main/domains/sessions');

function contextShape(messages) {
  return messages.map((msg) => ({
    role: msg.role,
    text: (msg.content || [])
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join('\n'),
  }));
}

describe('sendPrompt chat', { concurrency: 1 }, () => {
  const originalCreate = llama.chatbotCreate;
  const originalRun = llama.runChat;
  let root;
  let sessions;

  before(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-chat-send-'));
    sessions = path.join(root, 'Sessions');
    const resources = path.join(root, 'Resources');
    const cache = path.join(root, 'Models');
    const model = path.join(cache, 'org', 'chat');
    await fs.mkdir(sessions);
    await fs.mkdir(resources);
    await fs.mkdir(model, { recursive: true });
    await fs.writeFile(path.join(model, 'model.gguf'), 'gguf');
    await fs.writeFile(path.join(model, 'README.md'), '---\npipeline_tag: text-generation\n---\n');
    llama.chatbotCreate = async () => {};
    await engineManager.initialize({
      modelId: 'org/chat',
      modelsCacheDir: cache,
      sessionsRoot: sessions,
      resourcesRoot: resources,
      outputsRoot: path.join(root, 'Outputs'),
    });
  });

  after(async () => {
    llama.chatbotCreate = originalCreate;
    llama.runChat = originalRun;
    contextManager.clear();
    state.activeSessionFilename = null;
    state.engineBootstrapped = false;
    clearPendingSessionFilename();
    await engineManager.shutdown({ hfDestroyTimeoutMs: 20 });
    if (root) {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it('keeps a stopped reply when nothing newer was sent', async () => {
    contextManager.clear();
    let release;
    let markStarted;
    const started = new Promise((resolve) => {
      markStarted = resolve;
    });
    llama.runChat = (_model, _thinking, _message, opts) =>
      new Promise((resolve) => {
        opts.onToken('Hello');
        markStarted();
        release = () => resolve('Hello');
      });
    const pending = engineManager.sendPrompt('hi');
    await started;
    engineManager.cancelGeneration();
    release();
    assert.equal(await pending, withStopMarker('Hello'));
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: withStopMarker('Hello') },
    ]);
    assert.equal(engineManager.getStatus().phase, 'idle');
  });

  it('inserts the older stop marker after that user when a newer prompt is already running', async () => {
    contextManager.clear();
    let releaseFirst;
    let markFirstStarted;
    const firstStarted = new Promise((resolve) => {
      markFirstStarted = resolve;
    });
    llama.runChat = (_model, _thinking, _message, opts) =>
      new Promise((resolve) => {
        opts.onToken('partial');
        markFirstStarted();
        releaseFirst = () => resolve('partial');
      });
    const first = engineManager.sendPrompt('first prompt');
    await firstStarted;
    engineManager.cancelGeneration();

    let releaseSecond;
    let markSecondStarted;
    let secondMessages;
    const secondStarted = new Promise((resolve) => {
      markSecondStarted = resolve;
    });
    llama.runChat = (_model, _thinking, _message, opts) =>
      new Promise((resolve) => {
        secondMessages = opts.messages;
        markSecondStarted();
        releaseSecond = () => resolve('second reply');
      });
    const second = engineManager.sendPrompt('second prompt');
    await secondStarted;
    assert.deepEqual(
      secondMessages.map((msg) => msg.role),
      ['user', 'assistant', 'user']
    );
    assert.equal(
      secondMessages[1].content.filter((part) => part.type === 'text').map((part) => part.text).join('\n'),
      withStopMarker('partial')
    );
    releaseFirst();
    assert.equal(await first, withStopMarker('partial'));
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'first prompt' },
      { role: 'assistant', text: withStopMarker('partial') },
      { role: 'user', text: 'second prompt' },
    ]);
    releaseSecond();
    assert.equal(await second, 'second reply');
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'first prompt' },
      { role: 'assistant', text: withStopMarker('partial') },
      { role: 'user', text: 'second prompt' },
      { role: 'assistant', text: 'second reply' },
    ]);
  });

  it('inserts a stop marker when the older run throws or returns empty', async () => {
    contextManager.clear();
    let releaseFirst;
    let markFirstStarted;
    const firstStarted = new Promise((resolve) => {
      markFirstStarted = resolve;
    });
    llama.runChat = (_model, _thinking, _message, opts) =>
      new Promise((_resolve, reject) => {
        opts.onToken('partial');
        markFirstStarted();
        releaseFirst = () => reject(new Error('llama failed'));
      });
    const first = engineManager.sendPrompt('first prompt');
    await firstStarted;
    engineManager.cancelGeneration();
    let releaseSecond;
    const secondStarted = new Promise((resolve) => {
      llama.runChat = () =>
        new Promise((resolveReply) => {
          resolve();
          releaseSecond = () => resolveReply('second reply');
        });
    });
    const second = engineManager.sendPrompt('second prompt');
    await secondStarted;
    releaseFirst();
    assert.equal(await first, withStopMarker('partial'));
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'first prompt' },
      { role: 'assistant', text: withStopMarker('partial') },
      { role: 'user', text: 'second prompt' },
    ]);
    releaseSecond();
    await second;

    contextManager.clear();
    const emptyStarted = new Promise((resolve) => {
      llama.runChat = () =>
        new Promise((resolveReply) => {
          resolve();
          releaseFirst = () => resolveReply('');
        });
    });
    const older = engineManager.sendPrompt('empty prompt');
    await emptyStarted;
    engineManager.cancelGeneration();
    const newerStarted = new Promise((resolve) => {
      llama.runChat = () =>
        new Promise((resolveReply) => {
          resolve();
          releaseSecond = () => resolveReply('kept');
        });
    });
    const newer = engineManager.sendPrompt('kept prompt');
    await newerStarted;
    releaseFirst();
    assert.equal(await older, STOP_MARKER);
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'empty prompt' },
      { role: 'assistant', text: STOP_MARKER },
      { role: 'user', text: 'kept prompt' },
    ]);
    releaseSecond();
    await newer;
  });

  it('inserts the older stop marker after the newer turn has already finished', async () => {
    contextManager.clear();
    let releaseFirst;
    let markFirstStarted;
    const firstStarted = new Promise((resolve) => {
      markFirstStarted = resolve;
    });
    llama.runChat = (_model, _thinking, _message, opts) =>
      new Promise((resolve) => {
        opts.onToken('partial');
        markFirstStarted();
        releaseFirst = () => resolve('partial');
      });
    const first = engineManager.sendPrompt('first prompt');
    await firstStarted;
    engineManager.cancelGeneration();
    llama.runChat = async () => 'second reply';
    assert.equal(await engineManager.sendPrompt('second prompt'), 'second reply');
    releaseFirst();
    assert.equal(await first, withStopMarker('partial'));
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'first prompt' },
      { role: 'assistant', text: withStopMarker('partial') },
      { role: 'user', text: 'second prompt' },
      { role: 'assistant', text: 'second reply' },
    ]);
  });

  it('rejects a second prompt while the first is still preparing', async () => {
    contextManager.clear();
    llama.runChat = async () => 'ok';
    const first = engineManager.sendPrompt('one');
    const second = engineManager.sendPrompt('two');
    await assert.rejects(second, /Engine is busy/);
    assert.equal(await first, 'ok');
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'one' },
      { role: 'assistant', text: 'ok' },
    ]);
  });

  it('releases the engine when preparing the prompt throws', async () => {
    contextManager.clear();
    await assert.rejects(
      () =>
        engineManager.sendPrompt('read this', {
          files: [{ source: 'resources', relativePath: 'missing.md' }],
        }),
      /ENOENT/
    );
    assert.equal(engineManager.getStatus().phase, 'idle');
    assert.deepEqual(await engineManager.contextSnapshot(), []);
    llama.runChat = async () => 'later';
    assert.equal(await engineManager.sendPrompt('hi'), 'later');
    assert.deepEqual(await engineManager.contextSnapshot().then(contextShape), [
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: 'later' },
    ]);
  });

  it('sends a plain chat turn without allocating an image', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    state.engineBootstrapped = false;
    clearPendingSessionFilename();
    llama.runChat = async () => 'hello back';
    const denied = await handleEngineSendMessage('hello');
    assert.equal(denied.ok, false);
    assert.equal(denied.errorInfo.code, 'E_NOT_READY');

    const result = await deliverSendMessage('hello', {
      persist: {
        contextSnapshot: () => engineManager.contextSnapshot(),
        writeSessionFile: async (name, messages) => {
          await fs.writeFile(path.join(sessions, name), `${JSON.stringify(messages)}\n`);
        },
      },
    });
    assert.equal(result.response, 'hello back');
    assert.deepEqual(result.images, []);
    const names = await fs.readdir(sessions);
    assert.equal(names.some((name) => name.endsWith('.png')), false);
    assert.equal(getPendingSessionFilename(), null);
    assert.equal(state.activeSessionFilename.endsWith('.json'), true);

    state.engineBootstrapped = true;
    llama.runChat = async () => 'from ipc';
    const allowed = await handleEngineSendMessage('again');
    assert.equal(allowed.ok, true);
    assert.equal(allowed.response, 'from ipc');
    assert.deepEqual(allowed.images, []);
    state.engineBootstrapped = false;
  });

  it('does not rewrite a finished chat turn when that request is cancelled', async () => {
    contextManager.clear();
    llama.runChat = async () => 'hello';
    assert.equal(await engineManager.sendPrompt('q1', { requestId: 'req-q1', senderId: 3 }), 'hello');
    const before = await engineManager.contextSnapshot();
    assert.deepEqual(contextShape(before), [
      { role: 'user', text: 'q1' },
      { role: 'assistant', text: 'hello' },
    ]);
    let stops = 0;
    const originalStop = llama.chatStop;
    llama.chatStop = () => {
      stops += 1;
      return Promise.resolve();
    };
    engineManager.cancelGeneration();
    engineManager.cancelRequestGeneration({ requestId: 'req-q1', senderId: 3, senderFallback: true });
    llama.chatStop = originalStop;
    assert.equal(stops, 0);
    assert.equal(engineManager.getStatus().phase, 'idle');
    assert.equal(engineManager.activeRunStopSealed(), false);
    assert.deepEqual(await engineManager.contextSnapshot(), before);
  });

  it('forgets settled request ids and does not let a pruned id cancel the active run', async () => {
    contextManager.clear();
    llama.runChat = async () => 'ok';
    for (let i = 0; i < MAX_TRACKED_REQUEST_RUNS + 8; i += 1) {
      await engineManager.sendPrompt(`m${i}`, { requestId: `req-${i}`, senderId: 3 });
    }
    assert.equal(engineManager.requestRunTokenCount(), MAX_TRACKED_REQUEST_RUNS);

    let release;
    let markStarted;
    const started = new Promise((resolve) => {
      markStarted = resolve;
    });
    llama.runChat = () =>
      new Promise((resolve) => {
        markStarted();
        release = () => resolve('live');
      });
    const live = engineManager.sendPrompt('live', { requestId: 'req-live', senderId: 3 });
    await started;
    assert.ok(engineManager.requestRunTokenCount() <= MAX_TRACKED_REQUEST_RUNS);
    engineManager.cancelRequestGeneration({ requestId: 'req-0', senderId: 3, senderFallback: true });
    assert.equal(engineManager.getStatus().phase, 'generating');
    release();
    assert.equal(await live, 'live');
  });
});
