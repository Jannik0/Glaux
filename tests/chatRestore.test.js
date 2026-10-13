'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const engineManager = require('../engines/engineManager');
const contextManager = require('../engines/contextManager');
const llama = require('../engines/llamacpp/engine');
const state = require('../src/main/state');
const { handleEngineStreamStart, readChatRestoreState } = require('../src/main/domains/engineBridge');
const { clearPendingSessionFilename } = require('../src/main/domains/sessions');

function contextShape(messages) {
  return messages.map((msg) => ({
    role: msg.role,
    text: (msg.content || [])
      .filter((part) => part && part.type === 'text')
      .map((part) => part.text)
      .join('\n'),
  }));
}

function fakeSender() {
  return {
    events: [],
    id: 7,
    isDestroyed() {
      return false;
    },
    send(_channel, payload) {
      this.events.push(payload);
    },
    once() {},
    removeListener() {},
  };
}

describe('chat restore after reload', { concurrency: 1 }, () => {
  const originalRun = llama.runChat;
  const originalStarts = llama.chatGenerationStartsInThinking;
  const originalCreate = llama.chatbotCreate;
  let root;

  before(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-chat-restore-'));
    const sessions = path.join(root, 'Sessions');
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
    state.engineBootstrapped = true;
  });

  after(async () => {
    llama.runChat = originalRun;
    llama.chatGenerationStartsInThinking = originalStarts;
    llama.chatbotCreate = originalCreate;
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

  it('keeps the open turn and partial text while generation is running', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    state.activeStreamRequests.clear();
    let release;
    const started = new Promise((resolve) => {
      llama.runChat = (_model, _thinking, _message, opts) =>
        new Promise((resolveReply) => {
          opts.onToken('Hel');
          opts.onToken('lo');
          resolve();
          release = () => resolveReply('Hello');
        });
    });
    const sender = fakeSender();
    const turn = handleEngineStreamStart({ sender }, { requestId: 'restore-live', message: 'hi' });
    try {
      await started;
      const snap = await readChatRestoreState();
      assert.equal(snap.stream.requestId, 'restore-live');
      assert.equal(snap.stream.text, 'Hello');
      assert.equal(snap.stream.userPending, false);
      assert.ok(snap.stream.seq >= 2);
      assert.deepEqual(contextShape(snap.messages), [{ role: 'user', text: 'hi' }]);
      const chunk = sender.events.find((event) => event.type === 'chunk' && event.chunk === 'lo');
      assert.equal(chunk.text, 'Hello');
      assert.equal(engineManager.getStatus().phase, 'generating');
    } finally {
      release();
    }
    await turn;
    const done = await readChatRestoreState();
    assert.equal(done.stream, null);
    assert.deepEqual(contextShape(done.messages), [
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: 'Hello' },
    ]);
    assert.equal(state.activeStreamRequests.size, 0);
  });

  it('keeps the partial text when the reloading page cannot receive events', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    state.activeStreamRequests.clear();
    let release;
    const started = new Promise((resolve) => {
      llama.runChat = (_model, _thinking, _message, opts) =>
        new Promise((resolveReply) => {
          opts.onToken('Hel');
          opts.onToken('lo');
          resolve();
          release = () => resolveReply('Hello');
        });
    });
    const sender = fakeSender();
    sender.send = () => {
      throw new Error('frame disposed');
    };
    const turn = handleEngineStreamStart({ sender }, { requestId: 'restore-nav', message: 'hi' });
    try {
      await started;
      const snap = await readChatRestoreState();
      assert.equal(snap.stream.text, 'Hello');
      assert.equal(engineManager.getStatus().phase, 'generating');
      assert.equal(sender.events.length, 0);
    } finally {
      release();
    }
    await turn;
    const done = await readChatRestoreState();
    assert.equal(done.stream, null);
    assert.deepEqual(contextShape(done.messages), [
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: 'Hello' },
    ]);
  });

  it('delivers later tokens to the reloaded window, not the discarded page', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    state.activeStreamRequests.clear();
    const discarded = fakeSender();
    discarded.send = () => {
      throw new Error('frame disposed');
    };
    const reloaded = fakeSender();
    state.setMainWindow({
      isDestroyed() {
        return false;
      },
      webContents: reloaded,
    });
    let release;
    const started = new Promise((resolve) => {
      llama.runChat = (_model, _thinking, _message, opts) =>
        new Promise((resolveReply) => {
          opts.onToken('Hel');
          opts.onToken('lo');
          resolve();
          release = () => resolveReply('Hello');
        });
    });
    const turn = handleEngineStreamStart(
      { sender: discarded },
      { requestId: 'restore-retarget', message: 'hi' }
    );
    try {
      await started;
      assert.equal(discarded.events.length, 0);
      const chunk = reloaded.events.find((event) => event.type === 'chunk' && event.chunk === 'lo');
      assert.equal(chunk.text, 'Hello');
      const snap = await readChatRestoreState();
      assert.equal(snap.stream.text, 'Hello');
      release();
      await turn;
      const done = reloaded.events.find((event) => event.type === 'done');
      assert.equal(done.response, 'Hello');
      assert.equal((await readChatRestoreState()).stream, null);
    } finally {
      state.setMainWindow(null);
      if (release) {
        release();
      }
    }
  });

  it('reports the prompt before that user turn is in context', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    state.activeStreamRequests.clear();
    let releaseThinking;
    llama.chatGenerationStartsInThinking = () =>
      new Promise((resolve) => {
        releaseThinking = () => resolve(false);
      });
    llama.runChat = async () => 'ok';
    const sender = fakeSender();
    const turn = handleEngineStreamStart(
      { sender },
      { requestId: 'restore-pending', message: 'hello', enableThinking: true }
    );
    try {
      await new Promise((resolve) => setImmediate(resolve));
      const snap = await readChatRestoreState();
      assert.ok(snap.stream);
      assert.equal(snap.stream.userPending, true);
      assert.equal(snap.stream.userText, 'hello');
      assert.deepEqual(snap.messages, []);
    } finally {
      releaseThinking();
    }
    await turn;
    llama.chatGenerationStartsInThinking = originalStarts;
    const done = await readChatRestoreState();
    assert.equal(done.stream, null);
    assert.deepEqual(contextShape(done.messages), [
      { role: 'user', text: 'hello' },
      { role: 'assistant', text: 'ok' },
    ]);
  });
});
