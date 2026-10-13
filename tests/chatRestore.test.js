'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const engineManager = require('../engines/engineManager');
const contextManager = require('../engines/contextManager');
const llama = require('../engines/llamacpp/engine');
const stableDiffusion = require('../engines/stablediffusioncpp/engine');
const { STOP_MARKER } = require('../engines/common/stopMarker');
const state = require('../src/main/state');
const { initAppPaths, setActiveWorkspacePaths, getSessionsRoot } = require('../src/main/paths');
const {
  handleEngineStreamStart,
  readChatRestoreState,
  cancelStreamRequest,
} = require('../src/main/domains/engineBridge');
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
  const originalSdRun = stableDiffusion.runChat;
  let root;
  let cache;

  before(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-chat-restore-'));
    initAppPaths({ userDataRoot: root });
    await fs.mkdir(path.join(root, 'Workspaces', 'Ws', 'Sessions'), { recursive: true });
    setActiveWorkspacePaths('Ws');
    const sessions = getSessionsRoot();
    const resources = path.join(path.dirname(sessions), 'Resources');
    const outputs = path.join(path.dirname(sessions), 'Outputs');
    await fs.mkdir(resources, { recursive: true });
    await fs.mkdir(outputs, { recursive: true });
    cache = path.join(root, 'Models');
    const model = path.join(cache, 'org', 'chat');
    await fs.mkdir(model, { recursive: true });
    await fs.writeFile(path.join(model, 'model.gguf'), 'gguf');
    await fs.writeFile(path.join(model, 'README.md'), '---\npipeline_tag: text-generation\n---\n');
    llama.chatbotCreate = async () => {};
    await engineManager.initialize({
      modelId: 'org/chat',
      modelsCacheDir: cache,
      sessionsRoot: sessions,
      resourcesRoot: resources,
      outputsRoot: outputs,
    });
    state.engineBootstrapped = true;
  });

  after(async () => {
    llama.runChat = originalRun;
    llama.chatGenerationStartsInThinking = originalStarts;
    llama.chatbotCreate = originalCreate;
    stableDiffusion.runChat = originalSdRun;
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
    const errors = [];
    const originalError = console.error;
    console.error = (...args) => {
      errors.push(args.map((part) => (part instanceof Error ? part.message : String(part))).join(' '));
    };
    try {
      await turn;
    } finally {
      console.error = originalError;
    }
    llama.chatGenerationStartsInThinking = originalStarts;
    const done = await readChatRestoreState();
    assert.equal(done.stream, null);
    assert.deepEqual(contextShape(done.messages), [
      { role: 'user', text: 'hello' },
      { role: 'assistant', text: 'ok' },
    ]);
    assert.equal(
      errors.some((line) => line.includes('No active workspace')),
      false
    );
    assert.ok(state.activeSessionFilename);
    const saved = JSON.parse(
      await fs.readFile(path.join(getSessionsRoot(), state.activeSessionFilename), 'utf8')
    );
    assert.deepEqual(contextShape(saved), contextShape(done.messages));
  });

  it('keeps a settling lock after stop until the sealed turn is saved', async () => {
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
    const turn = handleEngineStreamStart({ sender }, { requestId: 'restore-stop', message: 'hi' });
    let duringSave = null;
    const originalWrite = fs.writeFile;
    fs.writeFile = async function writeDuringSave(...args) {
      const target = String(args[0] || '');
      if (target.endsWith('.json') && !duringSave) {
        duringSave = await readChatRestoreState();
      }
      return originalWrite.apply(this, args);
    };
    try {
      await started;
      cancelStreamRequest('restore-stop', sender);
      const snap = await readChatRestoreState();
      assert.equal(snap.stream.requestId, 'restore-stop');
      assert.equal(snap.stream.settling, true);
      assert.equal(snap.stream.userPending, false);
      assert.equal(snap.stream.userText, '');
      const assistants = snap.messages.filter((msg) => msg.role === 'assistant');
      assert.equal(assistants.length, 1);
      assert.ok(contextShape(assistants)[0].text.includes(STOP_MARKER));
      release();
      await turn;
    } finally {
      fs.writeFile = originalWrite;
      if (release) {
        release();
      }
    }
    const done = await readChatRestoreState();
    assert.equal(done.stream, null);
    assert.ok(duringSave);
    assert.equal(duringSave.stream.settling, true);
    assert.equal(duringSave.stream.userPending, false);
    assert.equal(duringSave.messages.filter((msg) => msg.role === 'assistant').length, 1);
    const saved = JSON.parse(
      await fs.readFile(path.join(getSessionsRoot(), state.activeSessionFilename), 'utf8')
    );
    assert.match(contextShape(saved).map((msg) => msg.text).join('\n'), /\[STOP\]/);
    assert.equal(state.activeStreamRequests.size, 0);
  });

  it('loads the session file into context when the engine context is empty', async () => {
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    state.activeStreamRequests.clear();
    const stored = [
      { role: 'user', content: [{ type: 'text', text: 'saved question' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'saved answer' }] },
    ];
    const filename = 'saved-chat.json';
    await fs.writeFile(
      path.join(getSessionsRoot(), filename),
      `${JSON.stringify(stored, null, 2)}\n`
    );
    state.activeSessionFilename = filename;
    const snap = await readChatRestoreState();
    assert.equal(snap.stream, null);
    assert.deepEqual(contextShape(snap.messages), [
      { role: 'user', text: 'saved question' },
      { role: 'assistant', text: 'saved answer' },
    ]);
    assert.deepEqual(contextShape(await engineManager.contextSnapshot()), contextShape(snap.messages));
  });

  it('restores an in-progress image run and delivers the image on done', async () => {
    const sdRoot = path.join(cache, 'org', 'sd');
    await fs.mkdir(sdRoot, { recursive: true });
    await fs.writeFile(path.join(sdRoot, 'model.gguf'), 'gguf');
    await fs.writeFile(path.join(sdRoot, 'README.md'), '---\npipeline_tag: text-to-image\n---\n');
    process.env.GLAUX_SD_CLI = process.execPath;
    contextManager.clear();
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    state.activeStreamRequests.clear();
    let release;
    const started = new Promise((resolve) => {
      stableDiffusion.runChat = (_model, _thinking, _prompt, opts) =>
        new Promise((resolveReply) => {
          resolve();
          release = async () => {
            await fs.writeFile(opts.outputPath, 'png');
            resolveReply({ text: 'fox', imagePaths: [opts.outputPath] });
          };
        });
    });
    try {
      await engineManager.reinitialize({ modelId: 'org/sd' });
      const sender = fakeSender();
      const turn = handleEngineStreamStart(
        { sender },
        { requestId: 'restore-image', message: 'a red fox' }
      );
      await started;
      const snap = await readChatRestoreState();
      assert.equal(snap.stream.requestId, 'restore-image');
      assert.equal(snap.stream.settling, false);
      assert.equal(snap.stream.userPending, false);
      assert.deepEqual(contextShape(snap.messages), [{ role: 'user', text: 'a red fox' }]);
      await release();
      await turn;
      const doneEvent = sender.events.find((event) => event.type === 'done');
      assert.ok(doneEvent);
      assert.equal(doneEvent.images.length, 1);
      assert.equal(doneEvent.images[0].source, 'sessions');
      const after = await readChatRestoreState();
      assert.equal(after.stream, null);
      const image = after.messages
        .flatMap((msg) => (Array.isArray(msg.content) ? msg.content : []))
        .find((part) => part.type === 'image');
      assert.equal(image.relativePath, doneEvent.images[0].relativePath);
      const saved = JSON.parse(
        await fs.readFile(path.join(getSessionsRoot(), state.activeSessionFilename), 'utf8')
      );
      const savedImage = saved
        .flatMap((msg) => (Array.isArray(msg.content) ? msg.content : []))
        .find((part) => part.type === 'image');
      assert.equal(savedImage.relativePath, image.relativePath);
    } finally {
      stableDiffusion.runChat = originalSdRun;
      delete process.env.GLAUX_SD_CLI;
      contextManager.clear();
      state.activeSessionFilename = null;
      clearPendingSessionFilename();
      state.activeStreamRequests.clear();
      await engineManager.reinitialize({ modelId: 'org/chat' });
    }
  });
});
