'use strict';

const { describe, it, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const state = require('../src/main/state');
const {
  allocateSessionImagePath,
  deleteSessionSidecarImages,
} = require('../src/main/sessionImages');
const {
  clearPendingSessionFilename,
  deleteSessionMessage,
  getPendingSessionFilename,
  loadSession,
  persistActiveSession,
  reserveActiveSessionFilename,
  startNewSession,
  trashNamedSession,
  trashSessionFile,
} = require('../src/main/domains/sessions');
const { activateWorkspace } = require('../src/main/domains/workspaces');

describe('session delete and trash order', () => {
  let root;
  let sessions;
  let outputs;

  after(async () => {
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    if (root) {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  async function layout() {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-session-order-'));
    sessions = path.join(root, 'Sessions');
    outputs = path.join(root, 'Outputs');
    await fs.mkdir(sessions);
    await fs.mkdir(outputs);
  }

  function imageMessage() {
    return {
      role: 'assistant',
      content: [
        {
          type: 'image',
          source: 'sessions',
          relativePath: 'chat-1.png',
          path: path.join(sessions, 'chat-1.png'),
        },
        {
          type: 'image',
          source: 'outputs',
          relativePath: 'keep.png',
          path: path.join(outputs, 'keep.png'),
        },
      ],
    };
  }

  it('leaves sidecars in place when contextReplace or the session write fails', async () => {
    await layout();
    const sidecar = path.join(sessions, 'chat-1.png');
    const kept = path.join(outputs, 'keep.png');
    await fs.writeFile(sidecar, 'png');
    await fs.writeFile(kept, 'keep');
    const msg = imageMessage();
    const calls = [];
    await assert.rejects(
      () =>
        deleteSessionMessage(0, {
          contextSnapshot: async () => [msg],
          contextReplace: async () => {
            calls.push('replace');
            throw new Error('replace failed');
          },
          writeSessionFile: async () => {
            calls.push('write');
          },
          getActiveSessionFilename: () => 'chat.json',
          sessionsRoot: sessions,
          deleteSessionSidecarImages: async (messages, dir) => {
            calls.push('sidecars');
            return deleteSessionSidecarImages(messages, dir);
          },
        }),
      /replace failed/
    );
    assert.deepEqual(calls, ['replace']);
    assert.equal(await fs.readFile(sidecar, 'utf8'), 'png');

    calls.length = 0;
    await assert.rejects(
      () =>
        deleteSessionMessage(0, {
          contextSnapshot: async () => [msg],
          contextReplace: async () => {
            calls.push('replace');
          },
          writeSessionFile: async () => {
            calls.push('write');
            throw new Error('disk full');
          },
          getActiveSessionFilename: () => 'chat.json',
          sessionsRoot: sessions,
          deleteSessionSidecarImages: async (messages, dir) => {
            calls.push('sidecars');
            return deleteSessionSidecarImages(messages, dir);
          },
        }),
      /disk full/
    );
    assert.deepEqual(calls, ['replace', 'write']);
    assert.equal(await fs.readFile(sidecar, 'utf8'), 'png');
    assert.equal(await fs.readFile(kept, 'utf8'), 'keep');
  });

  it('deletes sidecars only after replace and write, and never touches Outputs', async () => {
    await layout();
    const sidecar = path.join(sessions, 'chat-1.png');
    const kept = path.join(outputs, 'keep.png');
    await fs.writeFile(sidecar, 'png');
    await fs.writeFile(kept, 'keep');
    const calls = [];
    const warnings = [];
    await deleteSessionMessage(0, {
      contextSnapshot: async () => [imageMessage()],
      contextReplace: async () => {
        calls.push('replace');
      },
      writeSessionFile: async () => {
        calls.push('write');
      },
      getActiveSessionFilename: () => 'chat.json',
      sessionsRoot: sessions,
      warn: (message, err) => warnings.push([message, err]),
      deleteSessionSidecarImages: async (messages, dir) => {
        calls.push('sidecars');
        return deleteSessionSidecarImages(messages, dir);
      },
    });
    assert.deepEqual(calls, ['replace', 'write', 'sidecars']);
    await assert.rejects(fs.stat(sidecar));
    assert.equal(await fs.readFile(kept, 'utf8'), 'keep');

    await fs.writeFile(sidecar, 'png');
    warnings.length = 0;
    const messages = await deleteSessionMessage(0, {
      contextSnapshot: async () => [imageMessage()],
      contextReplace: async () => {},
      writeSessionFile: async () => {},
      getActiveSessionFilename: () => 'chat.json',
      sessionsRoot: sessions,
      warn: (message, err) => warnings.push(String(message)),
      deleteSessionSidecarImages: async () => {
        throw new Error('sidecar busy');
      },
    });
    assert.equal(messages.length, 0);
    assert.equal(warnings.length, 1);
    assert.equal(await fs.readFile(sidecar, 'utf8'), 'png');
  });

  it('trashes the JSON before sidecars and keeps them when the move fails', async () => {
    await layout();
    const jsonPath = path.join(sessions, 'chat.json');
    const sidecar = path.join(sessions, 'chat-1.png');
    const kept = path.join(outputs, 'keep.png');
    await fs.writeFile(jsonPath, '[]');
    await fs.writeFile(sidecar, 'png');
    await fs.writeFile(kept, 'keep');
    const calls = [];
    await assert.rejects(
      () =>
        trashSessionFile('chat.json', {
          resolveSessionsPath: () => jsonPath,
          readSessionFile: async () => [imageMessage()],
          moveEntryToRecycleBin: async () => {
            calls.push('move');
            throw new Error('recycle failed');
          },
          sessionsRoot: sessions,
          deleteSessionSidecarImages: async (messages, dir) => {
            calls.push('sidecars');
            return deleteSessionSidecarImages(messages, dir);
          },
        }),
      /recycle failed/
    );
    assert.deepEqual(calls, ['move']);
    assert.equal(await fs.readFile(jsonPath, 'utf8'), '[]');
    assert.equal(await fs.readFile(sidecar, 'utf8'), 'png');

    const trash = path.join(root, 'trashed.json');
    const warnings = [];
    await trashSessionFile('chat.json', {
      resolveSessionsPath: () => jsonPath,
      readSessionFile: async () => [imageMessage()],
      moveEntryToRecycleBin: async (abs) => {
        calls.push('move');
        await fs.rename(abs, trash);
      },
      sessionsRoot: sessions,
      warn: (message) => warnings.push(String(message)),
      deleteSessionSidecarImages: async () => {
        throw new Error('sidecar busy');
      },
    });
    assert.equal(await fs.readFile(trash, 'utf8'), '[]');
    assert.equal(await fs.readFile(sidecar, 'utf8'), 'png');
    assert.equal(await fs.readFile(kept, 'utf8'), 'keep');
    assert.equal(warnings.length, 1);

    await fs.writeFile(jsonPath, '[]');
    calls.length = 0;
    await trashSessionFile('chat.json', {
      resolveSessionsPath: () => jsonPath,
      readSessionFile: async () => [imageMessage()],
      moveEntryToRecycleBin: async (abs) => {
        calls.push('move');
        await fs.rm(abs);
      },
      sessionsRoot: sessions,
      deleteSessionSidecarImages: async (messages, dir) => {
        calls.push('sidecars');
        return deleteSessionSidecarImages(messages, dir);
      },
    });
    assert.deepEqual(calls, ['move', 'sidecars']);
    await assert.rejects(fs.stat(jsonPath));
    await assert.rejects(fs.stat(sidecar));
    assert.equal(await fs.readFile(kept, 'utf8'), 'keep');
  });
});

describe('pending session filename', () => {
  after(() => {
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
  });

  it('does not publish the name until persist succeeds', async () => {
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-pending-session-'));
    try {
      const sessions = path.join(root, 'Sessions');
      await fs.mkdir(sessions);
      const filename = reserveActiveSessionFilename();
      assert.equal(state.activeSessionFilename, null);
      assert.match(filename, /\.json$/);
      const allocated = await allocateSessionImagePath(sessions, filename);
      assert.equal(allocated.relativePath, `${filename.replace(/\.json$/i, '')}-1.png`);
      await fs.writeFile(allocated.absolutePath, 'png');

      await assert.rejects(
        () =>
          persistActiveSession({
            contextSnapshot: async () => [],
            writeSessionFile: async () => {
              throw new Error('disk');
            },
          }),
        /disk/
      );
      assert.equal(state.activeSessionFilename, null);
      assert.equal(getPendingSessionFilename(), filename);

      const next = reserveActiveSessionFilename();
      assert.equal(state.activeSessionFilename, null);
      let written = null;
      const committed = await persistActiveSession({
        contextSnapshot: async () => [{ role: 'user', content: 'hi' }],
        writeSessionFile: async (name, messages) => {
          written = { name, messages };
        },
      });
      assert.equal(committed, next);
      assert.equal(state.activeSessionFilename, next);
      assert.equal(written.name, next);
      assert.equal(reserveActiveSessionFilename(), next);
      assert.equal(getPendingSessionFilename(), null);
    } finally {
      state.activeSessionFilename = null;
      clearPendingSessionFilename();
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it('clears the pending name on new, load, trash of the active session, and workspace switch', async () => {
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    const workspaceBefore = state.activeWorkspaceName;
    try {
      reserveActiveSessionFilename();
      assert.ok(getPendingSessionFilename());
      await startNewSession({ contextClear: async () => {} });
      assert.equal(getPendingSessionFilename(), null);
      assert.equal(state.activeSessionFilename, null);

      reserveActiveSessionFilename();
      const loaded = await loadSession('saved.json', {
        readSessionFile: async () => [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
        contextReplace: async () => {},
      });
      assert.equal(loaded.sessionName, 'saved.json');
      assert.equal(getPendingSessionFilename(), null);
      assert.equal(state.activeSessionFilename, 'saved.json');

      state.activeSessionFilename = null;
      const pending = reserveActiveSessionFilename();
      state.activeSessionFilename = pending;
      const trashed = await trashNamedSession(pending, {
        trashSessionFile: async () => {},
        contextClear: async () => {},
      });
      assert.equal(trashed.wasActive, true);
      assert.equal(state.activeSessionFilename, null);
      assert.equal(getPendingSessionFilename(), null);

      reserveActiveSessionFilename();
      const activated = await activateWorkspace('Other', {
        setActiveWorkspacePaths: (name) => name,
        ensureWorkspaceDirectories: async () => {},
        persistActiveWorkspaceName: async () => {},
        reconfigureEnginePathsIfReady: async () => {},
        closeWorkspaceDependentWindows: () => {},
      });
      assert.equal(activated, 'Other');
      assert.equal(getPendingSessionFilename(), null);
      assert.equal(state.activeSessionFilename, null);
    } finally {
      state.activeSessionFilename = null;
      state.activeWorkspaceName = workspaceBefore;
      clearPendingSessionFilename();
    }
  });
});
