'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const {
  allocateSessionImagePath,
  copyAssistantImagesToOutputs,
  deleteSessionSidecarImages,
} = require('../src/main/sessionImages');

describe('session image sidecars', () => {
  let root;
  let sessions;
  let outputs;

  before(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'glaux-sessions-'));
    sessions = path.join(root, 'Sessions');
    outputs = path.join(root, 'Outputs');
    await fs.mkdir(sessions);
    await fs.mkdir(outputs);
  });

  after(async () => {
    await fs.rm(root, { recursive: true, force: true });
  });

  it('allocates the next free sibling png', async () => {
    await fs.writeFile(path.join(sessions, 'chat-1.png'), 'one');
    const allocated = await allocateSessionImagePath(sessions, 'chat.json');
    assert.equal(allocated.relativePath, 'chat-2.png');
    assert.equal(allocated.source, 'sessions');
    assert.equal(allocated.absolutePath, path.join(sessions, 'chat-2.png'));
  });

  it('copies assistant images into Outputs and leaves the sidecar', async () => {
    const sidecar = path.join(sessions, 'chat-1.png');
    const msg = {
      role: 'assistant',
      content: [
        {
          type: 'image',
          path: sidecar,
          relativePath: 'chat-1.png',
          source: 'sessions',
        },
      ],
    };
    const first = await copyAssistantImagesToOutputs(msg, { sessionsRoot: sessions, outputsRoot: outputs });
    assert.deepEqual(first, ['chat-1.png']);
    assert.equal(await fs.readFile(path.join(outputs, 'chat-1.png'), 'utf8'), 'one');
    assert.equal(await fs.readFile(sidecar, 'utf8'), 'one');

    const second = await copyAssistantImagesToOutputs(msg, { sessionsRoot: sessions, outputsRoot: outputs });
    assert.deepEqual(second, ['chat-1-2.png']);
    assert.equal(await fs.readFile(path.join(outputs, 'chat-1-2.png'), 'utf8'), 'one');
  });

  it('deletes session sidecars and keeps Outputs copies', async () => {
    const outside = path.join(outputs, 'keep.png');
    await fs.writeFile(outside, 'keep');
    const messages = [
      {
        role: 'assistant',
        content: [
          { type: 'image', path: path.join(sessions, 'chat-1.png'), relativePath: 'chat-1.png', source: 'sessions' },
          { type: 'image', path: outside, relativePath: 'keep.png', source: 'outputs' },
          { type: 'image', relativePath: '../Outputs/keep.png', source: 'sessions' },
        ],
      },
      {
        role: 'user',
        content: [{ type: 'image', path: path.join(sessions, 'chat-1.png'), relativePath: 'chat-1.png', source: 'sessions' }],
      },
    ];
    await deleteSessionSidecarImages(messages, sessions);
    await assert.rejects(fs.stat(path.join(sessions, 'chat-1.png')));
    assert.equal(await fs.readFile(outside, 'utf8'), 'keep');
    assert.equal(await fs.readFile(path.join(outputs, 'chat-1-2.png'), 'utf8'), 'one');
  });

  it('rejects sidecar names that are not a single non-json file in Sessions', async () => {
    const outside = path.join(root, 'outside.png');
    await fs.writeFile(outside, 'out');
    await fs.writeFile(path.join(sessions, 'other.json'), '{"no":true}');
    await fs.mkdir(path.join(sessions, 'nested'), { recursive: true });
    await fs.writeFile(path.join(sessions, 'nested', 'a.png'), 'nest');
    await fs.mkdir(path.join(sessions, 'dir.png'));
    const messages = [
      {
        role: 'assistant',
        content: [
          { type: 'image', source: 'sessions', relativePath: 'other.json' },
          { type: 'image', source: 'sessions', relativePath: '../x' },
          { type: 'image', source: 'sessions', relativePath: 'nested/a.png' },
          { type: 'image', source: 'sessions', path: outside },
          { type: 'image', source: 'sessions', relativePath: outside },
          { type: 'image', source: 'sessions', relativePath: 'dir.png' },
          { type: 'image', source: 'sessions', relativePath: 'missing.png' },
          { type: 'image', source: 'resources', relativePath: 'chat-1.png' },
        ],
      },
    ];
    await deleteSessionSidecarImages(messages, sessions);
    assert.equal(await fs.readFile(path.join(sessions, 'other.json'), 'utf8'), '{"no":true}');
    assert.equal(await fs.readFile(path.join(sessions, 'nested', 'a.png'), 'utf8'), 'nest');
    assert.equal((await fs.stat(path.join(sessions, 'dir.png'))).isDirectory(), true);
    assert.equal(await fs.readFile(outside, 'utf8'), 'out');
    await assert.rejects(
      () =>
        copyAssistantImagesToOutputs(messages[0], { sessionsRoot: sessions, outputsRoot: outputs }),
      /missing/
    );
  });

  it('still resolves a sidecar whose stem is not the current session name', async () => {
    const renamed = await allocateSessionImagePath(sessions, 'renamed.json');
    assert.equal(renamed.relativePath, 'renamed-1.png');
    await fs.writeFile(path.join(sessions, 'original-1.png'), 'kept-name');
    const msg = {
      role: 'assistant',
      content: [{ type: 'image', source: 'sessions', relativePath: 'original-1.png' }],
    };
    const copied = await copyAssistantImagesToOutputs(msg, {
      sessionsRoot: sessions,
      outputsRoot: outputs,
    });
    assert.equal(copied[0], 'original-1.png');
    await deleteSessionSidecarImages([msg], sessions);
    await assert.rejects(fs.stat(path.join(sessions, 'original-1.png')));
    assert.equal(await fs.readFile(path.join(outputs, 'original-1.png'), 'utf8'), 'kept-name');
  });

  it('rejects Windows JSON aliases and a png symlink to a session json', async () => {
    const names = ['other.json.', 'x.json::$DATA', 'X.JSON', 'shot.png '];
    for (const name of names) {
      await fs.writeFile(path.join(sessions, name), 'keep');
    }
    const jsonTarget = path.join(sessions, 'notes.json');
    await fs.writeFile(jsonTarget, '{"n":1}');
    const link = path.join(sessions, 'a.png');
    let linked = true;
    try {
      await fs.symlink(jsonTarget, link);
    } catch (err) {
      if (err && (err.code === 'EPERM' || err.code === 'ENOTSUP')) {
        linked = false;
      } else {
        throw err;
      }
    }
    const content = names.map((name) => ({ type: 'image', source: 'sessions', relativePath: name }));
    if (linked) {
      content.push({ type: 'image', source: 'sessions', relativePath: 'a.png' });
    }
    await deleteSessionSidecarImages([{ role: 'assistant', content }], sessions);
    for (const name of names) {
      assert.equal(await fs.readFile(path.join(sessions, name), 'utf8'), 'keep');
    }
    assert.equal(await fs.readFile(jsonTarget, 'utf8'), '{"n":1}');
    if (linked) {
      assert.equal((await fs.lstat(link)).isSymbolicLink(), true);
    }
  });

  it('does not follow a symlink inside Sessions that points outside', async () => {
    const outside = path.join(root, 'secret.png');
    await fs.writeFile(outside, 'secret');
    const link = path.join(sessions, 'link.png');
    try {
      await fs.symlink(outside, link);
    } catch (err) {
      if (err && (err.code === 'EPERM' || err.code === 'ENOTSUP')) {
        return;
      }
      throw err;
    }
    await deleteSessionSidecarImages(
      [{ role: 'assistant', content: [{ type: 'image', source: 'sessions', relativePath: 'link.png' }] }],
      sessions
    );
    assert.equal(await fs.readFile(outside, 'utf8'), 'secret');
    const stat = await fs.lstat(link);
    assert.equal(stat.isSymbolicLink(), true);
  });

  it('does not copy user attachments', async () => {
    const names = await copyAssistantImagesToOutputs(
      { role: 'user', content: [{ type: 'image', path: path.join(outputs, 'keep.png'), source: 'outputs', relativePath: 'keep.png' }] },
      { sessionsRoot: sessions, outputsRoot: outputs }
    );
    assert.deepEqual(names, []);
  });
});
