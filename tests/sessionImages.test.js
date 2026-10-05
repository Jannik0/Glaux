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

  it('does not copy user attachments', async () => {
    const names = await copyAssistantImagesToOutputs(
      { role: 'user', content: [{ type: 'image', path: path.join(outputs, 'keep.png'), source: 'outputs', relativePath: 'keep.png' }] },
      { sessionsRoot: sessions, outputsRoot: outputs }
    );
    assert.deepEqual(names, []);
  });
});
