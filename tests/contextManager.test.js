'use strict';

const { describe, it, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const contextManager = require('../engines/contextManager');

describe('contextManager message parts', () => {
  beforeEach(() => {
    contextManager.clear();
  });

  it('stores text-only assistant turns', () => {
    contextManager.appendAssistant('hi');
    const [msg] = contextManager.snapshot();
    assert.equal(msg.role, 'assistant');
    assert.deepEqual(msg.content, [{ type: 'text', text: 'hi' }]);
  });

  it('stores workspace identity on user attachments', () => {
    contextManager.appendUser('see attached', {
      imagePaths: ['/resources/photos/cat.png'],
      audioPaths: ['/outputs/voice.wav'],
      files: [
        {
          path: '/resources/photos/cat.png',
          source: 'resources',
          relativePath: 'photos/cat.png',
          kind: 'image',
        },
        {
          path: '/outputs/voice.wav',
          source: 'outputs',
          relativePath: 'voice.wav',
          kind: 'audio',
        },
        {
          source: 'outputs',
          relativePath: 'notes/brief.pdf',
          kind: 'document',
        },
      ],
    });
    const [msg] = contextManager.snapshot();
    assert.deepEqual(msg.content, [
      {
        type: 'image',
        path: '/resources/photos/cat.png',
        source: 'resources',
        relativePath: 'photos/cat.png',
      },
      {
        type: 'file',
        source: 'outputs',
        relativePath: 'notes/brief.pdf',
      },
      { type: 'text', text: 'see attached' },
      {
        type: 'audio',
        path: '/outputs/voice.wav',
        source: 'outputs',
        relativePath: 'voice.wav',
      },
    ]);
  });

  it('stores an image part without an empty text part', () => {
    contextManager.appendAssistant('', {
      imageParts: [
        {
          type: 'image',
          path: '/sessions/chat-1.png',
          relativePath: 'chat-1.png',
          source: 'sessions',
        },
      ],
    });
    const [msg] = contextManager.snapshot();
    assert.deepEqual(msg.content, [
      {
        type: 'image',
        path: '/sessions/chat-1.png',
        relativePath: 'chat-1.png',
        source: 'sessions',
      },
    ]);
  });

  it('inserts an assistant turn after one user message and can remove that user', () => {
    const first = contextManager.appendUser('one');
    const second = contextManager.appendUser('two');
    const inserted = contextManager.insertAssistantAfter(first, '[STOP]');
    assert.equal(typeof inserted, 'number');
    assert.equal(contextManager.insertAssistantAfter(999999, 'nope'), null);
    const snap = contextManager.snapshot();
    assert.deepEqual(
      snap.map((msg) => ({ role: msg.role, text: msg.content[0].text })),
      [
        { role: 'user', text: 'one' },
        { role: 'assistant', text: '[STOP]' },
        { role: 'user', text: 'two' },
      ]
    );
    assert.equal(JSON.stringify(snap).includes('messageId'), false);
    assert.equal(contextManager.removeMessageById(second), true);
    assert.equal(contextManager.removeMessageById(second), false);
    assert.deepEqual(
      contextManager.snapshot().map((msg) => msg.content[0].text),
      ['one', '[STOP]']
    );
  });
});
