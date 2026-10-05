'use strict';

const { describe, it, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const contextManager = require('../engines/contextManager');

describe('contextManager assistant image parts', () => {
  beforeEach(() => {
    contextManager.clear();
  });

  it('stores text-only assistant turns', () => {
    contextManager.appendAssistant('hi');
    const [msg] = contextManager.snapshot();
    assert.equal(msg.role, 'assistant');
    assert.deepEqual(msg.content, [{ type: 'text', text: 'hi' }]);
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
});
