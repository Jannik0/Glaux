'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const {
  parseTagsAndAnswer,
  stripThinkingFromText,
  stripThinkingFromMessages,
  stripNonThinkingMarkup,
} = require('../engines/common/stripThinking');

describe('stripThinking', () => {
  it('parses angle-bracket think tags', () => {
    const { thinking, answer } = parseTagsAndAnswer(
      '<think>reason</think>\nHello'
    );
    assert.equal(thinking, 'reason');
    assert.equal(answer, 'Hello');
  });

  it('parses square-bracket think tags', () => {
    const { thinking, answer } = parseTagsAndAnswer(
      '[think]reason[/think]Final'
    );
    assert.equal(thinking, 'reason');
    assert.equal(answer, 'Final');
  });

  it('returns original text when no thinking markup', () => {
    assert.equal(stripThinkingFromText('plain answer'), 'plain answer');
    assert.equal(stripThinkingFromText(''), '');
  });

  it('keeps a trailing [STOP] marker in the answer', () => {
    const { thinking, answer } = parseTagsAndAnswer('Hello\n\n[STOP]');
    assert.equal(thinking, '');
    assert.equal(answer, 'Hello\n\n[STOP]');
    assert.equal(
      stripThinkingFromText('<think>reason</think>\nHello\n\n[STOP]'),
      'Hello\n\n[STOP]',
    );
  });

  it('strips thinking from assistant message text parts', () => {
    const messages = [
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
      {
        role: 'assistant',
        content: [{ type: 'text', text: '<think>x</think>y' }],
      },
    ];
    const next = stripThinkingFromMessages(messages);
    assert.equal(next[0].content[0].text, 'hi');
    assert.equal(next[1].content[0].text, 'y');
  });
});

describe('stripNonThinkingMarkup', () => {
  it('strips trailing chat-template / EOS tokens from an answer', () => {
    assert.equal(stripNonThinkingMarkup('Hello world<|im_end|>'), 'Hello world');
    assert.equal(stripNonThinkingMarkup('Hello world</s>'), 'Hello world');
    assert.equal(stripNonThinkingMarkup('Hello world<|eot_id|>'), 'Hello world');
    assert.equal(stripNonThinkingMarkup('Hello world<|endoftext|>'), 'Hello world');
  });

  it('keeps think tags so the renderer can split thoughts from the answer', () => {
    assert.equal(
      stripNonThinkingMarkup('<think>reason</think>\nHello<|im_end|>'),
      '<think>reason</think>\nHello'
    );
    assert.equal(
      stripNonThinkingMarkup('[think]reason[/think]Final'),
      '[think]reason[/think]Final'
    );
  });

  it('keeps thinking delimiters that do not contain the word think', () => {
    assert.equal(
      stripNonThinkingMarkup('<|channel>thought here<|channel|>Hello<|im_end|>'),
      '<|channel>thought here<|channel|>Hello'
    );
  });

  it('keeps a trailing [STOP] marker', () => {
    assert.equal(
      stripNonThinkingMarkup('Hello\n\n[STOP]'),
      'Hello\n\n[STOP]'
    );
    assert.equal(
      stripNonThinkingMarkup('<think>reason</think>\nHello\n\n[STOP]'),
      '<think>reason</think>\nHello\n\n[STOP]'
    );
  });

  it('leaves plain text unchanged', () => {
    assert.equal(stripNonThinkingMarkup('plain answer'), 'plain answer');
    assert.equal(stripNonThinkingMarkup(''), '');
  });

  it('holds unresolved trailing tags while streaming', () => {
    assert.equal(
      stripNonThinkingMarkup('Hello<|im_end|>', { holdUnresolved: true }),
      'Hello'
    );
    assert.equal(
      stripNonThinkingMarkup('<think>', { holdUnresolved: true }),
      ''
    );
    assert.equal(
      stripNonThinkingMarkup('<think>reason', { holdUnresolved: true }),
      '<think>reason'
    );
  });

  it('is idempotent after a full clean', () => {
    const cleaned = stripNonThinkingMarkup(
      '<think>reason</think>\nHello<|im_end|>'
    );
    assert.equal(stripNonThinkingMarkup(cleaned), cleaned);
  });

  it('restores a template-provided opening think tag so reload can split', () => {
    const cleaned = stripNonThinkingMarkup('reason</think>\nHello<|im_end|>', {
      prefix: '<think>',
    });
    assert.equal(cleaned, '<think>reason</think>\nHello');
    const { thinking, answer } = parseTagsAndAnswer(cleaned);
    assert.equal(thinking, 'reason');
    assert.equal(answer, 'Hello');
  });

  it('restores a template-provided opener that does not contain think', () => {
    const cleaned = stripNonThinkingMarkup(
      'thought here<|channel|>Hello<|im_end|>',
      { prefix: '<|channel>' }
    );
    assert.equal(cleaned, '<|channel>thought here<|channel|>Hello');
    const { thinking, answer } = parseTagsAndAnswer(cleaned);
    assert.equal(thinking, 'thought here');
    assert.equal(answer, 'Hello');
  });

  it('drops a template prefix when generation is empty', () => {
    assert.equal(stripNonThinkingMarkup('', { prefix: '<think>' }), '');
  });

  it('streams cleaned text without emitting trailing specials until flush drops them', () => {
    function streamClean(chunks, options = {}) {
      const prefix = typeof options.prefix === 'string' ? options.prefix : '';
      let text = prefix;
      let emitted = 0;
      let out = '';
      for (const chunk of chunks) {
        if (!chunk) continue;
        text += chunk;
        const cleaned = stripNonThinkingMarkup(text, { holdUnresolved: true });
        out += cleaned.slice(emitted);
        emitted = cleaned.length;
      }
      const cleaned = stripNonThinkingMarkup(text);
      out += cleaned.slice(emitted);
      return out;
    }

    assert.equal(
      streamClean(['Hel', 'lo world', '<|im_end|>']),
      'Hello world'
    );
    assert.equal(
      streamClean(['<think>', 'reason', '</think>', '\nHi', '<|im_end|>']),
      '<think>reason</think>\nHi'
    );
    assert.equal(
      streamClean(['reason', '</think>', '\nHi', '<|im_end|>'], { prefix: '<think>' }),
      '<think>reason</think>\nHi'
    );
  });
});
