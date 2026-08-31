'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { STOP_MARKER, withStopMarker } = require('../engines/common/stopMarker');
const {
  closeUnterminatedThinking,
  parseTagsAndAnswer,
} = require('../engines/common/stripThinking');

describe('withStopMarker', () => {
  it('appends [STOP] after partial answer text', () => {
    assert.equal(withStopMarker('Hello'), `Hello\n\n${STOP_MARKER}`);
  });

  it('returns [STOP] when nothing was generated', () => {
    assert.equal(withStopMarker(''), STOP_MARKER);
    assert.equal(withStopMarker('   '), STOP_MARKER);
  });

  it('does not duplicate an existing [STOP] marker', () => {
    assert.equal(withStopMarker(`Hello\n\n${STOP_MARKER}`), `Hello\n\n${STOP_MARKER}`);
  });

  it('closes unterminated thinking so [STOP] stays in the answer', () => {
    const result = withStopMarker('<think>reason so far');
    assert.match(result, /<\/think>\s*\n\n\[STOP\]$/);
    assert.ok(result.startsWith('<think>reason so far'));
    const { thinking, answer } = parseTagsAndAnswer(result);
    assert.equal(thinking, 'reason so far');
    assert.equal(answer, STOP_MARKER);
  });
});

describe('closeUnterminatedThinking', () => {
  it('leaves completed thinking blocks unchanged', () => {
    const text = '<think>reason</think>\nHello';
    assert.equal(closeUnterminatedThinking(text), text);
  });
});
