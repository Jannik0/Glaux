'use strict';

const { closeUnterminatedThinking } = require('./stripThinking');

/** Locale-independent marker appended when the user stops generation. */
const STOP_MARKER = '[STOP]';

/**
 * Close any unterminated thinking block, then append `[STOP]` after the
 * generated text so the partial reply can be kept in chat context.
 * @param {string} text
 * @returns {string}
 */
function withStopMarker(text) {
  const closed = closeUnterminatedThinking(typeof text === 'string' ? text : '');
  const body = closed.replace(/\s+$/, '');
  if (!body) {
    return STOP_MARKER;
  }
  if (body.endsWith(STOP_MARKER)) {
    return body;
  }
  return `${body}\n\n${STOP_MARKER}`;
}

module.exports = {
  STOP_MARKER,
  withStopMarker,
};
