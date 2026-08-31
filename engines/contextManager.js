'use strict';

/**
 * Canonical chat history for Glaux. Engines do not own long-lived conversation
 * state — engineManager reads/writes through this module, and passes message
 * snapshots into engines for inference and usage measurement.
 */

/** @type {Array<{ role: string, content: Array<{ type: string, text?: string, path?: string }> }>} */
let messages = [];

/**
 * @param {unknown} value
 * @returns {Array}
 */
function validateMessages(value) {
  if (!Array.isArray(value)) {
    throw new Error('Context messages must be a list.');
  }
  for (let i = 0; i < value.length; i += 1) {
    const msg = value[i];
    if (!msg || typeof msg !== 'object') {
      throw new Error(`Context message at index ${i} must be an object.`);
    }
    const role = msg.role;
    if (role !== 'user' && role !== 'assistant') {
      throw new Error(`Context message at index ${i} has an invalid role.`);
    }
  }
  return value;
}

/**
 * Deep-clone via JSON to drop non-serializable values (matches prior engine snapshots).
 * @param {unknown} value
 */
function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

/**
 * @returns {Array}
 */
function snapshot() {
  return cloneJson(messages);
}

/**
 * @param {unknown} next
 */
function replace(next) {
  const validated = validateMessages(next == null ? [] : next);
  messages = cloneJson(validated);
}

function clear() {
  messages = [];
}

/**
 * Build a user message in the shared Glaux content-part shape.
 *
 * @param {string} text
 * @param {{ imagePaths?: string[], audioPaths?: string[], videoPaths?: string[] }} [media]
 */
function buildUserMessage(text, media = {}) {
  const content = [];
  for (const p of media.imagePaths || []) {
    if (typeof p === 'string' && p) {
      content.push({ type: 'image', path: p });
    }
  }
  for (const p of media.videoPaths || []) {
    if (typeof p === 'string' && p) {
      content.push({ type: 'video', path: p });
    }
  }
  if (typeof text === 'string' && text.length) {
    content.push({ type: 'text', text });
  }
  for (const p of media.audioPaths || []) {
    if (typeof p === 'string' && p) {
      content.push({ type: 'audio', path: p });
    }
  }
  if (!content.length) {
    content.push({ type: 'text', text: typeof text === 'string' ? text : '' });
  }
  return { role: 'user', content };
}

/**
 * @param {string} text
 */
function buildAssistantMessage(text) {
  return {
    role: 'assistant',
    content: [{ type: 'text', text: typeof text === 'string' ? text : '' }],
  };
}

/**
 * @param {{ role: string, content: unknown }} msg
 */
function append(msg) {
  if (!msg || typeof msg !== 'object') {
    throw new Error('Context message must be an object.');
  }
  if (msg.role !== 'user' && msg.role !== 'assistant') {
    throw new Error('Context message has an invalid role.');
  }
  messages.push(cloneJson(msg));
}

/**
 * @param {string} text
 * @param {{ imagePaths?: string[], audioPaths?: string[], videoPaths?: string[] }} [media]
 */
function appendUser(text, media) {
  append(buildUserMessage(text, media));
}

/**
 * @param {string} text
 */
function appendAssistant(text) {
  append(buildAssistantMessage(text));
}

/**
 * Remove the last message when it is a user turn (failed generation).
 * Stopped generations keep the user turn and append a partial assistant reply.
 * @returns {boolean}
 */
function rollbackLastUser() {
  if (!messages.length) {
    return false;
  }
  const last = messages[messages.length - 1];
  if (!last || last.role !== 'user') {
    return false;
  }
  messages.pop();
  return true;
}

function length() {
  return messages.length;
}

module.exports = {
  snapshot,
  replace,
  clear,
  append,
  appendUser,
  appendAssistant,
  rollbackLastUser,
  buildUserMessage,
  buildAssistantMessage,
  validateMessages,
  length,
};
