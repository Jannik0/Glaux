'use strict';

/**
 * Canonical chat history for Glaux. Engines do not own long-lived conversation
 * state — engineManager reads/writes through this module, and passes message
 * snapshots into engines for inference and usage measurement.
 */

/** @type {Array<{ role: string, content: Array<{ type: string, text?: string, path?: string }> }>} */
let messages = [];

/**
 * Identity of stored message objects. Not copied onto the message, so snapshots
 * and saved sessions stay free of it. A superseded run uses the id captured
 * when its user turn was appended.
 * @type {WeakMap<object, number>}
 */
const messageIds = new WeakMap();
let nextMessageId = 1;

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
 * Copy workspace identity onto a media part when the absolute path matches.
 *
 * @param {{ type: string, path: string, source?: string, relativePath?: string }} part
 * @param {Array<{ path?: string, source?: string, relativePath?: string, kind?: string }>} attachments
 * @param {string} kind
 */
function annotateMediaPart(part, attachments, kind) {
  const match = attachments.find((item) => item && item.path === part.path && item.kind === kind);
  if (!match) {
    return part;
  }
  if (typeof match.source === 'string' && match.source) {
    part.source = match.source;
  }
  if (typeof match.relativePath === 'string' && match.relativePath) {
    part.relativePath = match.relativePath;
  }
  return part;
}

/**
 * Build a user message in the shared Glaux content-part shape.
 * `files` records workspace identity (source + relative path) so chat chips can
 * reopen the same file after the session is saved. Document records become
 * `file` parts; media records annotate the matching path part.
 *
 * @param {string} text
 * @param {{ imagePaths?: string[], audioPaths?: string[], videoPaths?: string[], files?: Array<{ path?: string, source?: string, relativePath?: string, kind?: string }> }} [media]
 */
function buildUserMessage(text, media = {}) {
  const attachments = Array.isArray(media.files) ? media.files : [];
  const content = [];
  for (const p of media.imagePaths || []) {
    if (typeof p === 'string' && p) {
      content.push(annotateMediaPart({ type: 'image', path: p }, attachments, 'image'));
    }
  }
  for (const p of media.videoPaths || []) {
    if (typeof p === 'string' && p) {
      content.push(annotateMediaPart({ type: 'video', path: p }, attachments, 'video'));
    }
  }
  for (const item of attachments) {
    if (!item || item.kind !== 'document') {
      continue;
    }
    if (typeof item.source !== 'string' || !item.source) {
      continue;
    }
    if (typeof item.relativePath !== 'string' || !item.relativePath) {
      continue;
    }
    content.push({
      type: 'file',
      source: item.source,
      relativePath: item.relativePath,
    });
  }
  if (typeof text === 'string' && text.length) {
    content.push({ type: 'text', text });
  }
  for (const p of media.audioPaths || []) {
    if (typeof p === 'string' && p) {
      content.push(annotateMediaPart({ type: 'audio', path: p }, attachments, 'audio'));
    }
  }
  if (!content.length) {
    content.push({ type: 'text', text: typeof text === 'string' ? text : '' });
  }
  return { role: 'user', content };
}

/**
 * @param {string} text
 * @param {Array<{ type: string, path?: string, relativePath?: string, source?: string }> | undefined} [imageParts]
 */
function buildAssistantMessage(text, imageParts) {
  const content = [];
  for (const part of imageParts || []) {
    if (!part || part.type !== 'image') {
      continue;
    }
    const image = { type: 'image' };
    if (typeof part.path === 'string' && part.path) {
      image.path = part.path;
    }
    if (typeof part.relativePath === 'string' && part.relativePath) {
      image.relativePath = part.relativePath;
    }
    if (typeof part.source === 'string' && part.source) {
      image.source = part.source;
    }
    if (image.path || image.relativePath) {
      content.push(image);
    }
  }
  if (typeof text === 'string' && text.length) {
    content.push({ type: 'text', text });
  }
  if (!content.length) {
    content.push({ type: 'text', text: typeof text === 'string' ? text : '' });
  }
  return { role: 'assistant', content };
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
  const stored = cloneJson(msg);
  messages.push(stored);
  const id = nextMessageId;
  nextMessageId += 1;
  messageIds.set(stored, id);
  return id;
}

/**
 * @param {string} text
 * @param {{ imagePaths?: string[], audioPaths?: string[], videoPaths?: string[], files?: Array<{ path?: string, source?: string, relativePath?: string, kind?: string }> }} [media]
 */
function appendUser(text, media) {
  return append(buildUserMessage(text, media));
}

/**
 * @param {string} text
 * @param {{ imageParts?: Array<object> }} [extras]
 */
function appendAssistant(text, extras) {
  const imageParts = extras && Array.isArray(extras.imageParts) ? extras.imageParts : undefined;
  return append(buildAssistantMessage(text, imageParts));
}

/**
 * @param {number | null | undefined} id
 * @returns {number}
 */
function messageIndex(id) {
  if (id == null) {
    return -1;
  }
  for (let i = 0; i < messages.length; i += 1) {
    if (messageIds.get(messages[i]) === id) {
      return i;
    }
  }
  return -1;
}

/**
 * Insert an assistant turn immediately after the user message identified at
 * append time. Does nothing when that message is already gone, so a newer
 * turn is never used as the anchor.
 *
 * @param {number | null | undefined} userId
 * @param {string} text
 * @param {{ imageParts?: Array<object> }} [extras]
 * @returns {number | null}
 */
function insertAssistantAfter(userId, text, extras) {
  const index = messageIndex(userId);
  if (index < 0) {
    return null;
  }
  const imageParts = extras && Array.isArray(extras.imageParts) ? extras.imageParts : undefined;
  const stored = cloneJson(buildAssistantMessage(text, imageParts));
  const id = nextMessageId;
  nextMessageId += 1;
  messageIds.set(stored, id);
  messages.splice(index + 1, 0, stored);
  return id;
}

/**
 * Remove one message by the id captured when it was stored.
 * @param {number | null | undefined} id
 * @returns {boolean}
 */
function removeMessageById(id) {
  const index = messageIndex(id);
  if (index < 0) {
    return false;
  }
  messages.splice(index, 1);
  return true;
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
  insertAssistantAfter,
  removeMessageById,
  rollbackLastUser,
  buildUserMessage,
  buildAssistantMessage,
  validateMessages,
  length,
};
