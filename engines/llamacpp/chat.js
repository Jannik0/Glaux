'use strict';

/**
 * Glaux <-> OpenAI chat message conversion for llama-server, including media
 * URL resolution and Jinja chat-template "thinking support" detection.
 */

const fs = require('fs');
const path = require('path');
const { stripThinkingFromMessages } = require('../common/stripThinking');

/** Absolute userData-style root passed as --media-path (Models/ and Workspaces/ parent). */
let mediaPathRoot = null;

/**
 * @param {string | null} root
 */
function setMediaPathRoot(root) {
  mediaPathRoot = root || null;
}

/**
 * @returns {string | null}
 */
function getMediaPathRoot() {
  return mediaPathRoot;
}

function pathToFileUrl(absPath) {
  const resolved = path.resolve(absPath);
  let pathname = resolved.replace(/\\/g, '/');
  if (!pathname.startsWith('/')) {
    pathname = `/${pathname}`;
  }
  // Encode spaces etc. but keep path separators.
  return encodeURI(`file://${pathname}`);
}

/**
 * llama-server only accepts file:// URLs as paths *relative* to --media-path
 * (absolute Windows paths fail fs_validate_filename because of ':').
 * @param {string} absPath
 * @returns {string} file:// relative URL or data:/raw base64 fallback for images
 */
function mediaUrlForPath(absPath, { asImage = false } = {}) {
  const resolved = path.resolve(absPath);
  if (mediaPathRoot) {
    const root = path.resolve(mediaPathRoot);
    const rel = path.relative(root, resolved);
    if (rel && !rel.startsWith('..') && !path.isAbsolute(rel)) {
      // llama-server concatenates media_path + relative path as-is (no URL decoding).
      const normalized = rel.split(path.sep).join('/');
      return `file://${normalized}`;
    }
  }
  // Fallback for files outside media-path: images can use data: URIs.
  if (asImage) {
    const buf = fs.readFileSync(resolved);
    const ext = path.extname(resolved).slice(1).toLowerCase();
    const mime =
      ext === 'png'
        ? 'image/png'
        : ext === 'gif'
          ? 'image/gif'
          : ext === 'webp'
            ? 'image/webp'
            : 'image/jpeg';
    return `data:${mime};base64,${buf.toString('base64')}`;
  }
  // Audio/video outside the root: raw base64 (llama-server accepts this when not file://).
  return fs.readFileSync(resolved).toString('base64');
}

/**
 * Detect thinking/reasoning control from a Jinja chat template (mirrors llama.cpp WebUI detector).
 * @param {string} template
 * @returns {boolean}
 */
function detectThinkingSupport(template) {
  if (!template || typeof template !== 'string') {
    return false;
  }
  const THINKING_KWARG_VARS = ['enable_thinking', 'reasoning_effort', 'thinking_budget'];
  for (const kwarg of THINKING_KWARG_VARS) {
    const regex = new RegExp(
      `(\\{\\{[^{}]*\\b${kwarg}\\b[^{}]*\\}\\}|\\{%[^{}]*\\b${kwarg}\\b[^{}]*%\\})`,
      'i'
    );
    if (regex.test(template)) {
      return true;
    }
  }
  const conditionals = [
    /\{%-?\s*if\s+\(?\s*\w*enable[\s_]+\w*(thinking|think|reasoning)/i,
    /\{%-?\s*if\s+\w*(thinking|reasoning)\s*(is not|==|!=)/i,
    /\{%-?\s*if\s+not\s+\w*enable/i,
    /\{%-?\s*if\s+ns\.enable_thinking/i,
  ];
  for (const p of conditionals) {
    if (p.test(template)) {
      return true;
    }
  }
  const tagPairs = [
    ['<think>', '</think>'],
    ['<|channel>thought', '<|channel|>'],
    ['<|think|>', '</|think|>'],
    ['<seed:think>', '</seed:think>'],
  ];
  for (const [start, end] of tagPairs) {
    if (template.includes(start) && template.includes(end)) {
      return true;
    }
  }
  return false;
}

function audioFormatFromPath(filePath) {
  const ext = path.extname(filePath).slice(1).toLowerCase();
  if (ext === 'mp3' || ext === 'wav' || ext === 'flac' || ext === 'ogg') {
    return ext;
  }
  return 'wav';
}

/**
 * @param {object} msg
 * @returns {object}
 */
function glauxMessageToOpenAI(msg) {
  const role = msg.role === 'assistant' ? 'assistant' : 'user';
  const parts = Array.isArray(msg.content) ? msg.content : [];
  const openParts = [];
  let textOnly = '';
  for (const part of parts) {
    if (!part || typeof part !== 'object') {
      continue;
    }
    if (part.type === 'text' && typeof part.text === 'string') {
      openParts.push({ type: 'text', text: part.text });
      textOnly += part.text;
    } else if (part.type === 'image' && part.path) {
      openParts.push({
        type: 'image_url',
        image_url: { url: mediaUrlForPath(part.path, { asImage: true }) },
      });
    } else if (part.type === 'audio' && part.path) {
      openParts.push({
        type: 'input_audio',
        input_audio: {
          data: mediaUrlForPath(part.path, { asImage: false }),
          format: audioFormatFromPath(part.path),
        },
      });
    } else if (part.type === 'video' && part.path) {
      openParts.push({
        type: 'input_video',
        input_video: { url: mediaUrlForPath(part.path, { asImage: false }) },
      });
    }
  }
  if (openParts.length === 0) {
    return { role, content: textOnly };
  }
  if (openParts.length === 1 && openParts[0].type === 'text') {
    return { role, content: openParts[0].text };
  }
  return { role, content: openParts };
}

/**
 * Convert Glaux CONTEXT messages to OpenAI chat messages for template/tokenize,
 * replacing media parts with short text placeholders (avoids loading files just to count).
 * @param {object} msg
 * @returns {{ role: string, content: string }}
 */
function glauxMessageToOpenAITextOnly(msg) {
  const role = msg.role === 'assistant' ? 'assistant' : 'user';
  const parts = Array.isArray(msg.content) ? msg.content : [];
  const textBits = [];
  for (const part of parts) {
    if (!part || typeof part !== 'object') {
      continue;
    }
    if (part.type === 'text' && typeof part.text === 'string') {
      textBits.push(part.text);
    } else if (part.type === 'image') {
      textBits.push('[image]');
    } else if (part.type === 'audio') {
      textBits.push('[audio]');
    } else if (part.type === 'video') {
      textBits.push('[video]');
    }
  }
  return { role, content: textBits.join('\n').trim() };
}

function buildUserContent(message, imagePaths, audioPaths, videoPaths) {
  const parts = [];
  for (const p of imagePaths || []) {
    parts.push({ type: 'image', path: p });
  }
  for (const p of videoPaths || []) {
    parts.push({ type: 'video', path: p });
  }
  if (typeof message === 'string' && message.length) {
    parts.push({ type: 'text', text: message });
  }
  for (const p of audioPaths || []) {
    parts.push({ type: 'audio', path: p });
  }
  return parts.length ? parts : [{ type: 'text', text: message || '' }];
}

/**
 * Build OpenAI messages for inference from a Glaux message list.
 * When resubmit is false, strip media from all but the last message (HF parity).
 * @param {Array<object>} sourceMessages
 * @param {boolean} resubmit
 * @returns {object[]}
 */
function buildInferenceOpenAIMessages(sourceMessages, resubmit) {
  if (!sourceMessages || !sourceMessages.length) {
    return [];
  }
  // Persisted history may include <think>…</think> for the UI; models must not see it.
  let source = stripThinkingFromMessages(sourceMessages);
  if (resubmit === false && source.length > 1) {
    source = source.map((msg, index) => {
      if (index === source.length - 1 || !Array.isArray(msg.content)) {
        return msg;
      }
      const filtered = msg.content.filter(
        (part) =>
          !part ||
          typeof part !== 'object' ||
          (part.type !== 'image' && part.type !== 'audio' && part.type !== 'video')
      );
      return filtered.length === msg.content.length ? msg : { ...msg, content: filtered };
    });
  }
  return source.map(glauxMessageToOpenAI);
}

module.exports = {
  setMediaPathRoot,
  getMediaPathRoot,
  mediaUrlForPath,
  detectThinkingSupport,
  audioFormatFromPath,
  glauxMessageToOpenAI,
  glauxMessageToOpenAITextOnly,
  buildUserContent,
  buildInferenceOpenAIMessages,
};
