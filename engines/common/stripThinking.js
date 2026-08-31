'use strict';

/**
 * Strip thinking / reasoning markup from assistant text before it is sent to a model,
 * and drop non-thinking special-token markup from Hugging Face generation output.
 * Mirrors renderer `src/renderer/shared/thinkingParse.js` and Hugging Face
 * `engines/huggingface/worker/thinking.py`. Keep these algorithms in sync.
 */

/**
 * @param {string} openChar
 * @returns {string}
 */
function tagCloseChar(openChar) {
  return openChar === '<' ? '>' : ']';
}

/**
 * `[STOP]` is a Glaux cancellation marker, not thinking markup.
 * @param {string} inner
 */
function isStopMarkerTagInner(inner) {
  return String(inner).trim().toUpperCase() === 'STOP';
}

/**
 * @param {string} text
 * @param {number} fromIndex
 * @returns {{ tagStart: number, tagEnd: number } | { incomplete: true, tagStart: number } | null}
 */
function findMarkupTagAt(text, fromIndex) {
  const angleStart = text.indexOf('<', fromIndex);
  const squareStart = text.indexOf('[', fromIndex);
  let tagStart = -1;
  if (angleStart === -1) {
    tagStart = squareStart;
  } else if (squareStart === -1) {
    tagStart = angleStart;
  } else {
    tagStart = Math.min(angleStart, squareStart);
  }
  if (tagStart === -1) {
    return null;
  }
  const closeChar = tagCloseChar(text[tagStart]);
  const tagEnd = text.indexOf(closeChar, tagStart + 1);
  if (tagEnd === -1) {
    return { incomplete: true, tagStart };
  }
  if (text[tagStart] === '[' && isStopMarkerTagInner(text.slice(tagStart + 1, tagEnd))) {
    return findMarkupTagAt(text, tagStart + 1);
  }
  return { tagStart, tagEnd };
}

/**
 * @param {string} text
 * @param {number} tagStart
 */
function isClosingMarkupTag(text, tagStart) {
  return tagStart + 1 < text.length && text[tagStart + 1] === '/';
}

/**
 * @param {string} text
 * @param {number} tagStart
 * @param {number} tagEnd
 */
function isThinkingOpenMarkerTag(text, tagStart, tagEnd) {
  if (isClosingMarkupTag(text, tagStart)) {
    return false;
  }
  return text.slice(tagStart + 1, tagEnd).toLowerCase().includes('think');
}

/**
 * @param {string} text
 * @param {number} afterIndex
 */
function hasSubstantiveNonTagContentAfter(text, afterIndex) {
  let i = afterIndex;
  while (i < text.length) {
    if (text[i] === '<' || text[i] === '[') {
      const tag = findMarkupTagAt(text, i);
      if (!tag || tag.incomplete) {
        return false;
      }
      i = tag.tagEnd + 1;
      continue;
    }
    if (!/\s/.test(text[i])) {
      return true;
    }
    i += 1;
  }
  return false;
}

/**
 * @param {string} text
 * @returns {{ thinking: string, answer: string }}
 */
function parseTagsAndAnswer(text) {
  let mode = 'answer';
  let thinking = '';
  let answer = '';
  let i = 0;
  const source = typeof text === 'string' ? text : '';
  while (i < source.length) {
    const tag = findMarkupTagAt(source, i);
    if (!tag) {
      const chunk = source.slice(i);
      if (mode === 'thinking') {
        thinking += chunk;
      } else {
        answer += chunk;
      }
      break;
    }
    if (tag.incomplete) {
      const chunk = source.slice(i);
      if (mode === 'thinking') {
        thinking += chunk;
      } else {
        answer += chunk;
      }
      break;
    }
    const { tagStart, tagEnd } = tag;
    const chunk = source.slice(i, tagStart);
    if (mode === 'thinking') {
      thinking += chunk;
    } else {
      answer += chunk;
    }
    const close = isClosingMarkupTag(source, tagStart);
    const thinkingOpen = isThinkingOpenMarkerTag(source, tagStart, tagEnd);
    i = tagEnd + 1;
    if (mode === 'thinking') {
      if (close || !thinkingOpen) {
        mode = 'answer';
      }
    } else if (!close && hasSubstantiveNonTagContentAfter(source, i)) {
      mode = 'thinking';
    }
  }
  return { thinking: thinking.trim(), answer: answer.trim() };
}

/**
 * Return answer-only text (thinking markup removed).
 * @param {string} text
 * @returns {string}
 */
function stripThinkingFromText(text) {
  if (typeof text !== 'string' || !text) {
    return typeof text === 'string' ? text : '';
  }
  if (!/[<\[]/.test(text) || !/think/i.test(text)) {
    return text;
  }
  return parseTagsAndAnswer(text).answer;
}

/**
 * Deep-clone messages with assistant text parts reduced to answer-only content.
 * User messages and non-text parts are left unchanged.
 * @param {unknown} messages
 * @returns {Array}
 */
function stripThinkingFromMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }
  return messages.map((msg) => {
    if (!msg || typeof msg !== 'object' || msg.role !== 'assistant') {
      return msg;
    }
    const content = msg.content;
    if (typeof content === 'string') {
      return { ...msg, content: stripThinkingFromText(content) };
    }
    if (!Array.isArray(content)) {
      return msg;
    }
    let changed = false;
    const nextContent = content.map((part) => {
      if (!part || typeof part !== 'object' || part.type !== 'text') {
        return part;
      }
      if (typeof part.text !== 'string') {
        return part;
      }
      const answer = stripThinkingFromText(part.text);
      if (answer === part.text) {
        return part;
      }
      changed = true;
      return { ...part, text: answer };
    });
    return changed ? { ...msg, content: nextContent } : msg;
  });
}

/**
 * Drop chat-template / EOS markup that is not a thinking delimiter, keeping
 * tags the renderer uses to split thoughts from the answer.
 *
 * Mirrors `parseTagsAndAnswer`: any opening tag followed by non-tag content
 * starts thinking (the tag need not contain "think"); a close tag or a
 * non-thinking-open tag ends it. Tags that never participate in that split
 * (trailing `<|im_end|>`, stray `</s>`, …) are removed. `[STOP]` is left as-is.
 *
 * @param {string} text
 * @param {{ holdUnresolved?: boolean, prefix?: string }} [options] When
 *   `holdUnresolved` is true, omit a trailing incomplete tag or an opening
 *   answer-mode tag with no following content (streaming: wait for more bytes
 *   before deciding). `prefix` is an opening thinking tag the chat template
 *   already emitted (skipped by the streamer); it is prepended so a later
 *   closer is kept for reload.
 * @returns {string}
 */
function stripNonThinkingMarkup(text, options = {}) {
  const holdUnresolved = options.holdUnresolved === true;
  const prefix = typeof options.prefix === 'string' ? options.prefix : '';
  const source = prefix + (typeof text === 'string' ? text : '');
  if (!source) {
    return source;
  }
  if (!/[<\[]/.test(source)) {
    return source;
  }

  let mode = 'answer';
  let out = '';
  let i = 0;
  while (i < source.length) {
    const tag = findMarkupTagAt(source, i);
    if (!tag) {
      out += source.slice(i);
      break;
    }
    if (tag.incomplete) {
      out += source.slice(i, tag.tagStart);
      if (!holdUnresolved) {
        out += source.slice(tag.tagStart);
      }
      break;
    }
    const { tagStart, tagEnd } = tag;
    out += source.slice(i, tagStart);
    const close = isClosingMarkupTag(source, tagStart);
    const thinkingOpen = isThinkingOpenMarkerTag(source, tagStart, tagEnd);
    const tagText = source.slice(tagStart, tagEnd + 1);
    i = tagEnd + 1;
    if (mode === 'thinking') {
      out += tagText;
      if (close || !thinkingOpen) {
        mode = 'answer';
      }
      continue;
    }
    if (close) {
      continue;
    }
    if (hasSubstantiveNonTagContentAfter(source, i)) {
      out += tagText;
      mode = 'thinking';
      continue;
    }
    if (holdUnresolved) {
      break;
    }
  }
  return out;
}

/**
 * If generation stopped inside a thinking block, close the last open think tag
 * so a trailing marker like `[STOP]` is parsed as answer text, not thoughts.
 * @param {string} text
 * @returns {string}
 */
function closeUnterminatedThinking(text) {
  const source = typeof text === 'string' ? text : '';
  let mode = 'answer';
  let lastOpenChar = '<';
  let i = 0;
  while (i < source.length) {
    const tag = findMarkupTagAt(source, i);
    if (!tag) {
      break;
    }
    if (tag.incomplete) {
      break;
    }
    const { tagStart, tagEnd } = tag;
    const close = isClosingMarkupTag(source, tagStart);
    const thinkingOpen = isThinkingOpenMarkerTag(source, tagStart, tagEnd);
    i = tagEnd + 1;
    if (mode === 'thinking') {
      if (close || !thinkingOpen) {
        mode = 'answer';
      }
    } else if (!close && thinkingOpen && hasSubstantiveNonTagContentAfter(source, i)) {
      lastOpenChar = source[tagStart];
      mode = 'thinking';
    }
  }
  if (mode !== 'thinking') {
    return source;
  }
  const closeTag = lastOpenChar === '[' ? '[/think]' : '</think>';
  return `${source.replace(/\s+$/, '')}${closeTag}`;
}

module.exports = {
  parseTagsAndAnswer,
  stripThinkingFromText,
  stripThinkingFromMessages,
  stripNonThinkingMarkup,
  closeUnterminatedThinking,
};
