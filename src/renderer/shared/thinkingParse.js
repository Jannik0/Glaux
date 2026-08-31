// Markdown rendering + "<think>"-tag stream parsing shared by the chat panel.
// Exposed as window.Glaux.ThinkingParse.
(function () {
  window.Glaux = window.Glaux || {};

  /**
   * @param {HTMLElement} container
   * @param {string} text
   * @param {{ streaming?: boolean }} [options]
   */
  function renderFormattedMessage(container, text, options = {}) {
    const streaming = options.streaming === true;
    container.textContent = '';
    if (!text) {
      return;
    }
    if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
      container.textContent = text;
      return;
    }
    try {
      const html = marked.parse(text, {
        async: false,
        gfm: true,
        breaks: true,
        silent: streaming,
      });
      container.innerHTML = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    } catch (err) {
      if (!streaming) {
        console.error(err);
      }
      container.textContent = text;
    }
  }

  /**
   * @param {string} text
   * @param {number} fromIndex
   * @returns {{ tagStart: number, tagEnd: number } | { incomplete: true, tagStart: number } | null}
   */
  function findTagAt(text, fromIndex) {
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
    const closeChar = text[tagStart] === '<' ? '>' : ']';
    const tagEnd = text.indexOf(closeChar, tagStart + 1);
    if (tagEnd === -1) {
      return { incomplete: true, tagStart };
    }
    if (
      text[tagStart] === '[' &&
      text.slice(tagStart + 1, tagEnd).trim().toUpperCase() === 'STOP'
    ) {
      return findTagAt(text, tagStart + 1);
    }
    return { tagStart, tagEnd };
  }

  function isClosingTag(text, tagStart) {
    return tagStart + 1 < text.length && text[tagStart + 1] === '/';
  }

  /** Opening thinking marker duplicated in the stream (Qwen puts the real open tag in the prompt). */
  function isThinkingOpenMarkerTag(text, tagStart, tagEnd) {
    if (isClosingTag(text, tagStart)) {
      return false;
    }
    const inner = text.slice(tagStart + 1, tagEnd).toLowerCase();
    return inner.includes('think');
  }

  /** True when `afterIndex` is followed by a non-whitespace character outside of markup tags. */
  function hasSubstantiveNonTagContentAfter(text, afterIndex) {
    let i = afterIndex;
    while (i < text.length) {
      if (text[i] === '<' || text[i] === '[') {
        const tag = findTagAt(text, i);
        if (!tag || 'incomplete' in tag) {
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
   * Split assistant text: segments between consecutive effective markup tags go to thoughts.
   * A tag is effective only when followed by non-tag content (not another tag or EOF).
   * Supports angle (`<...>`) and square (`[...]`) bracket styles.
   * @param {string} text
   * @returns {{ thinking: string, answer: string }}
   */
  function parseTagsAndAnswer(text) {
    let mode = 'answer';
    let thinking = '';
    let answer = '';
    let i = 0;
    while (i < text.length) {
      const tag = findTagAt(text, i);
      if (!tag) {
        const chunk = text.slice(i);
        if (mode === 'thinking') {
          thinking += chunk;
        } else {
          answer += chunk;
        }
        break;
      }
      if ('incomplete' in tag) {
        const chunk = text.slice(i);
        if (mode === 'thinking') {
          thinking += chunk;
        } else {
          answer += chunk;
        }
        break;
      }
      const { tagStart, tagEnd } = tag;
      const chunk = text.slice(i, tagStart);
      if (mode === 'thinking') {
        thinking += chunk;
      } else {
        answer += chunk;
      }
      const close = isClosingTag(text, tagStart);
      const thinkingOpen = isThinkingOpenMarkerTag(text, tagStart, tagEnd);
      i = tagEnd + 1;
      if (mode === 'thinking') {
        if (close || !thinkingOpen) {
          mode = 'answer';
        }
      } else if (close) {
        // Stray close in answer mode — strip only.
      } else if (hasSubstantiveNonTagContentAfter(text, i)) {
        mode = 'thinking';
      }
    }
    return { thinking: thinking.trim(), answer: answer.trim() };
  }

  function retainedPartialTagLength(text) {
    let keep = 0;
    for (const openChar of ['<', '[']) {
      const closeChar = openChar === '<' ? '>' : ']';
      const tagStart = text.lastIndexOf(openChar);
      if (tagStart === -1) {
        continue;
      }
      if (text.indexOf(closeChar, tagStart) === -1) {
        keep = Math.max(keep, text.length - tagStart);
      }
    }
    return keep;
  }

  /**
   * Stream parser: any markup tag toggles answer/thoughts; text between two tags is thoughts.
   * @param {HTMLElement} thoughtsContentEl
   * @param {HTMLElement} answerEl
   * @param {HTMLElement} detailsEl
   * @param {{ startInThinking?: boolean }} [options]
   */
  function createTagStreamParser(thoughtsContentEl, answerEl, detailsEl, options = {}) {
    let mode = options.startInThinking ? 'thinking' : 'answer';
    let buffer = '';
    let thinkingPlain = '';
    let answerPlain = '';
    /** Effective tag seen with no following bytes yet; apply toggle when non-tag arrives. */
    let pendingTagToggle = false;

    if (options.startInThinking) {
      detailsEl.style.display = '';
    }

    function applyTagToggle() {
      mode = mode === 'answer' ? 'thinking' : 'answer';
    }

    function settlePendingTagToggle() {
      if (!pendingTagToggle) {
        return;
      }
      pendingTagToggle = false;
      applyTagToggle();
    }

    let rafThoughts = null;
    let rafAnswer = null;

    function cancelScheduledRenders() {
      if (rafThoughts != null) {
        cancelAnimationFrame(rafThoughts);
        rafThoughts = null;
      }
      if (rafAnswer != null) {
        cancelAnimationFrame(rafAnswer);
        rafAnswer = null;
      }
    }

    function scheduleThoughtsRender() {
      if (rafThoughts != null) {
        return;
      }
      rafThoughts = requestAnimationFrame(() => {
        rafThoughts = null;
        renderFormattedMessage(thoughtsContentEl, thinkingPlain, { streaming: true });
      });
    }

    function scheduleAnswerRender() {
      if (rafAnswer != null) {
        return;
      }
      rafAnswer = requestAnimationFrame(() => {
        rafAnswer = null;
        renderFormattedMessage(answerEl, answerPlain, { streaming: true });
      });
    }

    function appendToMode(text) {
      if (!text) {
        return;
      }
      if (mode === 'thinking') {
        thinkingPlain += text;
        detailsEl.style.display = '';
        scheduleThoughtsRender();
      } else {
        answerPlain += text;
        scheduleAnswerRender();
      }
    }

    function processBufferedText() {
      while (buffer.length > 0) {
        const tag = findTagAt(buffer, 0);
        if (!tag) {
          const keep = retainedPartialTagLength(buffer);
          if (buffer.length <= keep) {
            return;
          }
          if (pendingTagToggle) {
            if (!hasSubstantiveNonTagContentAfter(buffer, 0)) {
              return;
            }
            settlePendingTagToggle();
          }
          appendToMode(buffer.slice(0, buffer.length - keep));
          buffer = buffer.slice(buffer.length - keep);
          return;
        }
        if ('incomplete' in tag) {
          if (pendingTagToggle && tag.tagStart === 0) {
            return;
          }
          if (pendingTagToggle && hasSubstantiveNonTagContentAfter(buffer, 0)) {
            settlePendingTagToggle();
          }
          appendToMode(buffer.slice(0, tag.tagStart));
          buffer = buffer.slice(tag.tagStart);
          return;
        }
        const { tagStart, tagEnd } = tag;
        const close = isClosingTag(buffer, tagStart);
        const thinkingOpen = isThinkingOpenMarkerTag(buffer, tagStart, tagEnd);
        if (tagStart > 0 && pendingTagToggle) {
          if (hasSubstantiveNonTagContentAfter(buffer, 0)) {
            settlePendingTagToggle();
          } else {
            return;
          }
        }
        appendToMode(buffer.slice(0, tagStart));
        buffer = buffer.slice(tagEnd + 1);
        if (mode === 'thinking') {
          pendingTagToggle = false;
          if (close || !thinkingOpen) {
            mode = 'answer';
          }
          continue;
        }
        if (close) {
          continue;
        }
        if (hasSubstantiveNonTagContentAfter(buffer, 0)) {
          if (pendingTagToggle) {
            settlePendingTagToggle();
          }
          applyTagToggle();
        } else if (buffer.length === 0) {
          pendingTagToggle = true;
          return;
        }
      }
    }

    function processChunk(chunk) {
      buffer += chunk;
      processBufferedText();
    }

    processChunk.flush = function () {
      cancelScheduledRenders();
      if (pendingTagToggle) {
        if (mode === 'thinking' || hasSubstantiveNonTagContentAfter(buffer, 0)) {
          settlePendingTagToggle();
        } else {
          pendingTagToggle = false;
          buffer = '';
        }
      }
      if (buffer) {
        appendToMode(buffer);
        buffer = '';
      }
      if (!thinkingPlain.trim()) {
        detailsEl.style.display = 'none';
      }
      renderFormattedMessage(thoughtsContentEl, thinkingPlain, { streaming: true });
      renderFormattedMessage(answerEl, answerPlain, { streaming: true });
      return {
        state: mode === 'thinking' ? 'thinking' : 'answering',
        thinking: thinkingPlain,
        answer: answerPlain,
      };
    };

    return processChunk;
  }

  window.Glaux.ThinkingParse = {
    renderFormattedMessage,
    findTagAt,
    isClosingTag,
    isThinkingOpenMarkerTag,
    hasSubstantiveNonTagContentAfter,
    parseTagsAndAnswer,
    retainedPartialTagLength,
    createTagStreamParser,
  };
})();
