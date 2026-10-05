'use strict';

/**
 * Text-to-image request checks shared by engineManager and the native engine.
 * Generation uses the current prompt only — callers must not pass chat history.
 */

const { TEXT_TO_IMAGE_PIPELINE_TAG } = require('./resolveEngineId');

/**
 * @param {unknown} message
 * @param {Array<unknown> | null | undefined} files
 * @returns {string}
 */
function assertTextToImagePrompt(message, files) {
  const list = Array.isArray(files) ? files : [];
  if (list.length > 0) {
    throw new Error('Text-to-image models accept a text prompt only.');
  }
  const prompt = typeof message === 'string' ? message.trim() : '';
  if (!prompt) {
    throw new Error('Text-to-image requires a prompt.');
  }
  return prompt;
}

module.exports = {
  TEXT_TO_IMAGE_PIPELINE_TAG,
  assertTextToImagePrompt,
};
