'use strict';

/**
 * Diffusion request checks shared by engineManager and the native engines.
 * Generation uses the current prompt only — callers must not pass chat history.
 * Image-to-image accepts the same prompt, plus an optional image.
 */

const { mediaKindFromPath } = require('./mediaKinds');
const { IMAGE_TO_IMAGE_PIPELINE_TAG, TEXT_TO_IMAGE_PIPELINE_TAG } = require('./resolveEngineId');

/**
 * @param {unknown} message
 * @param {string} label
 * @returns {string}
 */
function requirePrompt(message, label) {
  const prompt = typeof message === 'string' ? message.trim() : '';
  if (!prompt) {
    throw new Error(`${label} requires a prompt.`);
  }
  return prompt;
}

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
  return requirePrompt(message, 'Text-to-image');
}

/**
 * A text prompt, and at most one image. Audio, video, and documents are rejected.
 * @param {unknown} message
 * @param {Array<{ relativePath?: string }> | null | undefined} files
 * @returns {{ prompt: string, imageFile: { relativePath?: string } | null }}
 */
function assertImageToImagePrompt(message, files) {
  const list = Array.isArray(files) ? files : [];
  /** @type {Array<{ relativePath?: string }>} */
  const images = [];
  for (const file of list) {
    const rel = file && typeof file.relativePath === 'string' ? file.relativePath : '';
    if (mediaKindFromPath(rel) !== 'image') {
      throw new Error('Image-to-image models accept a text prompt and an optional image.');
    }
    images.push(file);
  }
  if (images.length > 1) {
    throw new Error('Image-to-image models accept at most one image.');
  }
  return { prompt: requirePrompt(message, 'Image-to-image'), imageFile: images[0] || null };
}

module.exports = {
  TEXT_TO_IMAGE_PIPELINE_TAG,
  IMAGE_TO_IMAGE_PIPELINE_TAG,
  assertTextToImagePrompt,
  assertImageToImagePrompt,
};
