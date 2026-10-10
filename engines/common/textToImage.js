'use strict';

/**
 * Diffusion request checks shared by engineManager and the native engines.
 * Generation uses the current prompt only — callers must not pass chat history.
 * Image-to-image, and GGUF text-to-image, accept the same prompt plus an optional image.
 * Safetensors text-to-image stays prompt-only.
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
 * @param {string} label
 * @returns {{ prompt: string, imageFile: { relativePath?: string } | null }}
 */
function assertOptionalImagePrompt(message, files, label) {
  const list = Array.isArray(files) ? files : [];
  /** @type {Array<{ relativePath?: string }>} */
  const images = [];
  for (const file of list) {
    const rel = file && typeof file.relativePath === 'string' ? file.relativePath : '';
    if (mediaKindFromPath(rel) !== 'image') {
      throw new Error(`${label} models accept a text prompt and an optional image.`);
    }
    images.push(file);
  }
  if (images.length > 1) {
    throw new Error(`${label} models accept at most one image.`);
  }
  return { prompt: requirePrompt(message, label), imageFile: images[0] || null };
}

/**
 * @param {unknown} message
 * @param {Array<{ relativePath?: string }> | null | undefined} files
 * @returns {{ prompt: string, imageFile: { relativePath?: string } | null }}
 */
function assertImageToImagePrompt(message, files) {
  return assertOptionalImagePrompt(message, files, 'Image-to-image');
}

/**
 * GGUF text-to-image. The image is the stable-diffusion.cpp init image.
 * @param {unknown} message
 * @param {Array<{ relativePath?: string }> | null | undefined} files
 * @returns {{ prompt: string, imageFile: { relativePath?: string } | null }}
 */
function assertGgufTextToImagePrompt(message, files) {
  return assertOptionalImagePrompt(message, files, 'Text-to-image');
}

module.exports = {
  TEXT_TO_IMAGE_PIPELINE_TAG,
  IMAGE_TO_IMAGE_PIPELINE_TAG,
  assertTextToImagePrompt,
  assertImageToImagePrompt,
  assertGgufTextToImagePrompt,
};
