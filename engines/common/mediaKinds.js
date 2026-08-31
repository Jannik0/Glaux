'use strict';

/**
 * Single source of truth for media/document extension classification shared by
 * engineManager.js and the engine backends. See src/main/mediaKinds.js for the
 * (slightly different) renderer/main-process extension set.
 */

const path = require('path');

const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp']);
const AUDIO_EXTS = new Set(['wav', 'mp3', 'ogg', 'flac', 'm4a']);
const VIDEO_EXTS = new Set(['mp4', 'webm', 'mov', 'avi', 'mkv', 'mts', 'm2ts', 'ts']);
const MARKDOWN_EXTS = new Set(['md', 'txt']);
const PDF_EXTS = new Set(['pdf']);

/**
 * @param {string} filePath
 * @returns {string} Lowercased extension without the leading dot (empty string if none).
 */
function extOf(filePath) {
  return path.extname(String(filePath || '')).slice(1).toLowerCase();
}

/**
 * @param {string} filePath
 * @returns {'image' | 'audio' | 'video' | null}
 */
function mediaKindFromPath(filePath) {
  const ext = extOf(filePath);
  if (IMAGE_EXTS.has(ext)) {
    return 'image';
  }
  if (AUDIO_EXTS.has(ext)) {
    return 'audio';
  }
  if (VIDEO_EXTS.has(ext)) {
    return 'video';
  }
  return null;
}

/**
 * @param {string} filePath
 * @returns {boolean}
 */
function isDocumentPath(filePath) {
  const ext = extOf(filePath);
  return MARKDOWN_EXTS.has(ext) || PDF_EXTS.has(ext);
}

module.exports = {
  IMAGE_EXTS,
  AUDIO_EXTS,
  VIDEO_EXTS,
  MARKDOWN_EXTS,
  PDF_EXTS,
  extOf,
  mediaKindFromPath,
  isDocumentPath,
};
