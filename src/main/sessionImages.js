'use strict';

/**
 * Generated text-to-image files live next to the session JSON
 * (`<sessions>/<stem>-N.png`). A session rename does not rename those files;
 * the image part keeps the original single-segment relative path.
 * Export copies them into Outputs. Deleting a session removes those sidecars
 * and leaves Outputs copies alone.
 */

const fs = require('fs/promises');
const path = require('path');
const { isSubPath } = require('./pathSandbox');

const IMAGE_PART_TYPES = new Set(['image']);

/**
 * @param {string} root
 * @param {string} candidate
 * @returns {boolean}
 */
function isFileInside(root, candidate) {
  if (!root || !candidate) {
    return false;
  }
  const parent = path.resolve(root);
  const file = path.resolve(candidate);
  if (file === parent) {
    return false;
  }
  return isSubPath(parent, file);
}

/**
 * @param {unknown} part
 * @returns {boolean}
 */
function isImagePart(part) {
  return Boolean(part && typeof part === 'object' && IMAGE_PART_TYPES.has(part.type));
}

/**
 * @param {unknown} msg
 * @returns {Array<object>}
 */
function imagePartsOf(msg) {
  if (!msg || typeof msg !== 'object' || !Array.isArray(msg.content)) {
    return [];
  }
  return msg.content.filter(isImagePart);
}

/**
 * A session sidecar name is one path segment. It is not tied to the current
 * session filename: renaming the JSON leaves `<stem>-N.png` in place.
 * @param {unknown} relativePath
 * @returns {boolean}
 */
function isSessionSidecarName(relativePath) {
  if (typeof relativePath !== 'string' || !relativePath) {
    return false;
  }
  if (relativePath === '.' || relativePath === '..') {
    return false;
  }
  if (path.isAbsolute(relativePath)) {
    return false;
  }
  if (relativePath.includes('/') || relativePath.includes('\\') || relativePath.includes('\0')) {
    return false;
  }
  return path.extname(relativePath).toLowerCase() !== '.json';
}

/**
 * Resolve a non-session image part from its relative path. An absolute
 * `part.path` stored in the session JSON is not trusted.
 *
 * @param {object} part
 * @param {{ sessionsRoot?: string, resourcesRoot?: string, outputsRoot?: string }} roots
 * @returns {string | null}
 */
function resolveImagePartPath(part, roots) {
  if (!isImagePart(part) || part.source === 'sessions') {
    return null;
  }
  const resourcesRoot = roots && roots.resourcesRoot;
  const outputsRoot = roots && roots.outputsRoot;
  const relativePath = typeof part.relativePath === 'string' ? part.relativePath : '';
  const source = part.source;

  if (relativePath && source === 'resources' && resourcesRoot) {
    const abs = path.resolve(resourcesRoot, relativePath);
    return isFileInside(resourcesRoot, abs) ? abs : null;
  }
  if (relativePath && source === 'outputs' && outputsRoot) {
    const abs = path.resolve(outputsRoot, relativePath);
    return isFileInside(outputsRoot, abs) ? abs : null;
  }
  return null;
}

/**
 * Session sidecar after symlink resolution. The real file must be a regular
 * file directly inside the real Sessions directory.
 *
 * @param {object} part
 * @param {string} sessionsRoot
 * @returns {Promise<string | null>}
 */
async function resolveSessionSidecarFile(part, sessionsRoot) {
  if (!isImagePart(part) || part.source !== 'sessions' || !sessionsRoot) {
    return null;
  }
  if (!isSessionSidecarName(part.relativePath)) {
    return null;
  }
  const candidate = path.resolve(sessionsRoot, part.relativePath);
  if (!isFileInside(sessionsRoot, candidate)) {
    return null;
  }
  let realRoot;
  let realFile;
  try {
    realRoot = await fs.realpath(sessionsRoot);
    realFile = await fs.realpath(candidate);
  } catch {
    return null;
  }
  if (path.dirname(realFile) !== realRoot) {
    return null;
  }
  try {
    const stats = await fs.stat(realFile);
    if (!stats.isFile()) {
      return null;
    }
  } catch {
    return null;
  }
  return realFile;
}

/**
 * @param {string} sessionsRoot
 * @param {string} sessionFilename
 * @returns {Promise<{ absolutePath: string, relativePath: string, source: 'sessions' }>}
 */
async function allocateSessionImagePath(sessionsRoot, sessionFilename) {
  if (!sessionsRoot) {
    throw new Error('Sessions directory is not configured.');
  }
  const base = String(sessionFilename || '').replace(/\.json$/i, '');
  if (!base || base.includes('/') || base.includes('\\')) {
    throw new Error('Session name is required to store a generated image.');
  }
  await fs.mkdir(sessionsRoot, { recursive: true });
  for (let n = 1; n < 100000; n += 1) {
    const relativePath = `${base}-${n}.png`;
    const absolutePath = path.join(sessionsRoot, relativePath);
    try {
      await fs.access(absolutePath);
    } catch {
      return { absolutePath, relativePath, source: 'sessions' };
    }
  }
  throw new Error('Could not allocate a session image path.');
}

/**
 * Image files referenced by messages that live directly inside the sessions directory.
 *
 * @param {Array<object>} messages
 * @param {string} sessionsRoot
 * @returns {Promise<string[]>}
 */
async function sessionSidecarPaths(messages, sessionsRoot) {
  if (!sessionsRoot) {
    return [];
  }
  const found = [];
  const seen = new Set();
  for (const msg of messages || []) {
    for (const part of imagePartsOf(msg)) {
      const abs = await resolveSessionSidecarFile(part, sessionsRoot);
      if (!abs || seen.has(abs)) {
        continue;
      }
      seen.add(abs);
      found.push(abs);
    }
  }
  return found;
}

/**
 * @param {Array<object>} messages
 * @param {string} sessionsRoot
 * @param {{ warn?: (err: Error, filePath: string) => void }} [opts]
 * @returns {Promise<string[]>}
 */
async function deleteSessionSidecarImages(messages, sessionsRoot, opts = {}) {
  const warn =
    opts && typeof opts.warn === 'function'
      ? opts.warn
      : (err, filePath) => {
          console.warn(`Failed to delete session sidecar image ${filePath}:`, err);
        };
  const paths = await sessionSidecarPaths(messages, sessionsRoot);
  const removed = [];
  for (const filePath of paths) {
    try {
      await fs.rm(filePath, { force: true });
      removed.push(filePath);
    } catch (err) {
      warn(err, filePath);
    }
  }
  return removed;
}

/**
 * @param {string} outputsRoot
 * @param {string} preferredName
 * @returns {Promise<string>}
 */
async function uniqueOutputFileName(outputsRoot, preferredName) {
  const base = path.basename(String(preferredName || 'image.png'));
  const ext = path.extname(base) || '.png';
  const stem = path.basename(base, path.extname(base)) || 'image';
  let name = `${stem}${ext}`;
  let n = 2;
  while (true) {
    try {
      await fs.access(path.join(outputsRoot, name));
    } catch {
      return name;
    }
    name = `${stem}-${n}${ext}`;
    n += 1;
    if (n > 10000) {
      throw new Error('Could not copy image into Outputs.');
    }
  }
}

/**
 * @param {object} part
 * @param {{ sessionsRoot?: string, resourcesRoot?: string, outputsRoot?: string }} roots
 * @returns {Promise<string | null>}
 */
async function resolveExportImagePath(part, roots) {
  if (part && part.source === 'sessions') {
    return resolveSessionSidecarFile(part, roots && roots.sessionsRoot);
  }
  return resolveImagePartPath(part, roots);
}

/**
 * Copy assistant image parts into the Outputs library. Session sidecars stay
 * in place. Files that already live in Outputs are left as they are.
 *
 * @param {object} msg
 * @param {{ sessionsRoot?: string, resourcesRoot?: string, outputsRoot: string }} roots
 * @returns {Promise<string[]>}
 */
async function copyAssistantImagesToOutputs(msg, roots) {
  if (!msg || msg.role !== 'assistant') {
    return [];
  }
  const outputsRoot = roots && roots.outputsRoot;
  if (!outputsRoot) {
    throw new Error('Outputs directory is not configured.');
  }
  await fs.mkdir(outputsRoot, { recursive: true });
  const copied = [];
  for (const part of imagePartsOf(msg)) {
    const src = await resolveExportImagePath(part, roots);
    if (!src) {
      throw new Error('Generated image path is missing.');
    }
    try {
      const stats = await fs.stat(src);
      if (!stats.isFile()) {
        throw new Error('Generated image file is missing.');
      }
    } catch (err) {
      if (err && err.code === 'ENOENT') {
        throw new Error('Generated image file is missing.');
      }
      throw err;
    }
    if (isFileInside(outputsRoot, src)) {
      copied.push(path.relative(outputsRoot, src).split(path.sep).join('/'));
      continue;
    }
    const preferred =
      (typeof part.relativePath === 'string' && part.relativePath) || path.basename(src);
    const name = await uniqueOutputFileName(outputsRoot, preferred);
    await fs.copyFile(src, path.join(outputsRoot, name));
    copied.push(name);
  }
  return copied;
}

module.exports = {
  allocateSessionImagePath,
  copyAssistantImagesToOutputs,
  deleteSessionSidecarImages,
  imagePartsOf,
  isFileInside,
  isSessionSidecarName,
  resolveImagePartPath,
  resolveSessionSidecarFile,
  sessionSidecarPaths,
  uniqueOutputFileName,
};
