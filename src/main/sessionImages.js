'use strict';

/**
 * Generated text-to-image files live next to the session JSON
 * (`<sessions>/<session>-N.png`). Export copies them into Outputs.
 * Deleting a session removes those sidecars and leaves Outputs copies alone.
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
 * Resolve an image part to an absolute path. Session sidecars use
 * `source: 'sessions'` plus `relativePath`. A stored absolute `path` is used
 * when it still points at a file.
 *
 * @param {object} part
 * @param {{ sessionsRoot?: string, resourcesRoot?: string, outputsRoot?: string }} roots
 * @returns {string | null}
 */
function resolveImagePartPath(part, roots) {
  if (!isImagePart(part)) {
    return null;
  }
  const sessionsRoot = roots && roots.sessionsRoot;
  const resourcesRoot = roots && roots.resourcesRoot;
  const outputsRoot = roots && roots.outputsRoot;
  const relativePath = typeof part.relativePath === 'string' ? part.relativePath : '';
  const source = part.source;

  if (relativePath && source === 'sessions' && sessionsRoot) {
    const abs = path.resolve(sessionsRoot, relativePath);
    return isFileInside(sessionsRoot, abs) ? abs : null;
  }
  if (relativePath && source === 'resources' && resourcesRoot) {
    const abs = path.resolve(resourcesRoot, relativePath);
    return isFileInside(resourcesRoot, abs) ? abs : null;
  }
  if (relativePath && source === 'outputs' && outputsRoot) {
    const abs = path.resolve(outputsRoot, relativePath);
    return isFileInside(outputsRoot, abs) ? abs : null;
  }
  if (typeof part.path === 'string' && part.path.trim()) {
    return path.resolve(part.path);
  }
  return null;
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
 * Image files referenced by messages that live inside the sessions directory.
 *
 * @param {Array<object>} messages
 * @param {string} sessionsRoot
 * @returns {string[]}
 */
function sessionSidecarPaths(messages, sessionsRoot) {
  if (!sessionsRoot) {
    return [];
  }
  const found = [];
  const seen = new Set();
  for (const msg of messages || []) {
    for (const part of imagePartsOf(msg)) {
      const abs = resolveImagePartPath(part, { sessionsRoot });
      if (!abs || !isFileInside(sessionsRoot, abs) || seen.has(abs)) {
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
 * @returns {Promise<string[]>}
 */
async function deleteSessionSidecarImages(messages, sessionsRoot) {
  const paths = sessionSidecarPaths(messages, sessionsRoot);
  for (const filePath of paths) {
    await fs.rm(filePath, { force: true });
  }
  return paths;
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
    const src = resolveImagePartPath(part, roots);
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
  resolveImagePartPath,
  sessionSidecarPaths,
  uniqueOutputFileName,
};
