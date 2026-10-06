'use strict';

/**
 * A text-to-image GGUF is often the diffusion model only. sd-cli then needs
 * --vae and a text encoder (--clip_l / --clip_g / --t5xxl / --llm). Those
 * files live in the base repo's cache folder, the same path a download of
 * that repo uses. They are fetched there when the GGUF is downloaded, and
 * again at run time if that folder no longer has them.
 */

const fs = require('fs');
const path = require('path');

const REPO_ID_RE = /^[^/\s]+\/[^/\s]+$/;
const BASE_MODEL_INLINE_RE = /^base_model:\s*['"]?([^'"\s#][^'"\n#]*?)['"]?\s*$/m;
const BASE_MODEL_LIST_RE = /^base_model:\s*\r?\n[ \t]*-[ \t]*['"]?([^'"\s#][^'"\n#]*?)['"]?\s*$/m;
const ENCODER_DIRS = ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'text_encoders'];

/**
 * @param {string} modelRoot
 * @returns {string | null}
 */
function repoIdFrom(value) {
  const id = String(value || '').trim();
  return REPO_ID_RE.test(id) ? id : null;
}

/**
 * First Hub id in the model card. Accepts a scalar or a YAML list.
 * @param {string} modelRoot
 * @returns {string | null}
 */
function readBaseModelId(modelRoot) {
  let content;
  try {
    content = fs.readFileSync(path.join(modelRoot, 'README.md'), 'utf8');
  } catch {
    return null;
  }
  const front = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  const card = front ? front[1] : content;
  const inline = card.match(BASE_MODEL_INLINE_RE);
  if (inline) {
    const id = repoIdFrom(inline[1]);
    if (id) {
      return id;
    }
  }
  const list = card.match(BASE_MODEL_LIST_RE);
  if (list) {
    return repoIdFrom(list[1]);
  }
  return null;
}

/**
 * @param {string} dir
 * @returns {string[]}
 */
function weightNames(dir) {
  let names;
  try {
    names = fs.readdirSync(dir);
  } catch {
    return [];
  }
  return names.filter(
    (name) =>
      name === 'model.safetensors.index.json' ||
      name === 'diffusion_pytorch_model.safetensors.index.json' ||
      (/\.(safetensors|gguf)$/i.test(name) && !name.includes('.index.'))
  );
}

/**
 * A single weight file, or the directory when the component is sharded.
 * @param {string} dir
 * @returns {string | null}
 */
function componentPath(dir) {
  const names = weightNames(dir);
  if (!names.length) {
    return null;
  }
  const files = names.filter((name) => !name.endsWith('.index.json'));
  if (files.length === 1 && names.length === 1) {
    return path.join(dir, files[0]);
  }
  return dir;
}

/**
 * @param {string} dir
 * @param {string} folderName
 * @returns {'clipL' | 'clipG' | 't5xxl' | 'llm'}
 */
function encoderRole(dir, folderName) {
  let blob = folderName.toLowerCase();
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'config.json'), 'utf8'));
    const arch = Array.isArray(cfg.architectures) ? cfg.architectures.join(' ') : '';
    blob = `${arch} ${cfg.model_type || ''} ${folderName}`.toLowerCase();
  } catch {
    /* folder name only */
  }
  if (/t5/.test(blob)) {
    return 't5xxl';
  }
  if (/clip/.test(blob)) {
    if (/withprojection|open_?clip|clip[-_]?g|_2$/.test(blob)) {
      return 'clipG';
    }
    return 'clipL';
  }
  return 'llm';
}

/**
 * @param {string} fileName
 * @returns {'vae' | 'clipL' | 'clipG' | 't5xxl' | 'llm' | null}
 */
function fileRole(fileName) {
  const base = fileName.toLowerCase();
  if (!/\.(safetensors|gguf)$/.test(base) || base.includes('.index.')) {
    return null;
  }
  if (/vae/.test(base)) {
    return 'vae';
  }
  if (/clip[-_]?l/.test(base)) {
    return 'clipL';
  }
  if (/clip[-_]?g/.test(base)) {
    return 'clipG';
  }
  if (/t5/.test(base)) {
    return 't5xxl';
  }
  if (/(^|[^a-z])llm([^a-z]|$)|text[-_]?encoder/.test(base)) {
    return 'llm';
  }
  return null;
}

/**
 * @returns {{
 *   vae: string | null,
 *   clipL: string | null,
 *   clipG: string | null,
 *   t5xxl: string | null,
 *   llm: string | null,
 * }}
 */
function emptyComponents() {
  return { vae: null, clipL: null, clipG: null, t5xxl: null, llm: null };
}

/**
 * @param {{ vae: string | null, clipL: string | null, clipG: string | null, t5xxl: string | null, llm: string | null }} components
 * @returns {boolean}
 */
function hasConditioner(components) {
  return Boolean(components.clipL || components.clipG || components.t5xxl || components.llm);
}

/**
 * @param {{ vae: string | null, clipL: string | null, clipG: string | null, t5xxl: string | null, llm: string | null }} components
 * @returns {boolean}
 */
function hasSplitWeights(components) {
  return Boolean(components.vae || hasConditioner(components));
}

/**
 * Weights that live in this model directory only.
 * @param {string} modelRoot
 * @returns {{
 *   vae: string | null,
 *   clipL: string | null,
 *   clipG: string | null,
 *   t5xxl: string | null,
 *   llm: string | null,
 * }}
 */
function findLocalComponents(modelRoot) {
  const found = emptyComponents();
  if (!modelRoot || !fs.existsSync(modelRoot)) {
    return found;
  }

  const vaePath = componentPath(path.join(modelRoot, 'vae'));
  if (vaePath) {
    found.vae = vaePath;
  }

  for (const name of ENCODER_DIRS) {
    const dir = path.join(modelRoot, name);
    const component = componentPath(dir);
    if (!component) {
      continue;
    }
    const role = encoderRole(dir, name);
    if (!found[role]) {
      found[role] = component;
    }
  }

  let rootNames;
  try {
    rootNames = fs.readdirSync(modelRoot);
  } catch {
    return found;
  }
  for (const name of rootNames) {
    const role = fileRole(name);
    if (!role || found[role]) {
      continue;
    }
    const full = path.join(modelRoot, name);
    try {
      if (!fs.statSync(full).isFile()) {
        continue;
      }
    } catch {
      continue;
    }
    found[role] = full;
  }
  return found;
}

/**
 * The base repo folder already has a VAE and a text encoder.
 * @param {string} modelRoot
 * @returns {boolean}
 */
function baseCompanionsReady(modelRoot) {
  const found = findLocalComponents(modelRoot);
  return Boolean(found.vae && hasConditioner(found));
}

/**
 * Base repo id when this GGUF names one and that cache folder lacks a VAE and
 * a text encoder. Null when there is nothing to fetch.
 * @param {string} modelRoot
 * @param {string | null | undefined} modelsCacheDir
 * @returns {string | null}
 */
function missingBaseCompanions(modelRoot, modelsCacheDir) {
  const baseModelId = readBaseModelId(modelRoot);
  if (!baseModelId) {
    return null;
  }
  if (!modelsCacheDir || !String(modelsCacheDir).trim()) {
    return baseModelId;
  }
  const baseRoot = path.join(modelsCacheDir, ...baseModelId.split('/'));
  return baseCompanionsReady(baseRoot) ? null : baseModelId;
}

/**
 * Weights sd-cli should load. A card that names a base repo uses that repo's
 * cache folder. A card without one uses files in the GGUF directory.
 * @param {string} modelRoot
 * @param {string | null | undefined} modelsCacheDir
 * @returns {{
 *   vae: string | null,
 *   clipL: string | null,
 *   clipG: string | null,
 *   t5xxl: string | null,
 *   llm: string | null,
 * }}
 */
function resolveRunComponents(modelRoot, modelsCacheDir) {
  const baseModelId = readBaseModelId(modelRoot);
  if (!baseModelId) {
    return findLocalComponents(modelRoot);
  }
  const baseRoot =
    modelsCacheDir && String(modelsCacheDir).trim()
      ? path.join(modelsCacheDir, ...baseModelId.split('/'))
      : '';
  if (!baseRoot || !baseCompanionsReady(baseRoot)) {
    throw new Error(
      `This model needs the VAE and text encoder from ${baseModelId}. They are not in the local cache.`
    );
  }
  return findLocalComponents(baseRoot);
}

module.exports = {
  readBaseModelId,
  findLocalComponents,
  baseCompanionsReady,
  missingBaseCompanions,
  resolveRunComponents,
  hasSplitWeights,
  hasConditioner,
};
