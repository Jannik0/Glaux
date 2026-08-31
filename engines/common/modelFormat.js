'use strict';

/**
 * Shared helpers for detecting model weight format and selecting GGUF files.
 *
 * Distinguishes Hugging Face weights (safetensors / pytorch .bin) from GGUF,
 * groups Hub GGUF files by quant variant, builds download allow-patterns for a
 * chosen variant, and resolves which .gguf (+ optional mmproj) to load locally.
 */

const fs = require('fs');

const fsp = require('fs').promises;
const path = require('path');

const GGUF_SELECTION_FILE = '.glaux-gguf-selection.json';

/** Quant / precision tokens commonly used in GGUF filenames. */
const QUANT_RE =
  /(?:^|[._-])((?:IQ|UD)\d[_A-Z0-9]*|Q\d+(?:_[A-Z0-9_]+)?|F16|F32|BF16|FP16|FP32)(?=$|[._-])/i;

const SHARD_RE = /-(\d{5})-of-(\d{5})/i;

/**
 * @param {string} name
 * @returns {boolean}
 */
function isMmprojName(name) {
  return /^mmproj/i.test(path.basename(name));
}

/**
 * @param {string} name
 * @returns {boolean}
 */
function isGgufName(name) {
  return /\.gguf$/i.test(name);
}

/**
 * @param {string} fileName
 * @returns {string | null}
 */
function extractQuantKey(fileName) {
  const base = path.basename(fileName).replace(/\.gguf$/i, '');
  const withoutShard = base.replace(SHARD_RE, '');
  const match = withoutShard.match(QUANT_RE);
  if (!match) {
    return null;
  }
  return match[1].toUpperCase();
}

/**
 * @param {string} fileName
 * @returns {string}
 */
function variantGroupKey(fileName) {
  const quant = extractQuantKey(fileName);
  if (quant) {
    return quant;
  }
  // Fall back to basename without shard suffix so split files group together.
  return path
    .basename(fileName)
    .replace(/\.gguf$/i, '')
    .replace(SHARD_RE, '')
    .toLowerCase();
}

/**
 * @param {string} dir
 * @param {number} [depth]
 * @returns {Promise<{ hasSafetensors: boolean, hasPytorchBin: boolean, hasGguf: boolean, hasTextGguf: boolean }>}
 */
async function scanWeightKinds(dir, depth = 0) {
  const result = {
    hasSafetensors: false,
    hasPytorchBin: false,
    hasGguf: false,
    hasTextGguf: false,
  };
  if (depth > 48) {
    return result;
  }
  let entries;
  try {
    entries = await fsp.readdir(dir, { withFileTypes: true });
  } catch {
    return result;
  }
  for (const ent of entries) {
    const full = path.join(dir, ent.name);
    if (ent.isFile()) {
      if (/\.safetensors$/i.test(ent.name)) {
        result.hasSafetensors = true;
      } else if (/^pytorch_model.*\.bin$/i.test(ent.name)) {
        result.hasPytorchBin = true;
      } else if (isGgufName(ent.name)) {
        result.hasGguf = true;
        if (!isMmprojName(ent.name)) {
          result.hasTextGguf = true;
        }
      }
    } else if (ent.isDirectory()) {
      const nested = await scanWeightKinds(full, depth + 1);
      result.hasSafetensors = result.hasSafetensors || nested.hasSafetensors;
      result.hasPytorchBin = result.hasPytorchBin || nested.hasPytorchBin;
      result.hasGguf = result.hasGguf || nested.hasGguf;
      result.hasTextGguf = result.hasTextGguf || nested.hasTextGguf;
    }
    if (result.hasSafetensors && result.hasPytorchBin && result.hasTextGguf) {
      break;
    }
  }
  return result;
}

/**
 * @param {string} modelsCacheDir
 * @param {string} modelId
 * @returns {Promise<'huggingface' | 'llamacpp' | null>}
 */
async function detectModelFormat(modelsCacheDir, modelId) {
  if (!modelsCacheDir || !modelId) {
    return null;
  }
  const modelRoot = path.join(modelsCacheDir, ...String(modelId).split('/'));
  const kinds = await scanWeightKinds(modelRoot);
  if (kinds.hasSafetensors || kinds.hasPytorchBin) {
    return 'huggingface';
  }
  if (kinds.hasTextGguf) {
    return 'llamacpp';
  }
  return null;
}

/**
 * @param {string} dir
 * @param {number} [depth]
 * @returns {Promise<boolean>}
 */
async function directoryContainsModelWeights(dir, depth = 0) {
  const kinds = await scanWeightKinds(dir, depth);
  return kinds.hasSafetensors || kinds.hasPytorchBin || kinds.hasTextGguf;
}

/**
 * @param {{ path: string, size?: number }[]} files
 * @returns {{ kind: 'huggingface' | 'gguf' | 'unknown', variants: Array<{ key: string, label: string, files: string[], size: number }> }}
 */
function classifyHubRepoFiles(files) {
  const list = Array.isArray(files) ? files : [];
  let hasSafetensors = false;
  let hasPytorchBin = false;
  /** @type {Map<string, { key: string, files: string[], size: number }>} */
  const ggufGroups = new Map();

  for (const entry of list) {
    const filePath = typeof entry === 'string' ? entry : entry && entry.path;
    if (!filePath || typeof filePath !== 'string') {
      continue;
    }
    const base = path.basename(filePath).replace(/\\/g, '/');
    const size = typeof entry === 'object' && entry && typeof entry.size === 'number' ? entry.size : 0;
    if (/\.safetensors$/i.test(base)) {
      hasSafetensors = true;
    } else if (/^pytorch_model.*\.bin$/i.test(base)) {
      hasPytorchBin = true;
    } else if (isGgufName(base) && !isMmprojName(base)) {
      const key = variantGroupKey(base);
      let group = ggufGroups.get(key);
      if (!group) {
        group = { key, files: [], size: 0 };
        ggufGroups.set(key, group);
      }
      group.files.push(filePath.replace(/\\/g, '/'));
      group.size += size;
    }
  }

  if (hasSafetensors || hasPytorchBin) {
    return { kind: 'huggingface', variants: [] };
  }

  if (ggufGroups.size > 0) {
    const variants = [...ggufGroups.values()]
      .map((g) => ({
        key: g.key,
        label: formatVariantLabel(g.key, g.size, g.files.length),
        files: g.files.sort(),
        size: g.size,
      }))
      .sort((a, b) => a.key.localeCompare(b.key, undefined, { sensitivity: 'base' }));
    return { kind: 'gguf', variants };
  }

  return { kind: 'unknown', variants: [] };
}

/**
 * @param {string} key
 * @param {number} size
 * @param {number} fileCount
 */
function formatVariantLabel(key, size, fileCount) {
  const sizeLabel = formatBytes(size);
  const shardNote = fileCount > 1 ? ` · ${fileCount} shards` : '';
  return sizeLabel ? `${key} (${sizeLabel}${shardNote})` : `${key}${shardNote}`;
}

/**
 * @param {number} n
 * @returns {string}
 */
function formatBytes(n) {
  if (!n || n <= 0) {
    return '';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)}${units[i]}`;
}

/**
 * Build snapshot_download allow_patterns for a chosen GGUF variant.
 *
 * @param {{ path: string, size?: number }[]} allFiles
 * @param {{ key: string, files: string[] }} variant
 * @returns {string[]}
 */
function buildGgufAllowPatterns(allFiles, variant) {
  const patterns = new Set();
  for (const f of variant.files || []) {
    patterns.add(f.replace(/\\/g, '/'));
  }
  for (const entry of allFiles || []) {
    const filePath = (typeof entry === 'string' ? entry : entry && entry.path || '')
      .replace(/\\/g, '/');
    if (!filePath) {
      continue;
    }
    const base = path.basename(filePath);
    if (isMmprojName(base) && isGgufName(base)) {
      patterns.add(filePath);
      continue;
    }
    if (/^readme\.md$/i.test(base)) {
      patterns.add(filePath);
      continue;
    }
    // Sidecar tokenizer / config files (not other weight formats).
    if (
      /\.(json|txt|model|jinja)$/i.test(base) ||
      /^tokenizer/i.test(base) ||
      /^vocab/i.test(base) ||
      /^merges\.txt$/i.test(base) ||
      /^special_tokens_map\.json$/i.test(base) ||
      /^generation_config\.json$/i.test(base) ||
      /^config\.json$/i.test(base) ||
      /^chat_template/i.test(base)
    ) {
      patterns.add(filePath);
    }
  }
  return [...patterns];
}

/**
 * @param {string} modelRoot
 * @returns {{ variant: string, files?: string[] } | null}
 */
function readGgufSelection(modelRoot) {
  try {
    const raw = fs.readFileSync(path.join(modelRoot, GGUF_SELECTION_FILE), 'utf8');
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.variant === 'string') {
      return parsed;
    }
  } catch {
    /* missing or invalid */
  }
  return null;
}

/**
 * @param {string} modelRoot
 * @param {{ variant: string, files?: string[] }} selection
 */
function writeGgufSelection(modelRoot, selection) {
  fs.mkdirSync(modelRoot, { recursive: true });
  fs.writeFileSync(
    path.join(modelRoot, GGUF_SELECTION_FILE),
    JSON.stringify(selection, null, 2),
    'utf8'
  );
}

/**
 * Collect GGUF files under a model root.
 *
 * @param {string} dir
 * @param {number} [depth]
 * @returns {Promise<{ text: string[], mmproj: string[] }>}
 */
async function listLocalGgufFiles(dir, depth = 0) {
  const text = [];
  const mmproj = [];
  if (depth > 48) {
    return { text, mmproj };
  }
  let entries;
  try {
    entries = await fsp.readdir(dir, { withFileTypes: true });
  } catch {
    return { text, mmproj };
  }
  for (const ent of entries) {
    const full = path.join(dir, ent.name);
    if (ent.isFile() && isGgufName(ent.name)) {
      if (isMmprojName(ent.name)) {
        mmproj.push(full);
      } else {
        text.push(full);
      }
    } else if (ent.isDirectory()) {
      const nested = await listLocalGgufFiles(full, depth + 1);
      text.push(...nested.text);
      mmproj.push(...nested.mmproj);
    }
  }
  return { text, mmproj };
}

/**
 * Pick the text GGUF (and optional mmproj) to load for a cached model.
 *
 * @param {string} modelRoot
 * @returns {Promise<{ modelPath: string, mmprojPath: string | null }>}
 */
async function resolveLocalGgufPaths(modelRoot) {
  const { text, mmproj } = await listLocalGgufFiles(modelRoot);
  if (!text.length) {
    throw new Error('No GGUF model file found in the model cache.');
  }

  const selection = readGgufSelection(modelRoot);
  let chosen = text;

  if (selection && selection.variant) {
    const matched = text.filter((p) => variantGroupKey(path.basename(p)) === selection.variant);
    if (matched.length) {
      chosen = matched;
    }
  } else if (selection && Array.isArray(selection.files) && selection.files.length) {
    const wanted = new Set(selection.files.map((f) => f.replace(/\\/g, '/')));
    const matched = text.filter((p) => {
      const rel = path.relative(modelRoot, p).split(path.sep).join('/');
      return wanted.has(rel) || wanted.has(path.basename(p));
    });
    if (matched.length) {
      chosen = matched;
    }
  }

  // Prefer the first shard (00001) or the single file; sort for stability.
  chosen.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
  const modelPath = chosen[0];
  const mmprojPath = mmproj.length ? mmproj.sort()[0] : null;
  return { modelPath, mmprojPath };
}

module.exports = {
  GGUF_SELECTION_FILE,
  detectModelFormat,
  directoryContainsModelWeights,
  scanWeightKinds,
  classifyHubRepoFiles,
  buildGgufAllowPatterns,
  readGgufSelection,
  writeGgufSelection,
  resolveLocalGgufPaths,
  listLocalGgufFiles,
  extractQuantKey,
  variantGroupKey,
  isMmprojName,
  isGgufName,
};
