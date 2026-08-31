'use strict';

/**
 * Shared reader for the `pipeline_tag` YAML frontmatter field in a cached model's
 * README.md. Used by engineManager.js, llamacpp/engine.js, and (optionally)
 * src/main/domains/modelsPrefs.js.
 */

const fs = require('fs').promises;
const path = require('path');

const MODEL_PIPELINE_TAG_RE = /^pipeline_tag:\s*(.+)\s*$/m;

/**
 * @param {string} cacheRoot Root directory containing `<namespace>/<repo>/README.md`.
 * @param {string} modelId Hub-style model id, e.g. `namespace/repo`.
 * @returns {Promise<string | null>}
 */
async function readModelPipelineTag(cacheRoot, modelId) {
  if (!cacheRoot || !modelId) {
    return null;
  }
  const readmePath = path.join(cacheRoot, ...modelId.split('/'), 'README.md');
  try {
    const content = await fs.readFile(readmePath, 'utf8');
    const match = content.match(MODEL_PIPELINE_TAG_RE);
    if (!match) {
      return null;
    }
    return match[1].trim().replace(/^['"]|['"]$/g, '');
  } catch {
    return null;
  }
}

module.exports = {
  MODEL_PIPELINE_TAG_RE,
  readModelPipelineTag,
};
