'use strict';

/**
 * Task id for a cached model. `pipeline_tag` in the model-card front matter
 * wins. When that field is absent, `automatic-speech-recognition`,
 * `text-to-image`, or `image-to-image` in the card's `tags` list is used
 * for engine routing.
 */

const fs = require('fs').promises;
const path = require('path');

const {
  ASR_PIPELINE_TAG,
  IMAGE_TO_IMAGE_PIPELINE_TAG,
  TEXT_TO_IMAGE_PIPELINE_TAG,
} = require('./resolveEngineId');

const FRONT_MATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---/;
const PIPELINE_TAG_RE = /^pipeline_tag:[ \t]*(.*?)[ \t]*$/m;
const TAGS_FLOW_RE = /^tags:[ \t]*\[([^\]]*)\]/m;
const TAGS_BLOCK_RE = /^tags:[ \t]*\r?\n((?:[ \t]*-[^\n]*(?:\r?\n|$))*)/m;
const TAGS_SCALAR_RE = /^tags:[ \t]*(\S.*?)[ \t]*$/m;
const TAG_ITEM_RE = /^[ \t]*-[ \t]*(.*?)\s*$/;

const TASK_TAGS = new Set([
  ASR_PIPELINE_TAG,
  TEXT_TO_IMAGE_PIPELINE_TAG,
  IMAGE_TO_IMAGE_PIPELINE_TAG,
]);

/**
 * @param {string} value
 * @returns {string}
 */
function unquote(value) {
  return String(value || '')
    .trim()
    .replace(/^['"]|['"]$/g, '')
    .trim();
}

/**
 * @param {string} card
 * @returns {string[]}
 */
function cardTags(card) {
  const flow = card.match(TAGS_FLOW_RE);
  if (flow) {
    return flow[1]
      .split(',')
      .map((part) => unquote(part))
      .filter(Boolean);
  }
  const block = card.match(TAGS_BLOCK_RE);
  if (block && block[1].trim()) {
    const tags = [];
    for (const line of block[1].split(/\r?\n/)) {
      const item = line.match(TAG_ITEM_RE);
      if (!item) {
        continue;
      }
      const tag = unquote(item[1]);
      if (tag) {
        tags.push(tag);
      }
    }
    return tags;
  }
  const scalar = card.match(TAGS_SCALAR_RE);
  if (!scalar || scalar[1].startsWith('|') || scalar[1].startsWith('>')) {
    return [];
  }
  const tag = unquote(scalar[1]);
  return tag ? [tag] : [];
}

/**
 * @param {string[]} tags
 * @returns {string | null}
 */
function taskTagFromList(tags) {
  for (const tag of tags) {
    if (TASK_TAGS.has(tag)) {
      return tag;
    }
  }
  return null;
}

/**
 * @param {string} content README.md text.
 * @returns {string | null}
 */
function pipelineTagFromCard(content) {
  const text = String(content || '');
  const front = text.match(FRONT_MATTER_RE);
  const card = front ? front[1] : text;
  const declared = card.match(PIPELINE_TAG_RE);
  if (declared) {
    const tag = unquote(declared[1]);
    if (tag) {
      return tag;
    }
  }
  return taskTagFromList(cardTags(card));
}

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
    return pipelineTagFromCard(content);
  } catch {
    return null;
  }
}

module.exports = {
  pipelineTagFromCard,
  readModelPipelineTag,
};
