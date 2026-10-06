'use strict';

/**
 * stable-diffusion.cpp engine bridge: spawns bundled sd-cli once per image.
 * Empty --backend lets ggml pick a GPU; GLAUX_FORCE_CPU=1 passes --backend cpu.
 * Exports match engines/huggingface/engine.js and engines/transcribecpp/engine.js.
 * One-shot CLI (like transcribe-cli), not a long-lived server.
 */

const path = require('path');
const { resolveLocalGgufPaths } = require('../common/modelFormat');
const { readModelPipelineTag } = require('../common/pipelineTag');
const {
  IMAGE_TO_IMAGE_PIPELINE_TAG,
  isDiffusionPipelineTag,
} = require('../common/resolveEngineId');
const { pickSdCli } = require('./cli');
const { generateImage } = require('./generate');

let closed = false;

/** @type {string | null} */
let modelsCacheDir = null;
/** @type {string | null} */
let resourcesRoot = null;
/** @type {string | null} */
let outputsRoot = null;

/** @type {string | null} */
let activeModelId = null;
/** @type {string | null} */
let activeModelPath = null;
/** @type {string | null} */
let activeModelRoot = null;
/** @type {string | null} */
let pipelineTag = null;
/** @type {AbortController | null} */
let activeChatAbort = null;

/**
 * Ephemeral working buffer for the current inference call.
 * Canonical history lives in engines/contextManager.js.
 * Text-to-image does not feed prior turns into sd-cli.
 * @type {Array<object>}
 */
let CONTEXT = [];

let cachedUsage = { used: 0, total: null, valid: false };

/**
 * @param {{ modelsCacheDir?: string, resourcesRoot?: string, outputsRoot?: string }} [options]
 */
function configure(options = {}) {
  if (typeof options.modelsCacheDir === 'string' && options.modelsCacheDir.trim()) {
    modelsCacheDir = path.resolve(options.modelsCacheDir);
  }
  if (typeof options.resourcesRoot === 'string' && options.resourcesRoot.trim()) {
    resourcesRoot = path.resolve(options.resourcesRoot);
  }
  if (typeof options.outputsRoot === 'string' && options.outputsRoot.trim()) {
    outputsRoot = path.resolve(options.outputsRoot);
  }
  return Promise.resolve();
}

function clearActive() {
  activeModelId = null;
  activeModelPath = null;
  activeModelRoot = null;
  pipelineTag = null;
  CONTEXT = [];
  cachedUsage = { used: 0, total: null, valid: false };
  if (activeChatAbort) {
    try {
      activeChatAbort.abort();
    } catch {
      /* ignore */
    }
    activeChatAbort = null;
  }
}

/**
 * @param {string} modelId
 * @param {{ onProgress?: Function }} [options]
 */
async function chatbotCreate(modelId, options = {}) {
  if (closed) {
    throw new Error('Engine closed');
  }
  if (!modelsCacheDir) {
    throw new Error('Models cache directory is not configured.');
  }
  const onProgress =
    typeof options === 'function' ? options : options && options.onProgress;

  clearActive();
  pickSdCli();

  const tag = (await readModelPipelineTag(modelsCacheDir, modelId)) || null;
  if (!isDiffusionPipelineTag(tag)) {
    throw new Error(
      'stablediffusion engine requires pipeline_tag: text-to-image or image-to-image ' +
        `(got ${tag || 'none'}). Chat GGUFs should use the llama.cpp engine.`
    );
  }

  const modelRoot = path.join(modelsCacheDir, ...modelId.split('/'));
  const { modelPath } = await resolveLocalGgufPaths(modelRoot);

  if (onProgress) {
    onProgress({ status: 'progress', loaded: 0, total: 100, percent: 0 });
  }

  activeModelId = modelId;
  activeModelPath = modelPath;
  activeModelRoot = modelRoot;
  pipelineTag = tag;
  cachedUsage = { used: 0, total: null, valid: false };

  if (onProgress) {
    onProgress({ status: 'progress', loaded: 100, total: 100, percent: 100 });
  }
}

async function chatbotDestroy() {
  clearActive();
}

function contextClear() {
  CONTEXT = [];
  cachedUsage = { used: 0, total: null, valid: false };
  return Promise.resolve();
}

function contextSnapshot() {
  return Promise.resolve(JSON.parse(JSON.stringify(CONTEXT)));
}

async function contextReplace(newMessages) {
  CONTEXT = Array.isArray(newMessages) ? JSON.parse(JSON.stringify(newMessages)) : [];
  cachedUsage.valid = false;
}

/**
 * Text-to-image has no token context window; gauge is unsupported.
 */
async function contextUsage() {
  return { used: 0, total: null, supported: false };
}

function chatbotSupportsThinking() {
  return Promise.resolve(false);
}

function chatbotHasChatTemplate() {
  return Promise.resolve(false);
}

function chatGenerationStartsInThinking() {
  return Promise.resolve(false);
}

/**
 * @param {string} modelId
 * @param {boolean} _thinking
 * @param {string} message
 * @param {{ onToken?: Function, imagePaths?: string[], audioPaths?: string[], videoPaths?: string[], messages?: Array<object>, outputPath?: string }} [options]
 * @returns {Promise<string | { text: string, imagePaths: string[] }>}
 */
async function runChat(modelId, _thinking, message, options = {}) {
  if (!activeModelId || activeModelId !== modelId || !activeModelPath) {
    throw new Error('Model is not loaded.');
  }
  const opts = typeof options === 'function' ? { onToken: options } : options || {};
  const audioPaths = opts.audioPaths || [];
  const imagePaths = opts.imagePaths || [];
  const videoPaths = opts.videoPaths || [];

  // Prior turns stay in the session. sd-cli sees only this prompt.
  CONTEXT = [];

  const allowImage = pipelineTag === IMAGE_TO_IMAGE_PIPELINE_TAG;
  if (audioPaths.length > 0 || videoPaths.length > 0 || (!allowImage && imagePaths.length > 0)) {
    throw new Error(
      allowImage
        ? 'Image-to-image models accept a text prompt and an optional image.'
        : 'Text-to-image models accept a text prompt only.'
    );
  }
  if (imagePaths.length > 1) {
    throw new Error('Image-to-image models accept at most one image.');
  }
  const prompt = typeof message === 'string' ? message.trim() : '';
  if (!prompt) {
    throw new Error('Text-to-image requires a prompt.');
  }
  if (typeof opts.outputPath !== 'string' || !opts.outputPath.trim()) {
    throw new Error('Text-to-image output path is not configured.');
  }

  const abort = new AbortController();
  activeChatAbort = abort;

  try {
    const written = await generateImage(activeModelPath, prompt, opts.outputPath, {
      signal: abort.signal,
      modelRoot: activeModelRoot || undefined,
      modelsCacheDir,
      initImage: imagePaths[0] || null,
    });
    cachedUsage.valid = false;
    if (!written) {
      return '';
    }
    return { text: '', imagePaths: [written] };
  } catch (err) {
    if (abort.signal.aborted || /abort/i.test(String(err && err.message))) {
      return '';
    }
    throw err;
  } finally {
    if (activeChatAbort === abort) {
      activeChatAbort = null;
    }
  }
}

function chatStop() {
  if (activeChatAbort) {
    try {
      activeChatAbort.abort();
    } catch {
      /* ignore */
    }
  }
  return Promise.resolve();
}

function close(options = {}) {
  const final = options.final !== false;
  if (final) {
    closed = true;
  }
  clearActive();
  return Promise.resolve();
}

function downloadModel() {
  return Promise.reject(new Error('Use the Hub downloader via engineManager.downloadModel.'));
}

function cancelDownloadModel() {
  return Promise.resolve();
}

module.exports = {
  configure,
  close,
  downloadModel,
  cancelDownloadModel,
  chatbotCreate,
  chatbotDestroy,
  contextClear,
  contextSnapshot,
  contextUsage,
  contextReplace,
  chatbotSupportsThinking,
  chatbotHasChatTemplate,
  chatGenerationStartsInThinking,
  runChat,
  chatStop,
};
