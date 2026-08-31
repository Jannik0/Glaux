'use strict';

/**
 * transcribe.cpp engine bridge: spawns bundled transcribe-cli per transcription.
 * GPU backends are selected with --backend auto (GLAUX_FORCE_CPU=1 forces CPU).
 * Exports match engines/huggingface/engine.js and engines/llamacpp/engine.js.
 */

const path = require('path');
const { resolveLocalGgufPaths } = require('../common/modelFormat');
const { readModelPipelineTag } = require('../common/pipelineTag');
const { ASR_PIPELINE_TAG } = require('../common/resolveEngineId');
const { pickTranscribeCli } = require('./cli');
const { transcribeAudio } = require('./asr');

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
let pipelineTag = null;
/** @type {AbortController | null} */
let activeChatAbort = null;

/**
 * Ephemeral working buffer for the current inference call.
 * Canonical history lives in engines/contextManager.js.
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

  // Fail fast if the CLI is missing.
  pickTranscribeCli();

  const tag = (await readModelPipelineTag(modelsCacheDir, modelId)) || null;
  if (tag !== ASR_PIPELINE_TAG) {
    throw new Error(
      `transcribecpp engine requires pipeline_tag: ${ASR_PIPELINE_TAG} ` +
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
 * ASR has no token context window; gauge is unsupported.
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
 * @param {{ onToken?: Function, imagePaths?: string[], audioPaths?: string[], videoPaths?: string[], resubmit?: boolean, messages?: Array<object> }} [options]
 */
async function runChat(modelId, _thinking, message, options = {}) {
  if (!activeModelId || activeModelId !== modelId || !activeModelPath) {
    throw new Error('Model is not loaded.');
  }
  const opts = typeof options === 'function' ? { onToken: options } : options || {};
  const audioPaths = opts.audioPaths || [];
  const imagePaths = opts.imagePaths || [];
  const videoPaths = opts.videoPaths || [];

  if (Array.isArray(opts.messages)) {
    CONTEXT = JSON.parse(JSON.stringify(opts.messages));
  } else {
    const content = [];
    if (typeof message === 'string' && message.trim()) {
      content.push({ type: 'text', text: message });
    }
    for (const p of audioPaths) {
      content.push({ type: 'audio', path: p });
    }
    CONTEXT = [{ role: 'user', content }];
  }

  if (imagePaths.length > 0 || videoPaths.length > 0) {
    throw new Error('Automatic speech recognition accepts audio attachments only.');
  }
  if (audioPaths.length === 0) {
    throw new Error('Automatic speech recognition requires an audio attachment.');
  }

  const abort = new AbortController();
  activeChatAbort = abort;

  try {
    const reply = await transcribeAudio(activeModelPath, activeModelId, audioPaths[0], {
      onToken: opts.onToken,
      onReplace: opts.onReplace,
      signal: abort.signal,
    });
    cachedUsage.valid = false;
    return reply;
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
