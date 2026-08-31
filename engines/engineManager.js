/**
 * Inference facade. The rest of the app talks to engineManager only.
 * Routes by weight format and pipeline_tag:
 *   safetensors/pytorch → huggingface
 *   GGUF + ASR          → transcribecpp
 *   GGUF (chat/other)   → llamacpp
 * Hub downloads always use the huggingface downloader stack.
 */

const fs = require('fs').promises;
const path = require('path');
const hfEngine = require('./huggingface/engine');
const llamaEngine = require('./llamacpp/engine');
const transcribeEngine = require('./transcribecpp/engine');
const contextManager = require('./contextManager');
const { detectModelFormat } = require('./common/modelFormat');
const { stripThinkingFromMessages } = require('./common/stripThinking');
const { withStopMarker } = require('./common/stopMarker');
const {
  MARKDOWN_EXTS,
  PDF_EXTS,
  extOf,
  mediaKindFromPath,
  isDocumentPath,
} = require('./common/mediaKinds');
const { extractPdfToMarkdown } = require('./common/extractPdf');
const { extractVideoToWav } = require('./common/extractVideo');
const { readModelPipelineTag: readPipelineTagFromCache } = require('./common/pipelineTag');
const { ASR_PIPELINE_TAG, resolveEngineId } = require('./common/resolveEngineId');

/** @type {{ resourcesRoot?: string, outputsRoot?: string, modelsCacheDir?: string, modelId?: string, onProgress?: (info: object) => void } | null} */
let initOptions = null;

/** @type {string | null} */
let activeModelId = null;

/** @type {string | null} */
let activePipelineTag = null;

/** @type {'huggingface' | 'llamacpp' | 'transcribecpp' | null} */
let activeEngineId = null;

/** @type {'idle' | 'loading' | 'generating' | 'downloading'} */
let phase = 'idle';

let generationInFlight = false;

/** Set by cancelGeneration so sendPrompt can keep the partial assistant turn. */
let cancelRequested = false;

const engines = {
  huggingface: hfEngine,
  llamacpp: llamaEngine,
  transcribecpp: transcribeEngine,
};

function getActiveEngine() {
  if (!activeEngineId) {
    throw new Error('Engine not initialized.');
  }
  return engines[activeEngineId];
}

/**
 * Downloads and Hub probes always go through the Hugging Face Python worker.
 * @param {{ modelsCacheDir?: string }} [opts]
 */
async function ensureDownloadEnginePaths(opts) {
  const dir =
    (opts && typeof opts.modelsCacheDir === 'string' && opts.modelsCacheDir.trim()) ||
    (initOptions && initOptions.modelsCacheDir);
  if (!dir) {
    throw new Error('Models cache directory is not configured.');
  }
  await hfEngine.configure({ modelsCacheDir: dir });
  return dir;
}

/**
 * Configure all backends with the models cache (and media roots for native engines).
 * @param {{ modelsCacheDir?: string }} [opts]
 */
async function ensureEnginePaths(opts) {
  const dir = await ensureDownloadEnginePaths(opts);
  const mediaOpts = {
    modelsCacheDir: dir,
    resourcesRoot: initOptions && initOptions.resourcesRoot,
    outputsRoot: initOptions && initOptions.outputsRoot,
  };
  await llamaEngine.configure(mediaOpts);
  await transcribeEngine.configure(mediaOpts);
  return dir;
}

function getStatus() {
  return {
    phase: generationInFlight ? 'generating' : phase,
    modelId: activeModelId,
    pipelineTag: activePipelineTag,
    engineId: activeEngineId,
  };
}

function emitProgress(info) {
  if (initOptions && typeof initOptions.onProgress === 'function') {
    initOptions.onProgress(info);
  }
}

/**
 * @param {string} modelId
 * @returns {Promise<string | null>}
 */
async function readModelPipelineTag(modelId) {
  return readPipelineTagFromCache(initOptions?.modelsCacheDir, modelId);
}

/**
 * @param {string} modelId
 * @returns {Promise<'huggingface' | 'llamacpp' | 'transcribecpp'>}
 */
async function resolveEngineForModel(modelId) {
  const dir = initOptions && initOptions.modelsCacheDir;
  const format = await detectModelFormat(dir, modelId);
  const pipelineTag = await readModelPipelineTag(modelId);
  const engineId = resolveEngineId(format, pipelineTag);
  if (engineId) {
    return engineId;
  }
  throw new Error(
    `Could not determine model format for ${modelId}. Expected safetensors/pytorch or GGUF weights.`
  );
}

/**
 * For ASR models, extract audio from video attachments into sibling `.wav` files.
 *
 * @param {string[]} videoPaths
 * @returns {Promise<string[]>}
 */
async function convertAsrVideoToAudio(videoPaths) {
  if (activePipelineTag !== ASR_PIPELINE_TAG || !videoPaths.length) {
    return [];
  }
  const wavPaths = [];
  for (const videoPath of videoPaths) {
    wavPaths.push(await extractVideoToWav(videoPath));
  }
  return wavPaths;
}

/**
 * @param {{ source: string, relativePath: string }} file
 * @returns {string}
 */
function resolveInferenceFilePath(file) {
  if (!initOptions) {
    throw new Error('Engine not initialized.');
  }
  const rel = typeof file.relativePath === 'string' ? file.relativePath : '';
  if (!rel) {
    throw new Error('Invalid inference file path.');
  }
  const root =
    file.source === 'outputs' ? initOptions.outputsRoot : initOptions.resourcesRoot;
  if (!root) {
    throw new Error('Engine paths are not configured.');
  }
  return path.resolve(root, rel);
}

/**
 * @param {Array<{ source: string, relativePath: string }>} files
 * @returns {{ imagePaths: string[], audioPaths: string[], videoPaths: string[] }}
 */
function mediaPathsFromFiles(files) {
  const imagePaths = [];
  const audioPaths = [];
  const videoPaths = [];
  for (const file of files || []) {
    const abs = resolveInferenceFilePath(file);
    const kind = mediaKindFromPath(abs);
    if (kind === 'image') {
      imagePaths.push(abs);
    } else if (kind === 'audio') {
      audioPaths.push(abs);
    } else if (kind === 'video') {
      videoPaths.push(abs);
    }
  }
  return { imagePaths, audioPaths, videoPaths };
}

/**
 * @param {string} relativePath
 * @returns {'image' | 'audio' | 'video' | null}
 */
function mediaKindFromRelativePath(relativePath) {
  return mediaKindFromPath(relativePath);
}

/**
 * @param {Array<{ source: string, relativePath: string }>} files
 * @returns {string}
 */
function defaultMessageForMediaAttachments(files) {
  let hasAudio = false;
  let hasImage = false;
  let hasVideo = false;
  for (const file of files || []) {
    const kind = mediaKindFromRelativePath(file.relativePath);
    if (kind === 'audio') {
      hasAudio = true;
    } else if (kind === 'image') {
      hasImage = true;
    } else if (kind === 'video') {
      hasVideo = true;
    }
  }
  const videoCountsAsAudio = activePipelineTag === ASR_PIPELINE_TAG && hasVideo;
  if (hasImage || (hasVideo && !videoCountsAsAudio)) {
    return 'Describe';
  }
  if (hasAudio || videoCountsAsAudio) {
    return 'Transcribe';
  }
  return '';
}

/**
 * @param {string} filePath
 * @returns {boolean}
 */
function isDocumentAttachmentPath(filePath) {
  return isDocumentPath(filePath);
}

/**
 * @param {{ source: string, relativePath: string }} file
 * @returns {Promise<{ relativePath: string, content: string }>}
 */
async function readDocumentAttachmentContent(file) {
  const abs = resolveInferenceFilePath(file);
  const ext = extOf(abs);

  if (MARKDOWN_EXTS.has(ext)) {
    const content = await fs.readFile(abs, 'utf8');
    return { relativePath: file.relativePath, content };
  }

  if (PDF_EXTS.has(ext)) {
    const mdAbs = await extractPdfToMarkdown(abs);
    const content = await fs.readFile(mdAbs, 'utf8');
    const mdRelative = file.relativePath.replace(/\.pdf$/i, '.md');
    return { relativePath: mdRelative, content };
  }

  throw new Error(`Unsupported document attachment: ${file.relativePath}`);
}

/**
 * @param {string} relativePath
 * @param {string} content
 * @returns {string}
 */
function formatDocumentContextBlock(relativePath, content) {
  return `\n\nstart ${relativePath}\n${content}\nend ${relativePath}`;
}

/**
 * Inject markdown/PDF attachment contents into the message as plain text.
 *
 * @param {string} message
 * @param {Array<{ source: string, relativePath: string }>} files
 * @returns {Promise<{ message: string, files: Array<{ source: string, relativePath: string }> }>}
 */
async function enrichMessageWithDocumentAttachments(message, files) {
  const mediaFiles = [];
  const textParts = [typeof message === 'string' ? message : ''];

  for (const file of files || []) {
    const abs = resolveInferenceFilePath(file);
    if (isDocumentAttachmentPath(abs)) {
      const doc = await readDocumentAttachmentContent(file);
      textParts.push(formatDocumentContextBlock(doc.relativePath, doc.content));
    } else {
      mediaFiles.push(file);
    }
  }

  return { message: textParts.join(''), files: mediaFiles };
}

/**
 * @param {string} message
 * @param {Array<{ source: string, relativePath: string }>} [files]
 * @returns {Promise<{ message: string, imagePaths: string[], audioPaths: string[], videoPaths: string[] }>}
 */
async function prepareInferenceRequest(message, files) {
  const enriched = await enrichMessageWithDocumentAttachments(message, files);
  let finalMessage = enriched.message;
  if (!String(finalMessage || '').trim() && enriched.files.length > 0) {
    const defaultMessage = defaultMessageForMediaAttachments(enriched.files);
    if (defaultMessage) {
      finalMessage = defaultMessage;
    }
  }
  const { imagePaths, audioPaths, videoPaths } = mediaPathsFromFiles(enriched.files);
  const asrWavPaths = await convertAsrVideoToAudio(videoPaths);
  const finalAudioPaths = [...audioPaths, ...asrWavPaths];
  const finalVideoPaths = activePipelineTag === ASR_PIPELINE_TAG ? [] : videoPaths;
  return {
    message: finalMessage,
    imagePaths,
    audioPaths: finalAudioPaths,
    videoPaths: finalVideoPaths,
  };
}

/**
 * Configure cache paths without loading a model.
 *
 * @param {object} opts
 */
async function configurePaths(opts) {
  initOptions = {
    resourcesRoot: opts.resourcesRoot,
    outputsRoot: opts.outputsRoot,
    modelsCacheDir: opts.modelsCacheDir,
    onProgress: opts.onProgress,
  };
  await ensureEnginePaths(opts);
}

/**
 * @param {object} opts
 */
async function initialize(opts) {
  if (!opts || typeof opts.modelId !== 'string' || !opts.modelId.trim()) {
    throw new Error('Model id is required.');
  }
  initOptions = opts;
  await ensureEnginePaths(opts);
  const modelId = opts.modelId.trim();
  const engineId = await resolveEngineForModel(modelId);
  const engine = engines[engineId];

  activeModelId = modelId;
  activePipelineTag = await readModelPipelineTag(activeModelId);
  activeEngineId = engineId;
  phase = 'loading';
  emitProgress({ phase: 'loading', status: 'starting', modelId: activeModelId });
  try {
    await engine.chatbotCreate(activeModelId, {
      onProgress: (event) => {
        if (!event || typeof event !== 'object') {
          return;
        }
        emitProgress({
          phase: 'loading',
          status: 'progress',
          modelId: activeModelId,
          loaded: event.loaded,
          total: event.total,
          percent: event.percent,
        });
      },
    });
    emitProgress({ phase: 'loading', status: 'complete', modelId: activeModelId });
  } catch (err) {
    activeModelId = null;
    activePipelineTag = null;
    activeEngineId = null;
    throw err;
  } finally {
    if (!generationInFlight) {
      phase = 'idle';
    }
  }
}

/**
 * @param {object} opts
 */
async function downloadModel(opts) {
  if (!opts || typeof opts.modelId !== 'string' || !opts.modelId.trim()) {
    throw new Error('Model id is required.');
  }
  const modelId = opts.modelId.trim();
  await ensureDownloadEnginePaths(opts);
  const report = (info) => {
    if (typeof opts.onProgress === 'function') {
      opts.onProgress(info);
    } else {
      emitProgress(info);
    }
  };
  report({ phase: 'download', status: 'starting', modelId });
  await hfEngine.downloadModel(modelId, {
    onProgress: (event) => report(event),
    allowPatterns: opts.allowPatterns,
    ggufVariant: opts.ggufVariant,
  });
  report({ phase: 'download', status: 'complete', modelId });
}

/**
 * @param {string} modelId
 * @param {{ modelsCacheDir?: string }} [opts]
 */
async function cancelModelDownload(modelId, opts = {}) {
  if (!modelId || typeof modelId !== 'string' || !modelId.trim()) {
    throw new Error('Model id is required.');
  }
  await ensureDownloadEnginePaths(opts);
  return hfEngine.cancelDownloadModel(modelId.trim());
}

/**
 * List Hub repo files (name + size) for download probing / GGUF variant selection.
 *
 * @param {string} modelId
 * @param {{ modelsCacheDir?: string }} [opts]
 * @returns {Promise<Array<{ path: string, size: number }>>}
 */
async function listHubModelFiles(modelId, opts = {}) {
  if (!modelId || typeof modelId !== 'string' || !modelId.trim()) {
    throw new Error('Model id is required.');
  }
  await ensureDownloadEnginePaths(opts);
  return hfEngine.listModelFiles(modelId.trim());
}

/**
 * Reload the active model. Conversation history stays in contextManager
 * (engines do not own long-lived chat state).
 *
 * @param {object} opts
 */
async function reinitialize(opts) {
  const next = opts || initOptions;
  if (!next || typeof next.modelId !== 'string' || !next.modelId.trim()) {
    throw new Error('Model id is required.');
  }
  const nextId = next.modelId.trim();

  if (activeEngineId) {
    try {
      await engines[activeEngineId].chatbotDestroy();
    } catch {
      /* worker may not be running yet */
    }
  }

  initOptions = { ...initOptions, ...next, modelId: nextId };
  await ensureEnginePaths(initOptions);
  return initialize(initOptions);
}

/**
 * Unload the active chat model (keeps workers alive when possible).
 */
async function ejectModel() {
  if (generationInFlight) {
    throw new Error('Wait until generation finishes before ejecting the model.');
  }
  if (!activeModelId || !activeEngineId) {
    return;
  }
  try {
    await engines[activeEngineId].chatbotDestroy();
  } catch {
    /* worker may not be running yet */
  }
  activeModelId = null;
  activePipelineTag = null;
  activeEngineId = null;
  if (phase !== 'downloading') {
    phase = 'idle';
  }
  emitProgress({ phase: 'unload', status: 'complete' });
}

async function shutdown() {
  for (const engine of Object.values(engines)) {
    try {
      await engine.chatStop();
    } catch {
      /* ignore */
    }
  }
  await Promise.all(
    Object.values(engines).map((engine) => engine.close({ final: true }).catch(() => {}))
  );
  generationInFlight = false;
  phase = 'idle';
  activeModelId = null;
  activePipelineTag = null;
  activeEngineId = null;
}

async function resetInferenceWorker() {
  for (const engine of Object.values(engines)) {
    try {
      await engine.chatStop();
    } catch {
      /* ignore */
    }
  }
  await Promise.all(
    Object.values(engines).map((engine) => engine.close({ final: false }).catch(() => {}))
  );
  generationInFlight = false;
  if (phase === 'generating') {
    phase = 'idle';
  }
}

function cancelGeneration() {
  cancelRequested = true;
  if (activeEngineId) {
    engines[activeEngineId].chatStop();
  }
  generationInFlight = false;
  if (phase === 'generating') {
    phase = 'idle';
  }
}

/**
 * @param {string} message
 * @param {{ enableThinking?: boolean, resubmit?: boolean, files?: Array<{ source: string, relativePath: string }>, onToken?: (chunk: string) => void, onReplace?: (text: string) => void }} [options]
 * @returns {Promise<string>}
 */
async function sendPrompt(message, options = {}) {
  if (!activeModelId || !activeEngineId) {
    throw new Error('Engine not initialized.');
  }
  if (generationInFlight) {
    throw new Error('Engine is busy.');
  }

  const enableThinking = options.enableThinking === true;
  const resubmit = options.resubmit !== false;
  const { message: enrichedMessage, imagePaths, audioPaths, videoPaths } =
    await prepareInferenceRequest(message, options.files);

  cancelRequested = false;
  contextManager.appendUser(enrichedMessage, { imagePaths, audioPaths, videoPaths });

  generationInFlight = true;
  phase = 'generating';
  let accumulated = '';
  try {
    const messages = stripThinkingFromMessages(contextManager.snapshot());
    const response = await engines[activeEngineId].runChat(activeModelId, enableThinking, enrichedMessage, {
      onToken: (chunk) => {
        if (typeof chunk === 'string' && chunk) {
          accumulated += chunk;
        }
        if (typeof options.onToken === 'function') {
          options.onToken(chunk);
        }
      },
      onReplace: (text) => {
        accumulated = typeof text === 'string' ? text : '';
        if (typeof options.onReplace === 'function') {
          options.onReplace(text);
        }
      },
      imagePaths,
      audioPaths,
      videoPaths,
      resubmit,
      messages,
    });
    if (cancelRequested) {
      const partial =
        typeof response === 'string' && response.length > 0 ? response : accumulated;
      const stopped = withStopMarker(partial);
      contextManager.appendAssistant(stopped);
      return stopped;
    }
    contextManager.appendAssistant(typeof response === 'string' ? response : '');
    return typeof response === 'string' ? response : '';
  } catch (err) {
    if (cancelRequested) {
      const stopped = withStopMarker(accumulated);
      contextManager.appendAssistant(stopped);
      return stopped;
    }
    contextManager.rollbackLastUser();
    throw err;
  } finally {
    generationInFlight = false;
    phase = 'idle';
  }
}

async function chatbotSupportsThinking() {
  if (!activeModelId || !activeEngineId) {
    return false;
  }
  return engines[activeEngineId].chatbotSupportsThinking();
}

async function chatbotHasChatTemplate() {
  if (!activeModelId || !activeEngineId) {
    return false;
  }
  return engines[activeEngineId].chatbotHasChatTemplate();
}

async function chatGenerationStartsInThinking(message, options = {}) {
  if (!activeModelId || !activeEngineId) {
    return false;
  }
  const enableThinking = options.enableThinking === true;
  const resubmit = options.resubmit !== false;
  const { message: enrichedMessage, imagePaths, audioPaths, videoPaths } =
    await prepareInferenceRequest(message, options.files);
  return engines[activeEngineId].chatGenerationStartsInThinking(
    activeModelId,
    enableThinking,
    enrichedMessage,
    {
      imagePaths,
      audioPaths,
      videoPaths,
      resubmit,
      messages: stripThinkingFromMessages(contextManager.snapshot()),
    }
  );
}

function assertNotGenerating(action) {
  if (generationInFlight) {
    throw new Error(`Wait until generation finishes before ${action}.`);
  }
}

async function contextSnapshot() {
  return contextManager.snapshot();
}

async function contextReplace(messages) {
  assertNotGenerating('loading a session');
  contextManager.replace(messages);
  // Keep engine working buffers / usage caches in sync for the gauge (not the source of truth).
  if (activeEngineId) {
    try {
      await engines[activeEngineId].contextReplace(
        stripThinkingFromMessages(contextManager.snapshot())
      );
    } catch {
      /* engine may not be ready */
    }
  }
}

async function contextClear() {
  assertNotGenerating('starting a new session');
  contextManager.clear();
  if (activeEngineId) {
    try {
      await engines[activeEngineId].contextClear();
    } catch {
      /* engine may not be ready */
    }
  }
}

/**
 * @param {boolean} [resubmit=true]
 * @param {{ refresh?: boolean }} [options]
 * @returns {Promise<{ used: number, total: number | null, stale?: boolean }>}
 */
async function contextUsage(resubmit = true, options = {}) {
  const refresh = Boolean(options && options.refresh);
  if (!activeEngineId) {
    return { used: 0, total: null, supported: false };
  }
  return engines[activeEngineId].contextUsage(
    resubmit,
    refresh,
    stripThinkingFromMessages(contextManager.snapshot())
  );
}

module.exports = {
  configurePaths,
  initialize,
  downloadModel,
  cancelModelDownload,
  listHubModelFiles,
  reinitialize,
  ejectModel,
  shutdown,
  resetInferenceWorker,
  sendPrompt,
  cancelGeneration,
  getStatus,
  chatbotSupportsThinking,
  chatbotHasChatTemplate,
  chatGenerationStartsInThinking,
  contextSnapshot,
  contextReplace,
  contextClear,
  contextUsage,
};
