/**
 * Inference facade. The rest of the app talks to engineManager only.
 * Routes by weight format and pipeline_tag:
 *   safetensors/pytorch → huggingface (including diffusers text- and image-to-image)
 *   GGUF + ASR          → transcribecpp
 *   GGUF + text-to-image or image-to-image → stablediffusioncpp
 *   GGUF (chat/other)   → llamacpp
 * Hub downloads always use the huggingface downloader stack.
 */

const fs = require('fs').promises;
const path = require('path');
const hfEngine = require('./huggingface/engine');
const llamaEngine = require('./llamacpp/engine');
const transcribeEngine = require('./transcribecpp/engine');
const stableDiffusionEngine = require('./stablediffusioncpp/engine');
const contextManager = require('./contextManager');
const { COMPANION_ALLOW_PATTERNS, detectModelFormat } = require('./common/modelFormat');
const { baseCompanionsReady, missingBaseCompanions } = require('./stablediffusioncpp/weights');
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
const {
  ASR_PIPELINE_TAG,
  IMAGE_TO_IMAGE_PIPELINE_TAG,
  isDiffusionPipelineTag,
  resolveEngineId,
} = require('./common/resolveEngineId');
const {
  assertGgufTextToImagePrompt,
  assertImageToImagePrompt,
  assertTextToImagePrompt,
} = require('./common/textToImage');

/** @type {{ resourcesRoot?: string, outputsRoot?: string, sessionsRoot?: string, modelsCacheDir?: string, modelId?: string, onProgress?: (info: object) => void } | null} */
let initOptions = null;

/** @type {string | null} */
let activeModelId = null;

/** @type {string | null} */
let activePipelineTag = null;

/** @type {'huggingface' | 'llamacpp' | 'transcribecpp' | 'stablediffusioncpp' | null} */
let activeEngineId = null;

/** @type {'idle' | 'loading' | 'generating' | 'downloading'} */
let phase = 'idle';

let generationInFlight = false;

/** Set by cancelGeneration so sendPrompt can keep the partial assistant turn. */
let cancelRequested = false;

/**
 * GGUF id whose base-repo companion download is in flight. Cancel uses this
 * id because the snapshot reports progress under the model the user is running.
 * @type {string | null}
 */
let baseCompanionDownloadId = null;
/**
 * VAE and text-encoder fetches keyed by GGUF id. Switching models leaves these
 * running; only an explicit stop cancels one.
 * @type {Map<string, { promise: Promise<void>, cancelled: boolean }>}
 */
const companionJobs = new Map();

const engines = {
  huggingface: hfEngine,
  llamacpp: llamaEngine,
  transcribecpp: transcribeEngine,
  stablediffusioncpp: stableDiffusionEngine,
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
  await stableDiffusionEngine.configure(mediaOpts);
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

/** @type {((modelId: string, info: object) => void) | null} */
let modelDownloadProgressListener = null;

/**
 * Models-list progress for a transfer that is not the panel's Add-model download
 * (a VAE and text encoder fetched while running a diffusion GGUF).
 *
 * @param {(modelId: string, info: object) => void} listener
 */
function setModelDownloadProgressListener(listener) {
  modelDownloadProgressListener = typeof listener === 'function' ? listener : null;
}

/**
 * @param {string} modelId
 * @param {object} info
 */
function emitModelDownloadProgress(modelId, info) {
  if (typeof modelDownloadProgressListener !== 'function') {
    return;
  }
  try {
    modelDownloadProgressListener(modelId, info);
  } catch {
    // The models list observes the transfer; it must not fail or cancel it.
  }
}

/**
 * @param {unknown} err
 * @returns {boolean}
 */
function isDownloadCancelledError(err) {
  return Boolean(
    err &&
      (/** @type {{ code?: string, message?: string }} */ (err).code === 'DOWNLOAD_CANCELLED' ||
        /cancelled/i.test(String(/** @type {{ message?: string }} */ (err).message || '')))
  );
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
 * @returns {Promise<'huggingface' | 'llamacpp' | 'transcribecpp' | 'stablediffusioncpp'>}
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
 * Workspace identity for each attachment, keyed later by absolute path.
 * Documents keep their original relative path (a PDF stays a PDF, not the
 * extracted markdown path written into the prompt).
 *
 * @param {Array<{ source: string, relativePath: string }>} files
 * @returns {Array<{ path: string, source: string, relativePath: string, kind: string }>}
 */
function describeAttachments(files) {
  const described = [];
  for (const file of files || []) {
    if (!file || typeof file.relativePath !== 'string' || !file.relativePath) {
      continue;
    }
    const abs = resolveInferenceFilePath(file);
    const mediaKind = mediaKindFromPath(abs);
    const kind = mediaKind || (isDocumentPath(abs) ? 'document' : null);
    if (!kind || typeof file.source !== 'string' || !file.source) {
      continue;
    }
    described.push({
      path: abs,
      source: file.source,
      relativePath: file.relativePath,
      kind,
    });
  }
  return described;
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
 * @returns {Promise<{ message: string, imagePaths: string[], audioPaths: string[], videoPaths: string[], attachmentFiles: Array<{ path: string, source: string, relativePath: string, kind: string }> }>}
 */
async function prepareInferenceRequest(message, files) {
  const attachmentFiles = describeAttachments(files);
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
  for (let i = 0; i < asrWavPaths.length; i += 1) {
    const video = attachmentFiles.find((item) => item.path === videoPaths[i] && item.kind === 'video');
    if (!video) {
      continue;
    }
    attachmentFiles.push({
      path: asrWavPaths[i],
      source: video.source,
      relativePath: video.relativePath,
      kind: 'audio',
    });
  }
  const finalAudioPaths = [...audioPaths, ...asrWavPaths];
  const finalVideoPaths = activePipelineTag === ASR_PIPELINE_TAG ? [] : videoPaths;
  return {
    message: finalMessage,
    imagePaths,
    audioPaths: finalAudioPaths,
    videoPaths: finalVideoPaths,
    attachmentFiles,
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
    sessionsRoot: opts.sessionsRoot,
    modelsCacheDir: opts.modelsCacheDir,
    onProgress: opts.onProgress,
  };
  await ensureEnginePaths(opts);
}

/**
 * Drop a generated file when it lives inside the sessions directory.
 * Outputs copies are never passed here.
 *
 * @param {string | undefined} outputPath
 */
async function discardGeneratedOutput(outputPath) {
  const sessionsRoot = initOptions && initOptions.sessionsRoot;
  if (typeof outputPath !== 'string' || !outputPath || !sessionsRoot) {
    return;
  }
  const root = path.resolve(sessionsRoot);
  const file = path.resolve(outputPath);
  const rel = path.relative(root, file);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) {
    return;
  }
  try {
    const stats = await fs.stat(file);
    if (stats.isFile()) {
      await fs.unlink(file);
    }
  } catch {
    /* already gone */
  }
}

/**
 * @param {string[]} imagePaths
 * @param {string} text
 */
function imageResultFromPaths(imagePaths, text) {
  const sessionsRoot = initOptions && initOptions.sessionsRoot;
  const imageParts = [];
  const images = [];
  for (const raw of imagePaths) {
    if (typeof raw !== 'string' || !raw) {
      continue;
    }
    const abs = path.resolve(raw);
    let relativePath = path.basename(abs);
    if (sessionsRoot) {
      const rel = path.relative(path.resolve(sessionsRoot), abs);
      if (rel && !rel.startsWith('..') && !path.isAbsolute(rel)) {
        relativePath = rel.split(path.sep).join('/');
      }
    }
    imageParts.push({
      type: 'image',
      path: abs,
      relativePath,
      source: 'sessions',
    });
    images.push({
      source: 'sessions',
      relativePath,
      name: path.basename(relativePath),
    });
  }
  return {
    text: typeof text === 'string' ? text : '',
    imagePaths: imageParts.map((part) => part.path),
    images,
    imageParts,
  };
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
  const modelsCacheDir = initOptions && initOptions.modelsCacheDir;
  if (engineId === 'stablediffusioncpp' && modelsCacheDir) {
    const modelRoot = path.join(modelsCacheDir, ...modelId.split('/'));
    if (missingBaseCompanions(modelRoot, modelsCacheDir)) {
      activeModelId = null;
      activePipelineTag = null;
      activeEngineId = null;
      phase = 'idle';
      startCompanionRestore(modelId);
      return { downloadRequired: true };
    }
  }
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
    if (activeEngineId === 'stablediffusioncpp') {
      startCompanionRestore(activeModelId);
    }
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
  return { downloadRequired: false };
}

/**
 * A GGUF download passes a variant key or an allow-list that names a .gguf file.
 * @param {object} opts
 * @returns {boolean}
 */
function isGgufDownload(opts) {
  if (opts && typeof opts.ggufVariant === 'string' && opts.ggufVariant.trim()) {
    return true;
  }
  return (
    Array.isArray(opts && opts.allowPatterns) &&
    opts.allowPatterns.some((pattern) => typeof pattern === 'string' && /\.gguf$/i.test(pattern))
  );
}

/**
 * @param {string} baseModelId
 * @param {string} progressModelId
 * @param {((info: object) => void) | undefined} report
 */
async function downloadBaseCompanions(baseModelId, progressModelId, report) {
  process.stderr.write(`[glaux] Fetching VAE and text encoder into ${baseModelId}.\n`);
  baseCompanionDownloadId = progressModelId;
  try {
    await hfEngine.downloadModel(baseModelId, {
      allowPatterns: COMPANION_ALLOW_PATTERNS,
      progressId: progressModelId,
      onProgress: typeof report === 'function' ? (event) => report(event) : undefined,
    });
  } finally {
    if (baseCompanionDownloadId === progressModelId) {
      baseCompanionDownloadId = null;
    }
  }
}

/**
 * Text-to-image GGUF downloads need the base repo's VAE and text encoder in
 * that repo's cache folder before the GGUF itself is fetched.
 * @param {string} modelId
 * @param {string} modelsCacheDir
 * @param {(info: object) => void} report
 */
async function ensureT2iBaseCompanions(modelId, modelsCacheDir, report) {
  const card = await hfEngine.readHubModelCard(modelId);
  const pipelineTag = card && typeof card.pipeline_tag === 'string' ? card.pipeline_tag : '';
  const baseModelId = card && typeof card.base_model === 'string' ? card.base_model : '';
  if (!isDiffusionPipelineTag(pipelineTag) || !baseModelId) {
    return;
  }
  const baseRoot = path.join(modelsCacheDir, ...baseModelId.split('/'));
  if (baseCompanionsReady(baseRoot)) {
    return;
  }
  await downloadBaseCompanions(baseModelId, modelId, report);
}

/**
 * Stop one GGUF's VAE and text-encoder fetch and clear its models-list row.
 * Does not remove the GGUF that was already downloaded, and does not affect
 * another model's fetch.
 * @param {string} modelId
 * @returns {Promise<void>}
 */
function abandonCompanionRestore(modelId) {
  const id = typeof modelId === 'string' ? modelId.trim() : '';
  const job = id ? companionJobs.get(id) : null;
  if (!job) {
    return Promise.resolve();
  }
  job.cancelled = true;
  hfEngine.cancelDownloadModel(id).catch(() => {});
  emitModelDownloadProgress(id, {
    phase: 'download',
    status: 'cancelled',
    modelId: id,
  });
  return job.promise.catch(() => {});
}

/**
 * Check the selected diffusion GGUF's base repo and download a missing VAE and
 * text encoder. The model load does not wait for the transfer. A fetch already
 * running for this GGUF, or for another one, is left running.
 * @param {string} modelId
 */
function startCompanionRestore(modelId) {
  const existing = companionJobs.get(modelId);
  if (existing) {
    return existing.promise;
  }
  /** @type {{ promise: Promise<void>, cancelled: boolean }} */
  const job = { promise: Promise.resolve(), cancelled: false };
  const tracked = restoreMissingBaseCompanions(modelId, job).finally(() => {
    if (companionJobs.get(modelId) === job) {
      companionJobs.delete(modelId);
    }
  });
  job.promise = tracked;
  companionJobs.set(modelId, job);
  tracked.catch(() => {});
  return tracked;
}

/**
 * A diffusion GGUF reads its VAE and text encoder from the base repo folder.
 * Fetch them when that folder does not have both.
 * @param {string} modelId
 * @param {{ cancelled: boolean }} job
 */
async function restoreMissingBaseCompanions(modelId, job) {
  const modelsCacheDir = initOptions && initOptions.modelsCacheDir;
  if (!modelsCacheDir || !modelId || job.cancelled) {
    return;
  }
  const modelRoot = path.join(modelsCacheDir, ...String(modelId).split('/'));
  const baseModelId = missingBaseCompanions(modelRoot, modelsCacheDir);
  if (!baseModelId || job.cancelled) {
    return;
  }
  await ensureDownloadEnginePaths({ modelsCacheDir });
  if (job.cancelled) {
    return;
  }
  const report = (event) => {
    if (job.cancelled) {
      return;
    }
    emitModelDownloadProgress(modelId, event && typeof event === 'object' ? event : {});
  };
  report({ phase: 'download', status: 'starting', modelId });
  try {
    await downloadBaseCompanions(baseModelId, modelId, report);
  } catch (err) {
    const cancelled = job.cancelled || isDownloadCancelledError(err);
    report({
      phase: 'download',
      status: cancelled ? 'cancelled' : 'error',
      modelId,
      message: err && err.message ? String(err.message) : '',
    });
    throw err;
  }
  if (job.cancelled) {
    report({ phase: 'download', status: 'cancelled', modelId });
    return;
  }
  report({ phase: 'download', status: 'complete', modelId });
}

async function downloadModel(opts) {
  if (!opts || typeof opts.modelId !== 'string' || !opts.modelId.trim()) {
    throw new Error('Model id is required.');
  }
  const modelId = opts.modelId.trim();
  const modelsCacheDir = await ensureDownloadEnginePaths(opts);
  const report = (info) => {
    try {
      if (typeof opts.onProgress === 'function') {
        opts.onProgress(info);
      } else {
        emitProgress(info);
      }
    } catch {
      // Listeners observe the transfer; they must not fail or cancel it.
    }
  };
  report({ phase: 'download', status: 'starting', modelId });
  if (isGgufDownload(opts)) {
    await ensureT2iBaseCompanions(modelId, modelsCacheDir, report);
  }
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
  if (activeModelId) {
    await abandonCompanionRestore(activeModelId);
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

async function unloadHuggingFacePipeline() {
  try {
    await engines.huggingface.chatbotDestroyIfRunning();
  } catch {
    /* worker may already be gone */
  }
}

async function shutdown() {
  for (const engine of Object.values(engines)) {
    try {
      await engine.chatStop();
    } catch {
      /* ignore */
    }
  }
  // chatbot_destroy removes the disk-offload folder. close() kills the worker
  // and would leave those files behind.
  await unloadHuggingFacePipeline();
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
  await unloadHuggingFacePipeline();
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
  if (activeModelId) {
    abandonCompanionRestore(activeModelId);
  }
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

  if (isDiffusionPipelineTag(activePipelineTag)) {
    const imageToImage = activePipelineTag === IMAGE_TO_IMAGE_PIPELINE_TAG;
    const checked = imageToImage
      ? assertImageToImagePrompt(message, options.files)
      : activeEngineId === 'stablediffusioncpp'
        ? assertGgufTextToImagePrompt(message, options.files)
        : { prompt: assertTextToImagePrompt(message, options.files), imageFile: null };
    const prompt = checked.prompt;
    const imagePaths = checked.imageFile
      ? mediaPathsFromFiles([checked.imageFile]).imagePaths
      : [];
    if (typeof options.outputPath !== 'string' || !options.outputPath.trim()) {
      throw new Error('Image generation output path is not configured.');
    }
    const outputPath = options.outputPath;
    cancelRequested = false;
    contextManager.appendUser(prompt, {
      imagePaths,
      files: checked.imageFile ? describeAttachments([checked.imageFile]) : [],
    });
    generationInFlight = true;
    phase = 'generating';
    const pendingCompanions = companionJobs.get(activeModelId)?.promise;
    try {
      if (pendingCompanions) {
        await pendingCompanions;
      }
      if (cancelRequested) {
        await discardGeneratedOutput(outputPath);
        const stopped = withStopMarker('');
        contextManager.appendAssistant(stopped);
        return stopped;
      }
      const response = await engines[activeEngineId].runChat(activeModelId, false, prompt, {
        onToken: options.onToken,
        onReplace: options.onReplace,
        messages: [],
        outputPath,
        imagePaths,
        audioPaths: [],
        videoPaths: [],
      });
      if (cancelRequested) {
        await discardGeneratedOutput(outputPath);
        const stopped = withStopMarker('');
        contextManager.appendAssistant(stopped);
        return stopped;
      }
      const rawPaths =
        response && typeof response === 'object' && Array.isArray(response.imagePaths)
          ? response.imagePaths
          : [];
      const text =
        response && typeof response === 'object' && typeof response.text === 'string'
          ? response.text
          : typeof response === 'string'
            ? response
            : '';
      if (!rawPaths.length) {
        throw new Error('Image generation did not return an image.');
      }
      const packed = imageResultFromPaths(rawPaths, text);
      contextManager.appendAssistant(packed.text, { imageParts: packed.imageParts });
      return { text: packed.text, imagePaths: packed.imagePaths, images: packed.images };
    } catch (err) {
      if (cancelRequested) {
        await discardGeneratedOutput(outputPath);
        const stopped = withStopMarker('');
        contextManager.appendAssistant(stopped);
        return stopped;
      }
      await discardGeneratedOutput(outputPath);
      contextManager.rollbackLastUser();
      throw err;
    } finally {
      generationInFlight = false;
      phase = 'idle';
    }
  }

  const enableThinking = options.enableThinking === true;
  const resubmit = options.resubmit !== false;
  const { message: enrichedMessage, imagePaths, audioPaths, videoPaths, attachmentFiles } =
    await prepareInferenceRequest(message, options.files);

  cancelRequested = false;
  contextManager.appendUser(enrichedMessage, {
    imagePaths,
    audioPaths,
    videoPaths,
    files: attachmentFiles,
  });

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
  if (isDiffusionPipelineTag(activePipelineTag)) {
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
  setModelDownloadProgressListener,
  cancelCompanionDownload: abandonCompanionRestore,
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
