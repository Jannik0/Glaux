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
const { isFileInside } = require('./common/fileInside');
const { releaseSessionImagePath } = require('./common/imagePathReservation');

/** How long shutdown waits for the Hugging Face pipeline to drop its offload files. */
const HF_DESTROY_TIMEOUT_MS = 5000;

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

/**
 * Identifies the sendPrompt call that owns generationInFlight.
 * cancelGeneration clears the flag immediately; a newer run must not have it
 * reset by the older run's finally. Each run keeps its own cancelled flag.
 * @type {{ token: number, cancelled: boolean, requestId: string, senderId: unknown } | null}
 */
let activeGeneration = null;
let generationRun = 0;

/**
 * How long a Diffusers stop waits for the next denoising step before the
 * worker is killed. Tests replace it with {@link setDiffusersStopGraceMs}.
 */
const DIFFUSERS_STOP_GRACE_MS = 5000;
let diffusersStopGraceMs = DIFFUSERS_STOP_GRACE_MS;
/**
 * How long to wait for the killed worker to exit before deleting its offload
 * folder. Tests replace it with {@link setDiffusersWorkerExitMs}.
 */
const DIFFUSERS_WORKER_EXIT_MS = 5000;
let diffusersWorkerExitMs = DIFFUSERS_WORKER_EXIT_MS;
/** True from the cooperative stop until that run settles or the reload begins. */
let diffusersStopPending = false;
/** @type {{ stopGraceTimer?: ReturnType<typeof setTimeout> | null, killStarted?: boolean, settled?: boolean, armedEngineId?: string | null, armedModelId?: string | null } | null} */
let diffusersStopRun = null;
/** @type {Promise<void> | null} */
let diffusersReloadPromise = null;
let diffusersReloadToken = 0;
/** One in-flight model reload, shared by the stop kill and the stream idle timeout. */
/** @type {Promise<void> | null} */
let modelReloadJob = null;
let shuttingDown = false;
/** @type {(() => Promise<unknown> | unknown) | null} */
let diffusersKillRecovery = null;
/**
 * Offload folder parents. Null uses the production defaults (models cache,
 * the model directory's parent, and /var/tmp except on Windows).
 * @type {string[] | null}
 */
let diffusersOffloadBases = null;
/**
 * Downloads whose worker RPC was rejected by the stop kill. Resumed after a
 * successful reload; rejected if the user cancels, the model is deleted, or
 * the reload fails.
 * @type {Array<{ modelId: string, progressId: string, hfOptions: object, existed: boolean, cancelled: boolean, error: Error, resolve: (value?: unknown) => void, reject: (err: Error) => void }>}
 */
let parkedDownloads = [];
/** Resumes already taken out of `parkedDownloads` and waiting on the worker. */
let activeResumes = [];
let resumingDownloads = false;

/**
 * request id → run token. Settled ids are pruned down to the newest
 * MAX_TRACKED_REQUEST_RUNS so the map cannot grow without bound. A missing id
 * is unknown and must not cancel whatever run is active now.
 * @type {Map<string, number>}
 */
const requestRunTokens = new Map();
const MAX_TRACKED_REQUEST_RUNS = 64;

/**
 * @param {string} requestId
 * @param {number} token
 */
function rememberRequestRun(requestId, token) {
  if (!requestId) {
    return;
  }
  if (requestRunTokens.has(requestId)) {
    requestRunTokens.delete(requestId);
  }
  requestRunTokens.set(requestId, token);
  while (requestRunTokens.size > MAX_TRACKED_REQUEST_RUNS) {
    const oldest = requestRunTokens.keys().next().value;
    if (
      activeGeneration &&
      activeGeneration.requestId &&
      oldest === activeGeneration.requestId
    ) {
      const activeToken = requestRunTokens.get(oldest);
      requestRunTokens.delete(oldest);
      requestRunTokens.set(oldest, activeToken);
      const nextOldest = requestRunTokens.keys().next().value;
      if (!nextOldest || nextOldest === activeGeneration.requestId) {
        break;
      }
      requestRunTokens.delete(nextOldest);
      continue;
    }
    requestRunTokens.delete(oldest);
  }
}

/**
 * @param {{ requestId?: string, senderId?: unknown }} [meta]
 * @returns {{ token: number, cancelled: boolean, requestId: string, senderId: unknown, userId: number | null, partialText: string, stopSealed: boolean, stoppedText: string }}
 */
function startGenerationRun(meta = {}) {
  const requestId = meta && meta.requestId ? String(meta.requestId) : '';
  const senderId = meta && meta.senderId != null ? meta.senderId : undefined;
  const run = {
    token: ++generationRun,
    cancelled: false,
    requestId,
    senderId,
    userId: null,
    partialText: '',
    stopSealed: false,
    stoppedText: '',
    settled: false,
  };
  activeGeneration = run;
  generationInFlight = true;
  phase = 'generating';
  rememberRequestRun(requestId, run.token);
  return run;
}

/**
 * @param {{ requestId?: string, senderId?: unknown }} [options]
 * @returns {{ requestId: string, senderId: unknown }}
 */
function generationMeta(options) {
  const requestId = options && options.requestId ? String(options.requestId) : '';
  const senderId = options && options.senderId != null ? options.senderId : undefined;
  return { requestId, senderId };
}

/**
 * @param {{ token: number, cancelled: boolean } | null | undefined} run
 * @returns {boolean}
 */
function generationOwns(run) {
  return Boolean(run) && run.token === generationRun;
}

/**
 * @param {{ stopGraceTimer?: ReturnType<typeof setTimeout> | null } | null | undefined} run
 */
function clearStopGrace(run) {
  if (run && run.stopGraceTimer) {
    clearTimeout(run.stopGraceTimer);
    run.stopGraceTimer = null;
  }
}

/**
 * The owning run has returned. A later cancel must not seal this turn.
 * @param {{ token: number, settled?: boolean, killStarted?: boolean, stopGraceTimer?: ReturnType<typeof setTimeout> | null } | null | undefined} run
 */
function settleOwnedRun(run) {
  if (run) {
    clearStopGrace(run);
  }
  if (!run || !generationOwns(run)) {
    return;
  }
  run.settled = true;
  generationInFlight = false;
  if (activeGeneration === run) {
    activeGeneration = null;
  }
  if (diffusersStopRun === run && !run.killStarted) {
    diffusersStopPending = false;
    diffusersStopRun = null;
    phase = 'idle';
    return;
  }
  if (diffusersStopPending || diffusersReloadPromise) {
    phase = 'loading';
    return;
  }
  phase = 'idle';
}

function diffusersStopProcedureActive() {
  return Boolean(diffusersStopPending || diffusersReloadPromise);
}

/**
 * Overlapping generation uses this same unlocalized message. The IPC layer
 * classifies any message containing "busy" as E_BUSY. There is no locale key.
 * @returns {Error}
 */
function engineBusyError() {
  return new Error('Engine is busy.');
}

function assertDiffusersStopIdle() {
  if (diffusersStopProcedureActive()) {
    throw engineBusyError();
  }
}

/**
 * @param {ReturnType<typeof setTimeout>} timer
 */
function unrefTimer(timer) {
  if (timer && typeof timer.unref === 'function') {
    timer.unref();
  }
  return timer;
}

/**
 * @param {Promise<void>} promise
 * @param {number} ms
 */
function raceUnrefTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((resolve) => {
    timer = unrefTimer(setTimeout(resolve, ms));
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

function isHuggingFaceDiffusion() {
  return activeEngineId === 'huggingface' && isDiffusionPipelineTag(activePipelineTag);
}

function huggingfaceWorkerPid() {
  const engine = engines.huggingface;
  if (!engine || typeof engine.workerPid !== 'function') {
    return null;
  }
  const pid = engine.workerPid();
  return Number.isInteger(pid) && pid > 0 ? pid : null;
}

/**
 * Relative offload folder for one worker pid. Anything else is ignored.
 * @param {number | null | undefined} pid
 * @returns {string | null}
 */
function offloadRelativeDir(pid) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return null;
  }
  const rel = path.join('.glaux-t2i-offload', `pid-${pid}`);
  if (!/^\.glaux-t2i-offload[\\/]pid-[1-9]\d*$/.test(rel)) {
    return null;
  }
  return rel;
}

/**
 * Parents of `.glaux-t2i-offload/pid-<pid>`. Production uses the models cache,
 * the armed model directory's parent, and `/var/tmp` on Linux and macOS.
 * Windows skips `/var/tmp`. Tests inject temp directories instead.
 *
 * @param {string | null | undefined} modelId Model id recorded when the grace was armed
 * @param {NodeJS.Platform} [platform]
 * @returns {string[]}
 */
function resolveOffloadBaseDirs(modelId, platform = process.platform) {
  if (diffusersOffloadBases) {
    return [...diffusersOffloadBases];
  }
  const bases = [];
  const cache = initOptions && initOptions.modelsCacheDir;
  if (typeof cache === 'string' && cache) {
    bases.push(path.resolve(cache));
    if (modelId) {
      const modelDir = path.join(cache, ...String(modelId).split('/'));
      bases.push(path.resolve(path.dirname(modelDir)));
    }
  }
  if (platform !== 'win32') {
    bases.push(path.resolve('/var/tmp'));
  }
  return bases;
}

/**
 * Absolute offload directories for a killed worker. Empty when pid is not a
 * positive integer.
 *
 * @param {number | null | undefined} pid
 * @param {string | null | undefined} modelId
 * @param {NodeJS.Platform} [platform]
 * @returns {string[]}
 */
function killedOffloadPaths(pid, modelId, platform = process.platform) {
  const rel = offloadRelativeDir(pid);
  if (!rel) {
    return [];
  }
  const targets = [];
  for (const base of resolveOffloadBaseDirs(modelId, platform)) {
    const root = path.resolve(base);
    const target = path.resolve(root, rel);
    const prefix = root.endsWith(path.sep) ? root : root + path.sep;
    if (!target.startsWith(prefix)) {
      continue;
    }
    targets.push(target);
  }
  return targets;
}

/**
 * @param {string} dir
 */
async function removeOffloadDir(dir) {
  try {
    await fs.rm(dir, { recursive: true, force: true });
  } catch {
    await fs.rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}

/**
 * The next Diffusers load deletes a dead pid folder. Remove this one now so
 * the killed worker's offload files do not wait for that prepare.
 * `modelId` is the id captured when the grace was armed.
 *
 * @param {number | null} pid
 * @param {string | null | undefined} modelId
 */
async function removeKilledDiffusersOffload(pid, modelId) {
  const dirs = killedOffloadPaths(pid, modelId);
  await Promise.all(dirs.map((dir) => removeOffloadDir(dir)));
}

function armDiffusersStopGrace(run) {
  if (!isHuggingFaceDiffusion() || !run || run.stopGraceTimer || run.killStarted) {
    return;
  }
  diffusersStopPending = true;
  diffusersStopRun = run;
  run.armedEngineId = activeEngineId;
  run.armedModelId = activeModelId;
  phase = 'loading';
  run.stopGraceTimer = unrefTimer(
    setTimeout(() => {
      run.stopGraceTimer = null;
      void killHungDiffusersRun(run);
    }, diffusersStopGraceMs)
  );
}

/**
 * @param {{ settled?: boolean, killStarted?: boolean, stopGraceTimer?: ReturnType<typeof setTimeout> | null }} run
 */
function endDiffusersStopProcedure() {
  diffusersStopPending = false;
  diffusersStopRun = null;
  if (!generationInFlight && !diffusersReloadPromise && phase === 'loading') {
    phase = 'idle';
  }
}

async function killHungDiffusersRun(run) {
  if (!run || run.settled || run.killStarted || shuttingDown) {
    return;
  }
  if (diffusersStopRun !== run) {
    return;
  }
  if (run.armedEngineId !== activeEngineId || run.armedModelId !== activeModelId) {
    endDiffusersStopProcedure();
    return;
  }
  run.killStarted = true;
  clearStopGrace(run);
  const pid = huggingfaceWorkerPid();
  const armedModelId = run.armedModelId;
  let closing = Promise.resolve();
  try {
    const result = engines.huggingface.close({ final: false });
    if (result && typeof result.then === 'function') {
      closing = result;
    }
  } catch {
    /* worker already gone */
  }
  // The stream settles from rejectAll inside close(). Cleanup waits for the
  // process to exit, bounded, so a Windows handle is not still open.
  await raceUnrefTimeout(
    Promise.resolve(closing).catch(() => {}),
    diffusersWorkerExitMs
  );
  await removeKilledDiffusersOffload(pid, armedModelId);
  const deadline = Date.now() + 1000;
  while (!run.settled && Date.now() < deadline) {
    await new Promise((resolve) => {
      setImmediate(resolve);
    });
  }
  if (shuttingDown) {
    endDiffusersStopProcedure();
    await resumeParkedDownloads(false);
    return;
  }
  startDiffusersReload();
}

/**
 * Run `fn` unless a model reload is already in flight, in which case wait for
 * that one. The stop-kill recovery and the stream idle-timeout recovery both
 * use this so two triggers produce one reload.
 *
 * @param {() => Promise<unknown> | unknown} fn
 * @returns {Promise<unknown>}
 */
function joinModelReload(fn) {
  if (shuttingDown) {
    return Promise.resolve();
  }
  if (modelReloadJob) {
    return modelReloadJob;
  }
  const job = Promise.resolve()
    .then(() => fn())
    .finally(() => {
      if (modelReloadJob === job) {
        modelReloadJob = null;
      }
    });
  modelReloadJob = job;
  return job;
}

function startDiffusersReload() {
  if (shuttingDown) {
    endDiffusersStopProcedure();
    resumeParkedDownloadsInBackground(false);
    return;
  }
  const token = ++diffusersReloadToken;
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  diffusersReloadPromise = gate;
  diffusersStopPending = false;
  diffusersStopRun = null;
  phase = 'loading';
  const job = (async () => {
    let reloadOk = false;
    try {
      if (!shuttingDown && token === diffusersReloadToken && typeof diffusersKillRecovery === 'function') {
        const result = await diffusersKillRecovery();
        const failed = Boolean(result && /** @type {{ ok?: boolean }} */ (result).ok === false);
        reloadOk = !failed && !shuttingDown && Boolean(activeModelId);
      }
    } catch (err) {
      console.error('Failed to reload the model after stopping image generation:', err);
      reloadOk = false;
    } finally {
      // Drop the busy gate before resuming downloads. A multi-GB resume must
      // not keep phase at loading or reject model changes.
      release();
      if (token === diffusersReloadToken && diffusersReloadPromise === gate) {
        diffusersReloadPromise = null;
      }
      if (!generationInFlight && !diffusersStopPending && phase === 'loading') {
        phase = 'idle';
      }
      resumeParkedDownloadsInBackground(reloadOk);
    }
  })();
  job.catch(() => {});
}

/**
 * @param {boolean} reloadOk
 */
function resumeParkedDownloadsInBackground(reloadOk) {
  void resumeParkedDownloads(reloadOk).catch((err) => {
    console.error('Failed to resume downloads after stopping image generation:', err);
  });
}

/**
 * @param {number} ms
 */
function setDiffusersStopGraceMs(ms) {
  const next = Number(ms);
  diffusersStopGraceMs = Number.isFinite(next) && next >= 0 ? next : DIFFUSERS_STOP_GRACE_MS;
}

/**
 * @param {number} ms
 */
function setDiffusersWorkerExitMs(ms) {
  const next = Number(ms);
  diffusersWorkerExitMs = Number.isFinite(next) && next >= 0 ? next : DIFFUSERS_WORKER_EXIT_MS;
}

/**
 * @param {string[] | null} dirs Null restores the production defaults.
 */
function setDiffusersOffloadBaseDirs(dirs) {
  if (dirs == null) {
    diffusersOffloadBases = null;
    return;
  }
  diffusersOffloadBases = dirs.map((dir) => path.resolve(String(dir)));
}

/**
 * @param {(() => Promise<unknown> | unknown) | null} fn
 */
function setDiffusersKillRecovery(fn) {
  diffusersKillRecovery = typeof fn === 'function' ? fn : null;
}

function isShuttingDown() {
  return shuttingDown;
}

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
  let reportedPhase = phase;
  if (generationInFlight) {
    reportedPhase = 'generating';
  } else if (diffusersStopProcedureActive()) {
    reportedPhase = 'loading';
  }
  return {
    phase: reportedPhase,
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
 * @param {number | undefined | null} [reservationId]
 */
async function discardGeneratedOutput(outputPath, reservationId) {
  try {
    const sessionsRoot = initOptions && initOptions.sessionsRoot;
    if (typeof outputPath !== 'string' || !outputPath || !sessionsRoot) {
      return;
    }
    if (!isFileInside(sessionsRoot, outputPath)) {
      return;
    }
    const file = path.resolve(outputPath);
    try {
      const stats = await fs.stat(file);
      if (stats.isFile()) {
        await fs.unlink(file);
      }
    } catch {
      /* already gone */
    }
  } finally {
    releaseSessionImagePath(outputPath, reservationId);
  }
}

/**
 * Place the stop-marker assistant immediately after this run's own user turn.
 * A missing id is left alone so the marker cannot land on a newer turn.
 *
 * @param {number | null | undefined} userId
 * @param {string} text
 * @returns {string}
 */
function recordStoppedAssistant(userId, text) {
  const stopped = withStopMarker(text);
  contextManager.insertAssistantAfter(userId, stopped);
  return stopped;
}

/**
 * Insert this run's stop marker once, immediately after its own user message.
 * A later settle of the same run must not insert a second marker.
 *
 * @param {{ userId?: number | null, partialText?: string, stopSealed?: boolean, stoppedText?: string } | null | undefined} run
 * @param {string} [text]
 * @returns {string}
 */
function sealRun(run, text) {
  const body = typeof text === 'string' ? text : run && typeof run.partialText === 'string' ? run.partialText : '';
  if (!run || run.userId == null) {
    return withStopMarker(body);
  }
  if (run.stopSealed) {
    return run.stoppedText || withStopMarker(body);
  }
  run.stopSealed = true;
  run.stoppedText = recordStoppedAssistant(run.userId, body);
  return run.stoppedText;
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
  // Only the stop-recovery reload and enterNoModelState may set allowDuringDiffusersStop.
  const privileged = Boolean(opts && opts.allowDuringDiffusersStop);
  if (!privileged) {
    assertDiffusersStopIdle();
  }
  if (opts && Object.prototype.hasOwnProperty.call(opts, 'allowDuringDiffusersStop')) {
    opts = { ...opts };
    delete opts.allowDuringDiffusersStop;
  }
  shuttingDown = false;
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
  await runHfDownload(
    baseModelId,
    {
      allowPatterns: COMPANION_ALLOW_PATTERNS,
      progressId: progressModelId,
      onProgress: typeof report === 'function' ? (event) => report(event) : undefined,
    },
    progressModelId
  );
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
  if (id) {
    rejectParkedDownload(id);
  }
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

/**
 * @param {unknown} err
 * @returns {boolean}
 */
function isWorkerKillError(err) {
  const message = String(err && /** @type {{ message?: string }} */ (err).message ? err.message : err || '');
  return /Engine closed|Python process killed|Python process exited|Python process is not running|Python worker is shutting down/i.test(
    message
  );
}

/**
 * @param {string} modelId
 * @returns {Promise<boolean>}
 */
async function modelCacheDirExists(modelId) {
  const cache = initOptions && initOptions.modelsCacheDir;
  if (!cache || !modelId) {
    return false;
  }
  try {
    const stats = await fs.stat(path.join(cache, ...String(modelId).split('/')));
    return stats.isDirectory();
  } catch {
    return false;
  }
}

/**
 * @param {{ settled?: boolean, cancelled?: boolean, reject: (err: Error) => void }} entry
 * @param {Error} err
 */
function settleDownloadReject(entry, err) {
  if (entry.settled) {
    return;
  }
  entry.settled = true;
  entry.cancelled = err && err.code === 'DOWNLOAD_CANCELLED';
  entry.reject(err);
}

/**
 * @param {{ settled?: boolean, resolve: (value?: unknown) => void }} entry
 */
function settleDownloadResolve(entry) {
  if (entry.settled) {
    return;
  }
  entry.settled = true;
  entry.resolve();
}

/**
 * @param {string} id
 * @returns {false | 'parked' | 'resume'}
 */
function rejectParkedDownload(id) {
  const key = typeof id === 'string' ? id.trim() : '';
  if (!key) {
    return false;
  }
  const err = new Error('Download cancelled');
  err.code = 'DOWNLOAD_CANCELLED';
  let kind = false;
  parkedDownloads = parkedDownloads.filter((entry) => {
    if (entry.modelId !== key && entry.progressId !== key) {
      return true;
    }
    kind = 'parked';
    settleDownloadReject(entry, err);
    return false;
  });
  activeResumes = activeResumes.filter((entry) => {
    if (entry.modelId !== key && entry.progressId !== key) {
      return true;
    }
    kind = 'resume';
    settleDownloadReject(entry, err);
    return false;
  });
  return kind;
}

/**
 * @param {boolean} reloadOk
 */
async function resumeParkedDownloads(reloadOk) {
  const batch = parkedDownloads.splice(0, parkedDownloads.length);
  if (!batch.length) {
    return;
  }
  // While this is set, a kill that aborts the resumed RPC is not parked again.
  resumingDownloads = true;
  try {
    for (const entry of batch) {
      if (entry.settled || entry.cancelled) {
        continue;
      }
      if (!reloadOk || (entry.existed && !(await modelCacheDirExists(entry.modelId)))) {
        settleDownloadReject(entry, entry.error);
        continue;
      }
      activeResumes.push(entry);
      try {
        if (typeof entry.hfOptions.onProgress === 'function') {
          entry.hfOptions.onProgress({
            phase: 'download',
            status: 'starting',
            modelId: entry.progressId || entry.modelId,
          });
        }
        await hfEngine.downloadModel(entry.modelId, entry.hfOptions);
        settleDownloadResolve(entry);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        // A user cancel already rejected the caller with DOWNLOAD_CANCELLED.
        settleDownloadReject(entry, error);
      } finally {
        activeResumes = activeResumes.filter((item) => item !== entry);
      }
    }
  } finally {
    resumingDownloads = false;
  }
}

/**
 * Hugging Face download that stays pending when the stop kill rejects it, then
 * resumes on the existing path after the model reload.
 *
 * @param {string} modelId
 * @param {object} hfOptions
 * @param {string} progressId
 */
async function runHfDownload(modelId, hfOptions, progressId) {
  try {
    await hfEngine.downloadModel(modelId, hfOptions);
  } catch (err) {
    const error = err instanceof Error ? err : new Error(String(err));
    if (
      resumingDownloads ||
      shuttingDown ||
      !isWorkerKillError(error) ||
      !diffusersStopProcedureActive()
    ) {
      throw error;
    }
    const existed = await modelCacheDirExists(modelId);
    // stat() yields. Parking after the procedure has ended, or after shutdown,
    // would leave a download that nobody resumes.
    if (resumingDownloads || shuttingDown || !diffusersStopProcedureActive()) {
      throw error;
    }
    await new Promise((resolve, reject) => {
      parkedDownloads.push({
        modelId,
        progressId: progressId || modelId,
        hfOptions,
        existed,
        cancelled: false,
        error,
        resolve,
        reject,
      });
    });
  }
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
  await runHfDownload(
    modelId,
    {
      onProgress: (event) => report(event),
      allowPatterns: opts.allowPatterns,
      ggufVariant: opts.ggufVariant,
    },
    modelId
  );
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
  const id = modelId.trim();
  const hit = rejectParkedDownload(id);
  if (hit) {
    if (hit === 'resume') {
      hfEngine.cancelDownloadModel(id).catch(() => {});
    }
    return;
  }
  await ensureDownloadEnginePaths(opts);
  return hfEngine.cancelDownloadModel(id);
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
  // Only the stop-recovery reload and enterNoModelState may set allowDuringDiffusersStop.
  const privileged = Boolean(opts && opts.allowDuringDiffusersStop);
  if (!privileged) {
    assertDiffusersStopIdle();
  }
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

  const stored = { ...initOptions, ...next, modelId: nextId };
  delete stored.allowDuringDiffusersStop;
  initOptions = stored;
  await ensureEnginePaths(initOptions);
  return initialize(privileged ? { ...initOptions, allowDuringDiffusersStop: true } : initOptions);
}

/**
 * Unload the active chat model (keeps workers alive when possible).
 */
async function ejectModel(opts) {
  // Only the stop-recovery reload and enterNoModelState may set allowDuringDiffusersStop.
  if (!(opts && opts.allowDuringDiffusersStop)) {
    assertDiffusersStopIdle();
  }
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

/**
 * @param {number} [timeoutMs]
 */
async function unloadHuggingFacePipeline(timeoutMs = HF_DESTROY_TIMEOUT_MS) {
  const limit = Number.isFinite(timeoutMs) ? timeoutMs : HF_DESTROY_TIMEOUT_MS;
  let timer;
  let timedOut = false;
  const timeout = new Promise((resolve) => {
    timer = setTimeout(() => {
      timedOut = true;
      console.warn(
        `Hugging Face pipeline unload timed out after ${limit}ms; closing the worker.`
      );
      resolve();
    }, limit);
  });
  try {
    await Promise.race([engines.huggingface.chatbotDestroyIfRunning(), timeout]);
  } catch {
    /* worker may already be gone */
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
  return timedOut;
}

async function shutdown(options = {}) {
  shuttingDown = true;
  diffusersReloadToken += 1;
  diffusersReloadPromise = null;
  diffusersStopPending = false;
  clearStopGrace(diffusersStopRun);
  clearStopGrace(activeGeneration);
  diffusersStopRun = null;
  await resumeParkedDownloads(false);
  for (const engine of Object.values(engines)) {
    try {
      await engine.chatStop();
    } catch {
      /* ignore */
    }
  }
  // chatbot_destroy removes the disk-offload folder. close() kills the worker
  // and would leave those files behind. A hung destroy must not block close();
  // the next text-to-image load deletes a leftover offload folder.
  await unloadHuggingFacePipeline(options.hfDestroyTimeoutMs);
  await Promise.all(
    Object.values(engines).map((engine) => engine.close({ final: true }).catch(() => {}))
  );
  generationInFlight = false;
  phase = 'idle';
  activeModelId = null;
  activePipelineTag = null;
  activeEngineId = null;
}

async function resetInferenceWorker(options = {}) {
  for (const engine of Object.values(engines)) {
    try {
      await engine.chatStop();
    } catch {
      /* ignore */
    }
  }
  await unloadHuggingFacePipeline(options.hfDestroyTimeoutMs);
  await Promise.all(
    Object.values(engines).map((engine) => engine.close({ final: false }).catch(() => {}))
  );
  generationInFlight = false;
  if (phase === 'generating') {
    phase = 'idle';
  }
}

function cancelActiveGeneration() {
  const run = activeGeneration;
  // A finished run stays reachable until its finally clears it. Sealing then
  // splices [STOP] into a turn that already completed.
  if (!run || run.settled || !generationInFlight || run.token !== generationRun) {
    return;
  }
  run.cancelled = true;
  // The newer prompt snapshots context as soon as this returns. The stop
  // marker has to be in place before that snapshot, not when this run settles.
  sealRun(run, run.partialText || '');
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
  // The stop marker is already in place. Diffusers may not notice chat_stop
  // until the next denoising step, so kill the worker if this run is still
  // going when the 5 second grace expires. Chat, ASR, and sd-cli do not arm it.
  armDiffusersStopGrace(run);
}

function cancelGeneration() {
  cancelActiveGeneration();
}

/**
 * Cancel the run bound to this request id. An unknown id is a no-op: it must
 * not cancel the active run. Sender fallback is only for sender destruction,
 * and only when the active run was started by that same request.
 *
 * @param {{ requestId?: string, senderId?: unknown, senderFallback?: boolean }} [scope]
 */
function cancelRequestGeneration(scope = {}) {
  const requestId = scope && scope.requestId ? String(scope.requestId) : '';
  if (requestId && requestRunTokens.has(requestId)) {
    const token = requestRunTokens.get(requestId);
    if (activeGeneration && activeGeneration.token === token) {
      cancelActiveGeneration();
    }
    return;
  }
  if (!scope || scope.senderFallback !== true) {
    return;
  }
  const senderId = scope.senderId;
  if (
    activeGeneration &&
    requestId &&
    activeGeneration.requestId === requestId &&
    senderId != null &&
    activeGeneration.senderId != null &&
    activeGeneration.senderId === senderId
  ) {
    cancelActiveGeneration();
  }
}

function requestRunTokenCount() {
  return requestRunTokens.size;
}

/**
 * True when the active run has already inserted its stop marker.
 * A settled or idle engine has no active run, so this is false.
 * @returns {boolean}
 */
function activeRunStopSealed() {
  return Boolean(activeGeneration && activeGeneration.stopSealed);
}

function configuredSessionsRoot() {
  const root = initOptions && initOptions.sessionsRoot;
  return typeof root === 'string' ? root : '';
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
  if (generationInFlight || diffusersStopPending || diffusersReloadPromise) {
    throw engineBusyError();
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
    const reservationId = options.reservationId;
    const userId = contextManager.appendUser(prompt, {
      imagePaths,
      files: checked.imageFile ? describeAttachments([checked.imageFile]) : [],
    });
    const run = startGenerationRun(generationMeta(options));
    run.userId = userId;
    const pendingCompanions = companionJobs.get(activeModelId)?.promise;
    const stopThisRun = async () => {
      await discardGeneratedOutput(outputPath, reservationId);
      return sealRun(run, '');
    };
    try {
      if (pendingCompanions) {
        await pendingCompanions;
      }
      if (!generationOwns(run) || run.cancelled) {
        clearStopGrace(run);
        return await stopThisRun();
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
      if (!generationOwns(run) || run.cancelled) {
        clearStopGrace(run);
        return await stopThisRun();
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
      if (!generationOwns(run) || run.cancelled) {
        clearStopGrace(run);
        return await stopThisRun();
      }
      // Drop this turn and settle before the await. A Stop during the discard
      // must not seal a run whose user message is about to be removed.
      contextManager.removeMessageById(userId);
      settleOwnedRun(run);
      await discardGeneratedOutput(outputPath, reservationId);
      throw err;
    } finally {
      releaseSessionImagePath(outputPath, reservationId);
      settleOwnedRun(run);
    }
  }

  const enableThinking = options.enableThinking === true;
  const resubmit = options.resubmit !== false;
  const run = startGenerationRun(generationMeta(options));
  try {
    const prepared = await prepareInferenceRequest(message, options.files);
    if (!generationOwns(run)) {
      return withStopMarker('');
    }
    const { message: enrichedMessage, imagePaths, audioPaths, videoPaths, attachmentFiles } = prepared;
    const userId = contextManager.appendUser(enrichedMessage, {
      imagePaths,
      audioPaths,
      videoPaths,
      files: attachmentFiles,
    });
    run.userId = userId;
    if (run.cancelled) {
      return sealRun(run, run.partialText || '');
    }
    let accumulated = '';
    try {
      const messages = stripThinkingFromMessages(contextManager.snapshot());
      const response = await engines[activeEngineId].runChat(
        activeModelId,
        enableThinking,
        enrichedMessage,
        {
          onToken: (chunk) => {
            if (typeof chunk === 'string' && chunk) {
              accumulated += chunk;
              run.partialText = accumulated;
            }
            if (typeof options.onToken === 'function') {
              options.onToken(chunk);
            }
          },
          onReplace: (text) => {
            accumulated = typeof text === 'string' ? text : '';
            run.partialText = accumulated;
            if (typeof options.onReplace === 'function') {
              options.onReplace(text);
            }
          },
          imagePaths,
          audioPaths,
          videoPaths,
          resubmit,
          messages,
        }
      );
      if (!generationOwns(run)) {
        return sealRun(run, run.partialText || accumulated);
      }
      if (run.cancelled) {
        const partial =
          typeof response === 'string' && response.length > 0 ? response : accumulated;
        return sealRun(run, run.stopSealed ? run.partialText : partial);
      }
      const text = typeof response === 'string' ? response : '';
      contextManager.appendAssistant(text);
      return text;
    } catch (err) {
      if (!generationOwns(run) || run.cancelled) {
        return sealRun(run, run.partialText || accumulated);
      }
      contextManager.removeMessageById(userId);
      throw err;
    }
  } finally {
    settleOwnedRun(run);
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
  setDiffusersStopGraceMs,
  setDiffusersWorkerExitMs,
  setDiffusersOffloadBaseDirs,
  killedOffloadPaths,
  setDiffusersKillRecovery,
  joinModelReload,
  isShuttingDown,
  DIFFUSERS_STOP_GRACE_MS,
  DIFFUSERS_WORKER_EXIT_MS,
  sendPrompt,
  cancelGeneration,
  cancelRequestGeneration,
  requestRunTokenCount,
  activeRunStopSealed,
  MAX_TRACKED_REQUEST_RUNS,
  configuredSessionsRoot,
  discardSessionOutput: discardGeneratedOutput,
  getStatus,
  chatbotSupportsThinking,
  chatbotHasChatTemplate,
  chatGenerationStartsInThinking,
  contextSnapshot,
  contextReplace,
  contextClear,
  contextUsage,
  HF_DESTROY_TIMEOUT_MS,
};
