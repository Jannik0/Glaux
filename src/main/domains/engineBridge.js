const { ipcMain } = require('electron');
const { t } = require('../../i18n');
const engineManager = require('../../../engines/engineManager');
const state = require('../state');
const { ok, fail, toStructuredError, emitStreamEvent } = require('../ipc/result');
const {
  getResourcesRoot,
  getOutputsRoot,
  getSessionsRoot,
  getModelsRoot,
} = require('../paths');
const {
  persistActiveSession,
  reserveActiveSessionFilename,
  clearPendingSessionFilename,
  readActiveSessionFile,
} = require('./sessions');
const { allocateSessionImagePath, releaseSessionImagePath } = require('../sessionImages');
const { isDiffusionPipelineTag } = require('../../../engines/common/resolveEngineId');
const { clearPersistedModelIfNotCached, modelIdHasCachedWeights } = require('./modelsPrefs');
const {
  getEngineInitModelId,
  enterNoModelState,
  emitInitProgress,
  buildEngineStatusForRenderer,
  getEngineInitOptions,
  finishEngineBootstrapModelLoad,
  noteCompanionDownloadRequired,
} = require('./engineCore');

const STREAM_TIMEOUT_MS = 120000;
/** Reset the idle timeout while non-streaming pipelines (e.g. ASR) run without token output. */
const STREAM_KEEPALIVE_MS = 30000;
const MAX_MESSAGE_LENGTH = 100000;

function validateMessagePayload(message, { allowEmpty = false } = {}) {
  if (typeof message !== 'string') {
    throw new Error(t('errors.engineBridge.messageMustBeString'));
  }
  if (!message.trim() && !allowEmpty) {
    throw new Error(t('errors.engineBridge.messageCannotBeEmpty'));
  }
  if (message.length > MAX_MESSAGE_LENGTH) {
    throw new Error(t('errors.engineBridge.messageTooLong', { max: MAX_MESSAGE_LENGTH }));
  }
  return message;
}

/**
 * Reserve a session JSON name and a sibling PNG before a diffusion run.
 * The session file itself is written after the turn, next to that image.
 *
 * @returns {Promise<{ absolutePath: string, relativePath: string, source: 'sessions', reservationId: number } | null>}
 */
async function textToImageOutputPath() {
  const status = engineManager.getStatus();
  if (!status || !isDiffusionPipelineTag(status.pipelineTag)) {
    return null;
  }
  const filename = reserveActiveSessionFilename();
  const sessionsRoot = getSessionsRoot() || engineManager.configuredSessionsRoot();
  try {
    return await allocateSessionImagePath(sessionsRoot, filename);
  } catch (err) {
    if (!state.activeSessionFilename) {
      clearPendingSessionFilename();
    }
    throw err;
  }
}

/**
 * @param {unknown} response
 * @returns {{ text: string, images: Array<object> }}
 */
function splitPromptResult(response) {
  if (response && typeof response === 'object' && Array.isArray(response.imagePaths)) {
    return {
      text: typeof response.text === 'string' ? response.text : '',
      images: Array.isArray(response.images) ? response.images : [],
    };
  }
  return {
    text: typeof response === 'string' ? response : '',
    images: [],
  };
}

function normalizeFilesPayload(rawFiles) {
  if (!Array.isArray(rawFiles)) return [];
  const normalized = [];
  for (const entry of rawFiles) {
    if (!entry || typeof entry !== 'object') continue;
    if (typeof entry.relativePath !== 'string' || !entry.relativePath) continue;
    const source = entry.source === 'outputs' ? 'outputs' : 'resources';
    normalized.push({ source, relativePath: entry.relativePath });
  }
  return normalized;
}

/**
 * Same recovery as a stream idle timeout: reset the worker, reload the
 * selected model, and emit the loading progress that reload already uses.
 * A failure drops to the no-model state. Quit abandons the reload.
 *
 * @param {string} logLabel
 */
async function recoverLoadedModelAfterWorkerReset(logLabel) {
  try {
    if (engineManager.isShuttingDown()) {
      return { ok: false };
    }
    await engineManager.resetInferenceWorker();
    await clearPersistedModelIfNotCached();
    const modelId = getEngineInitModelId();
    if (modelId && (await modelIdHasCachedWeights(modelId))) {
      if (engineManager.isShuttingDown()) {
        return { ok: false };
      }
      const loaded = await engineManager.reinitialize({
        ...getEngineInitOptions(),
        allowDuringDiffusersStop: true,
      });
      if (loaded && loaded.downloadRequired) {
        await noteCompanionDownloadRequired(modelId);
      } else {
        state.engineBootstrapped = true;
      }
      return { ok: true };
    }
    await enterNoModelState({ allowDuringDiffusersStop: true });
    return { ok: false };
  } catch (reinitErr) {
    console.error(logLabel, reinitErr);
    try {
      if (!engineManager.isShuttingDown()) {
        await enterNoModelState({ allowDuringDiffusersStop: true });
      }
    } catch {
      /* ignore */
    }
    return { ok: false };
  }
}

/**
 * One shared reload for the Diffusers stop kill and a stream idle timeout.
 * A second caller waits for the reload already in flight.
 *
 * @param {string} logLabel
 * @returns {Promise<unknown>}
 */
function requestWorkerReload(logLabel) {
  return engineManager.joinModelReload(() => recoverLoadedModelAfterWorkerReset(logLabel));
}

async function persistTurnSession(logLabel, deps) {
  try {
    await persistActiveSession(deps);
  } catch (persistErr) {
    console.error(logLabel, persistErr);
    // Keep the sidecar. The live context and the done event already reference
    // it; the next successful save writes that path again.
    if (!state.activeSessionFilename) {
      clearPendingSessionFilename();
    }
  }
}

/**
 * Non-streaming send. Persists through the same helper as the stream handler.
 * A failed turn clears a still-unpublished session name. The generated PNG is
 * discarded inside sendPrompt.
 *
 * @param {unknown} payload
 * @param {{ outputPath?: string, reservationId?: number, persist?: object, requestId?: string, senderId?: unknown }} [deps]
 * @returns {Promise<{ response: string, images: Array<object> }>}
 */
async function deliverSendMessage(payload, deps = {}) {
  const { message, enableThinking, resubmit, files } = parseInferenceRequest(payload);
  let outputPath;
  let reservationId;
  if (Object.prototype.hasOwnProperty.call(deps, 'outputPath')) {
    outputPath = deps.outputPath;
    reservationId = deps.reservationId;
  } else {
    const allocated = await textToImageOutputPath();
    outputPath = allocated ? allocated.absolutePath : undefined;
    reservationId = allocated ? allocated.reservationId : undefined;
  }
  try {
    const response = await engineManager.sendPrompt(message, {
      enableThinking,
      resubmit,
      files,
      outputPath,
      reservationId,
      requestId: deps.requestId,
      senderId: deps.senderId,
    });
    const split = splitPromptResult(response);
    await persistTurnSession('Failed to persist session after chat turn:', deps.persist);
    return { response: split.text, images: split.images };
  } catch (err) {
    releaseSessionImagePath(outputPath, reservationId);
    if (!state.activeSessionFilename) {
      clearPendingSessionFilename();
    }
    throw err;
  }
}

/**
 * @param {unknown} payload
 * @returns {Promise<{ ok: boolean, response?: string, images?: Array<object>, error?: string, errorInfo?: object }>}
 */
async function handleEngineSendMessage(payload) {
  if (!state.engineBootstrapped) {
    return fail(new Error(t('errors.engineBridge.engineNotInitialized')), 'E_NOT_READY');
  }
  try {
    return ok(await deliverSendMessage(payload));
  } catch (err) {
    return fail(err, 'E_INFERENCE');
  }
}

/**
 * Cancel the run bound to this stream request. An unknown request id does not
 * cancel the active run. Sender fallback is only for sender destruction of the
 * request that owns the active run.
 *
 * @param {string} requestId
 * @param {{ id?: unknown } | null | undefined} sender
 * @param {{ senderId?: unknown, senderFallback?: boolean }} [options]
 */
function cancelStreamRequest(requestId, sender, options = {}) {
  const id = requestId ? String(requestId) : '';
  const senderId =
    options && options.senderId != null
      ? options.senderId
      : sender && sender.id != null
        ? sender.id
        : undefined;
  engineManager.cancelRequestGeneration({
    requestId: id,
    senderId,
    senderFallback: Boolean(options && options.senderFallback),
  });
  if (!id) {
    return;
  }
  const streamState = state.activeStreamRequests.get(id);
  if (streamState) {
    streamState.canceled = true;
  }
}

/**
 * Explicit Stop for whatever run is active.
 * @returns {Promise<{ ok: boolean, error?: string, errorInfo?: object }>}
 */
async function handleEngineCancel() {
  try {
    engineManager.cancelGeneration();
    return ok();
  } catch (err) {
    return fail(err, 'E_CANCELED');
  }
}

function parseInferenceRequest(payload) {
  if (typeof payload === 'string') {
    return { message: validateMessagePayload(payload), enableThinking: false, resubmit: true, files: [] };
  }
  if (payload && typeof payload === 'object' && typeof payload.message === 'string') {
    const files = normalizeFilesPayload(payload.files);
    return {
      message: validateMessagePayload(payload.message, { allowEmpty: files.length > 0 }),
      enableThinking: payload.enableThinking === true,
      resubmit: payload.resubmit !== false,
      files,
    };
  }
  throw new Error(t('errors.engineBridge.invalidInferencePayload'));
}

/**
 * @param {{ seq?: number } | null | undefined} streamState
 * @param {object} payload
 * @returns {object}
 */
function stampStreamEvent(streamState, payload) {
  if (!streamState) {
    return payload;
  }
  streamState.seq = (streamState.seq || 0) + 1;
  return { ...payload, seq: streamState.seq };
}

/**
 * @param {string} requestId
 * @param {{ text?: string, seq?: number, startsInThinking?: boolean, userText?: string, userFiles?: Array<object> }} streamState
 * @param {boolean} userPending
 */
function describeStreamState(requestId, streamState, userPending) {
  return {
    requestId,
    text: typeof streamState.text === 'string' ? streamState.text : '',
    startsInThinking: Boolean(streamState.startsInThinking),
    seq: streamState.seq || 0,
    userPending: Boolean(userPending),
    userText: userPending && typeof streamState.userText === 'string' ? streamState.userText : '',
    userFiles: userPending && Array.isArray(streamState.userFiles) ? streamState.userFiles : [],
  };
}

/**
 * The generation a language-switch reload should paint. A committed turn is
 * already in context, so it is not reported here.
 * @returns {object | null}
 */
function activeStreamSnapshot() {
  /** @type {object | null} */
  let preparing = null;
  for (const [requestId, streamState] of state.activeStreamRequests) {
    if (!streamState || streamState.canceled || streamState.committed) {
      continue;
    }
    const pending = engineManager.streamRunUserPending(requestId);
    if (pending == null) {
      preparing = describeStreamState(requestId, streamState, true);
      continue;
    }
    return describeStreamState(requestId, streamState, pending);
  }
  return preparing;
}

/**
 * Transcript plus any in-flight generation. Context is the live transcript;
 * the session file fills in only when context is still empty.
 * @returns {Promise<{ sessionName: string | null, messages: Array<object>, stream: object | null }>}
 */
async function readChatRestoreState() {
  let messages = await engineManager.contextSnapshot();
  let stream = activeStreamSnapshot();
  if (!stream) {
    messages = await engineManager.contextSnapshot();
    stream = activeStreamSnapshot();
  }
  if (
    (!Array.isArray(messages) || messages.length === 0) &&
    state.activeSessionFilename &&
    !stream
  ) {
    try {
      const stored = await readActiveSessionFile();
      if (Array.isArray(stored) && stored.length > 0) {
        messages = stored;
      }
    } catch {
      /* unsaved, or the workspace path is not ready */
    }
  }
  return {
    sessionName: state.activeSessionFilename,
    messages: Array.isArray(messages) ? messages : [],
    stream,
  };
}

function handleEngineStreamStart(event, payload) {
  if (!payload || typeof payload !== 'object') {
    return Promise.resolve();
  }
  const requestId = payload.requestId
    ? String(payload.requestId)
    : `stream-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const sender = event.sender;
  const rawSenderId = sender ? sender.id : undefined;
  const senderId = rawSenderId != null ? rawSenderId : undefined;

  if (!state.engineBootstrapped) {
    const errorInfo = toStructuredError(new Error(t('errors.engineBridge.engineNotInitialized')), 'E_NOT_READY');
    emitStreamEvent(sender, { requestId, type: 'error', errorInfo });
    return Promise.resolve();
  }

  let message;
  let enableThinking = false;
  let resubmit = true;
  let files = [];
  try {
    const parsed = parseInferenceRequest(payload);
    message = parsed.message;
    enableThinking = parsed.enableThinking;
    resubmit = parsed.resubmit;
    files = parsed.files;
  } catch (validationErr) {
    const errorInfo = toStructuredError(validationErr, 'E_INPUT');
    emitStreamEvent(sender, { requestId, type: 'error', errorInfo });
    return Promise.resolve();
  }

  if (state.activeStreamRequests.has(requestId)) {
    const errorInfo = toStructuredError(new Error(t('errors.engineBridge.duplicateStreamRequestId')), 'E_DUPLICATE');
    emitStreamEvent(sender, { requestId, type: 'error', errorInfo });
    return Promise.resolve();
  }

  const streamState = {
    canceled: false,
    committed: false,
    sender,
    timeoutHandle: null,
    keepaliveHandle: null,
    text: '',
    seq: 0,
    startsInThinking: false,
    userText: message,
    userFiles: files,
  };
  state.activeStreamRequests.set(requestId, streamState);

  function onSenderDestroyed() {
    const activeState = state.activeStreamRequests.get(requestId);
    if (activeState) {
      activeState.canceled = true;
      cleanupStream();
      cancelStreamRequest(requestId, null, { senderId, senderFallback: true });
    }
  }

  const cleanupStream = () => {
    if (streamState.keepaliveHandle) {
      clearInterval(streamState.keepaliveHandle);
      streamState.keepaliveHandle = null;
    }
    if (streamState.timeoutHandle) {
      clearTimeout(streamState.timeoutHandle);
      streamState.timeoutHandle = null;
    }
    try {
      sender.removeListener('destroyed', onSenderDestroyed);
    } catch {
      /* webContents already gone */
    }
    state.activeStreamRequests.delete(requestId);
  };

  sender.once('destroyed', onSenderDestroyed);

  return Promise.resolve().then(async () => {
    let startsInThinking = false;
    if (enableThinking) {
      try {
        startsInThinking = await engineManager.chatGenerationStartsInThinking(message, {
          enableThinking,
          resubmit,
          files,
        });
      } catch {
        startsInThinking = false;
      }
    }
    if (streamState.canceled) {
      cleanupStream();
      return;
    }
    streamState.startsInThinking = startsInThinking;
    emitStreamEvent(sender, stampStreamEvent(streamState, { requestId, type: 'started', startsInThinking }));

    let streamedAnyChunk = false;
    let rejectTimeout;
    const resetTimeout = () => {
      if (streamState.timeoutHandle) clearTimeout(streamState.timeoutHandle);
      streamState.timeoutHandle = setTimeout(
        () => {
          const timeoutErr = new Error(t('errors.engineBridge.streamTimedOut'));
          timeoutErr.code = 'E_TIMEOUT';
          rejectTimeout && rejectTimeout(timeoutErr);
        },
        STREAM_TIMEOUT_MS
      );
    };
    let sendPromptPromise;
    let outputPath;
    let reservationId;
    try {
      const allocated = await textToImageOutputPath();
      outputPath = allocated ? allocated.absolutePath : undefined;
      reservationId = allocated ? allocated.reservationId : undefined;
      sendPromptPromise = engineManager.sendPrompt(message, {
        enableThinking,
        resubmit,
        files,
        outputPath,
        reservationId,
        requestId,
        senderId,
        onToken: (chunk) => {
          if (streamState.canceled) {
            return;
          }
          if (typeof chunk === 'string' && chunk.length > 0) {
            streamedAnyChunk = true;
            streamState.text = `${streamState.text || ''}${chunk}`;
            emitStreamEvent(
              sender,
              stampStreamEvent(streamState, {
                requestId,
                type: 'chunk',
                chunk,
                text: streamState.text,
              })
            );
            resetTimeout();
          }
        },
        onReplace: (text) => {
          if (streamState.canceled) {
            return;
          }
          if (typeof text === 'string') {
            streamedAnyChunk = true;
            streamState.text = text;
            emitStreamEvent(
              sender,
              stampStreamEvent(streamState, { requestId, type: 'snapshot', text })
            );
            resetTimeout();
          }
        },
        onSettle: () => {
          streamState.committed = true;
        },
      });
      streamState.keepaliveHandle = setInterval(() => {
        if (!streamState.canceled) {
          resetTimeout();
        }
      }, STREAM_KEEPALIVE_MS);
      const response = await Promise.race([
        sendPromptPromise,
        new Promise((_, reject) => {
          rejectTimeout = reject;
          resetTimeout();
        }),
      ]);

      if (streamState.canceled) {
        emitStreamEvent(sender, stampStreamEvent(streamState, { requestId, type: 'canceled' }));
        await persistTurnSession('Failed to persist session after canceled chat turn:');
        return;
      }

      const split = splitPromptResult(response);
      if (!streamedAnyChunk && split.text.length > 0) {
        streamState.text = `${streamState.text || ''}${split.text}`;
        emitStreamEvent(
          sender,
          stampStreamEvent(streamState, {
            requestId,
            type: 'chunk',
            chunk: split.text,
            text: streamState.text,
          })
        );
      }
      emitStreamEvent(
        sender,
        stampStreamEvent(streamState, {
          requestId,
          type: 'done',
          response: split.text,
          images: split.images,
        })
      );
      await persistTurnSession('Failed to persist session after chat turn:');
    } catch (err) {
      streamState.committed = true;
      if (sendPromptPromise) {
        sendPromptPromise.catch(() => {});
      }
      if (streamState.canceled) {
        emitStreamEvent(sender, stampStreamEvent(streamState, { requestId, type: 'canceled' }));
        await persistTurnSession('Failed to persist session after canceled chat turn:');
        return;
      }
      const isStreamIdleTimeout =
        (err && err.code === 'E_TIMEOUT') ||
        /stream timed out/i.test(String(err && err.message ? err.message : err));
      if (isStreamIdleTimeout) {
        await requestWorkerReload(
          'Failed to reinitialize inference engine after stream timeout:'
        );
      } else {
        engineManager.cancelRequestGeneration({ requestId });
      }
      if (!state.activeSessionFilename) {
        clearPendingSessionFilename();
      }
      const errorInfo = toStructuredError(err, 'E_INFERENCE');
      emitStreamEvent(sender, stampStreamEvent(streamState, { requestId, type: 'error', errorInfo }));
    } finally {
      releaseSessionImagePath(outputPath, reservationId);
      cleanupStream();
    }
    });
}

function registerEngineBridgeIpc() {
  ipcMain.handle('engine:bootstrap', async () => {
    if (state.engineBootstrapped) {
      return ok({
        modelId: getEngineInitModelId(),
        fellBack: false,
        clearedPreference: false,
        loadFailed: false,
        pending: false,
      });
    }
    if (state.engineBootstrapLoadPromise) {
      return ok(await state.engineBootstrapLoadPromise);
    }
    try {
      await engineManager.configurePaths({
        resourcesRoot: getResourcesRoot(),
        outputsRoot: getOutputsRoot(),
        sessionsRoot: getSessionsRoot(),
        modelsCacheDir: getModelsRoot(),
        onProgress: (info) => emitInitProgress(info),
      });

      const clearedMissing = await clearPersistedModelIfNotCached();
      const modelId = getEngineInitModelId();
      if (!modelId) {
        emitInitProgress({ phase: 'unload', status: 'complete' });
        return ok({
          modelId: null,
          fellBack: false,
          clearedPreference: clearedMissing,
          loadFailed: false,
          pending: false,
        });
      }

      const immediate = ok({
        modelId,
        fellBack: clearedMissing,
        clearedPreference: clearedMissing,
        loadFailed: false,
        pending: true,
      });

      state.engineBootstrapLoadPromise = finishEngineBootstrapModelLoad(modelId, clearedMissing).finally(
        () => {
          state.engineBootstrapLoadPromise = null;
        }
      );

      void state.engineBootstrapLoadPromise;

      return immediate;
    } catch (err) {
      return fail(err, 'E_INIT');
    }
  });

  ipcMain.handle('engine:sendMessage', async (_event, payload) => handleEngineSendMessage(payload));

  ipcMain.handle('engine:getStatus', async () => {
    try {
      const status = await buildEngineStatusForRenderer();
      return ok({ status });
    } catch (err) {
      return fail(err, 'E_STATUS');
    }
  });

  ipcMain.handle('engine:getContextUsage', async (_event, payload) => {
    try {
      const resubmit = !(payload && payload.resubmit === false);
      const refresh = Boolean(payload && payload.refresh);
      const usage = await engineManager.contextUsage(resubmit, { refresh });
      return ok({ usage });
    } catch (err) {
      return fail(err, 'E_CONTEXT_USAGE');
    }
  });

  ipcMain.handle('engine:reinitialize', async () => {
    try {
      const clearedMissing = await clearPersistedModelIfNotCached();
      const modelId = getEngineInitModelId();
      if (!modelId) {
        throw new Error(t('errors.engineBridge.noModelSelected'));
      }
      if (!(await modelIdHasCachedWeights(modelId))) {
        await enterNoModelState();
        throw new Error(t('errors.engineBridge.modelNotDownloaded', { modelId }));
      }
      const loaded = await engineManager.reinitialize(getEngineInitOptions());
      if (loaded && loaded.downloadRequired) {
        await noteCompanionDownloadRequired(modelId);
        return ok({
          modelId: null,
          fellBack: false,
          clearedPreference: true,
          loadFailed: false,
          downloadRequired: true,
        });
      }
      state.engineBootstrapped = true;
      emitInitProgress({ phase: 'loading', status: 'complete', modelId });
      return ok({
        modelId,
        fellBack: clearedMissing,
        clearedPreference: clearedMissing,
        loadFailed: false,
      });
    } catch (err) {
      return fail(err, 'E_INIT');
    }
  });

  ipcMain.handle('engine:eject', async () => {
    try {
      await enterNoModelState();
      return ok({ modelId: null });
    } catch (err) {
      return fail(err, 'E_EJECT');
    }
  });

  ipcMain.on('engine:streamStart', (event, payload) => {
    void handleEngineStreamStart(event, payload);
  });

  ipcMain.on('engine:streamCancel', (event, payload) => {
    const requestId = payload && payload.requestId ? String(payload.requestId) : '';
    cancelStreamRequest(requestId, event && event.sender);
  });

  ipcMain.handle('engine:cancel', async () => handleEngineCancel());

  ipcMain.handle('engine:getChatRestoreState', async () => {
    try {
      return ok(await readChatRestoreState());
    } catch (err) {
      return fail(err, 'E_RESTORE');
    }
  });
}

engineManager.setDiffusersKillRecovery(() =>
  requestWorkerReload('Failed to reload the model after stopping image generation:')
);

module.exports = {
  registerEngineBridgeIpc,
  deliverSendMessage,
  handleEngineSendMessage,
  handleEngineStreamStart,
  handleEngineCancel,
  readChatRestoreState,
  cancelStreamRequest,
  requestWorkerReload,
};
