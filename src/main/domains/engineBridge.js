const { ipcMain } = require('electron');
const { t } = require('../../i18n');
const engineManager = require('../../../engines/engineManager');
const state = require('../state');
const { ok, fail, toStructuredError, emitStreamEvent } = require('../ipc/result');
const {
  getResourcesRoot,
  getOutputsRoot,
  getModelsRoot,
} = require('../paths');
const { persistActiveSession } = require('./sessions');
const { clearPersistedModelIfNotCached, modelIdHasCachedWeights } = require('./modelsPrefs');
const {
  getEngineInitModelId,
  enterNoModelState,
  emitInitProgress,
  buildEngineStatusForRenderer,
  getEngineInitOptions,
  finishEngineBootstrapModelLoad,
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

  ipcMain.handle('engine:sendMessage', async (_event, payload) => {
    if (!state.engineBootstrapped) {
      return fail(new Error(t('errors.engineBridge.engineNotInitialized')), 'E_NOT_READY');
    }
    try {
      const { message, enableThinking, resubmit, files } = parseInferenceRequest(payload);
      const response = await engineManager.sendPrompt(message, { enableThinking, resubmit, files });
      return ok({ response });
    } catch (err) {
      return fail(err, 'E_INFERENCE');
    }
  });

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
      await engineManager.reinitialize(getEngineInitOptions());
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
    if (!payload || typeof payload !== 'object') {
      return;
    }
    const requestId = payload.requestId
      ? String(payload.requestId)
      : `stream-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const sender = event.sender;

    if (!state.engineBootstrapped) {
      const errorInfo = toStructuredError(new Error(t('errors.engineBridge.engineNotInitialized')), 'E_NOT_READY');
      emitStreamEvent(sender, { requestId, type: 'error', errorInfo });
      return;
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
      return;
    }

    if (state.activeStreamRequests.has(requestId)) {
      const errorInfo = toStructuredError(new Error(t('errors.engineBridge.duplicateStreamRequestId')), 'E_DUPLICATE');
      emitStreamEvent(sender, { requestId, type: 'error', errorInfo });
      return;
    }

    const streamState = { canceled: false, sender, timeoutHandle: null, keepaliveHandle: null };
    state.activeStreamRequests.set(requestId, streamState);

    const cleanupStream = () => {
      if (streamState.keepaliveHandle) {
        clearInterval(streamState.keepaliveHandle);
        streamState.keepaliveHandle = null;
      }
      if (streamState.timeoutHandle) {
        clearTimeout(streamState.timeoutHandle);
        streamState.timeoutHandle = null;
      }
      state.activeStreamRequests.delete(requestId);
    };

    sender.once('destroyed', () => {
      const activeState = state.activeStreamRequests.get(requestId);
      if (activeState) {
        activeState.canceled = true;
        cleanupStream();
        engineManager.cancelGeneration();
      }
    });

    Promise.resolve().then(async () => {
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
      emitStreamEvent(sender, { requestId, type: 'started', startsInThinking });

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
      try {
        sendPromptPromise = engineManager.sendPrompt(message, {
          enableThinking,
          resubmit,
          files,
          onToken: (chunk) => {
            if (streamState.canceled) {
              return;
            }
            if (typeof chunk === 'string' && chunk.length > 0) {
              streamedAnyChunk = true;
              emitStreamEvent(sender, { requestId, type: 'chunk', chunk });
              resetTimeout();
            }
          },
          onReplace: (text) => {
            if (streamState.canceled) {
              return;
            }
            if (typeof text === 'string') {
              streamedAnyChunk = true;
              emitStreamEvent(sender, { requestId, type: 'snapshot', text });
              resetTimeout();
            }
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
          emitStreamEvent(sender, { requestId, type: 'canceled' });
          try {
            await persistActiveSession();
          } catch (persistErr) {
            console.error('Failed to persist session after canceled chat turn:', persistErr);
          }
          return;
        }

        if (!streamedAnyChunk && typeof response === 'string' && response.length > 0) {
          emitStreamEvent(sender, { requestId, type: 'chunk', chunk: response });
        }
        emitStreamEvent(sender, { requestId, type: 'done', response: response || '' });
        try {
          await persistActiveSession();
        } catch (persistErr) {
          console.error('Failed to persist session after chat turn:', persistErr);
        }
      } catch (err) {
        if (sendPromptPromise) {
          sendPromptPromise.catch(() => {});
        }
        if (streamState.canceled) {
          emitStreamEvent(sender, { requestId, type: 'canceled' });
          try {
            await persistActiveSession();
          } catch (persistErr) {
            console.error('Failed to persist session after canceled chat turn:', persistErr);
          }
          return;
        }
        const isStreamIdleTimeout =
          (err && err.code === 'E_TIMEOUT') ||
          /stream timed out/i.test(String(err && err.message ? err.message : err));
        if (isStreamIdleTimeout) {
          try {
            await engineManager.resetInferenceWorker();
            await clearPersistedModelIfNotCached();
            const modelId = getEngineInitModelId();
            if (modelId && (await modelIdHasCachedWeights(modelId))) {
              await engineManager.reinitialize(getEngineInitOptions());
              state.engineBootstrapped = true;
            } else {
              await enterNoModelState();
            }
          } catch (reinitErr) {
            console.error('Failed to reinitialize inference engine after stream timeout:', reinitErr);
            try {
              await enterNoModelState();
            } catch {
              /* ignore */
            }
          }
        } else {
          engineManager.cancelGeneration();
        }
        const errorInfo = toStructuredError(err, 'E_INFERENCE');
        emitStreamEvent(sender, { requestId, type: 'error', errorInfo });
      } finally {
        cleanupStream();
      }
    });
  });

  ipcMain.on('engine:streamCancel', (_event, payload) => {
    engineManager.cancelGeneration();
    const requestId = payload && payload.requestId ? String(payload.requestId) : '';
    if (!requestId) {
      return;
    }
    const streamState = state.activeStreamRequests.get(requestId);
    if (streamState) {
      streamState.canceled = true;
    }
  });

  ipcMain.handle('engine:cancel', async () => {
    try {
      engineManager.cancelGeneration();
      return ok();
    } catch (err) {
      return fail(err, 'E_CANCELED');
    }
  });
}

module.exports = {
  registerEngineBridgeIpc,
};
