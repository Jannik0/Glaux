const engineManager = require('../../../engines/engineManager');
const state = require('../state');
const {
  getResourcesRoot,
  getOutputsRoot,
  getModelsRoot,
} = require('../paths');
const { updatePreferences } = require('./preferences');

function getEngineInitModelId() {
  return state.selectedModelId;
}

/**
 * @param {string | null} modelId
 * @returns {Promise<void>}
 */
async function persistSelectedModelId(modelId) {
  state.selectedModelId = modelId;
  await updatePreferences({ selectedModelId: modelId });
}

/**
 * Unload any loaded model and clear the selected model preference.
 *
 * @param {{ emitProgress?: boolean }} [opts]
 */
async function enterNoModelState(opts = {}) {
  const { emitProgress = true } = opts;
  try {
    await engineManager.ejectModel();
  } catch {
    /* worker may not be running */
  }
  await persistSelectedModelId(null);
  state.engineBootstrapped = false;
  if (emitProgress) {
    emitInitProgress({ phase: 'unload', status: 'complete' });
  }
}

/**
 * Maps engineManager progress events to the shape expected by renderer `onInitProgress`.
 *
 * @param {object | null | undefined} info
 */
function mapEngineInitProgress(info) {
  if (!info || typeof info !== 'object') {
    return info;
  }
  const perFileStatuses = new Set(['initiate', 'progress', 'done', 'download']);
  if (
    typeof info.status === 'string' &&
    perFileStatuses.has(info.status) &&
    info.phase !== 'loading'
  ) {
    return info;
  }
  const { phase, status, modelId } = info;
  if (phase === 'loading') {
    if (status === 'starting') {
      return {
        status: 'loadStage',
        message: modelId ? `Loading ${modelId}…` : 'Loading model…',
      };
    }
    if (status === 'progress') {
      const rawPercent = Number(info.percent);
      const percent = Number.isFinite(rawPercent)
        ? Math.max(0, Math.min(100, Math.round(rawPercent)))
        : 0;
      const base = modelId ? `Loading ${modelId}…` : 'Loading model…';
      return {
        status: 'loadProgress',
        modelId,
        percent,
        message: `${base} ${percent}%`,
      };
    }
    if (status === 'complete') {
      return { status: 'ready', modelId };
    }
  }
  if (phase === 'download') {
    if (status === 'starting') {
      return { status: 'downloadStart', modelId };
    }
    if (status === 'complete') {
      return { status: 'downloadDone', modelId };
    }
  }
  if (phase === 'unload' && status === 'complete') {
    return { status: 'ejected' };
  }
  return info;
}

function emitInitProgress(info) {
  const mainWindow = state.getMainWindow();
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('engine:initProgress', mapEngineInitProgress(info));
  }
}

/**
 * Status object consumed by the renderer (`pollStatus`, thinking UI).
 */
async function buildEngineStatusForRenderer() {
  const raw = engineManager.getStatus();
  let thinkingSupported = false;
  let chatTemplateSupported = false;
  if (state.engineBootstrapped && raw.modelId) {
    try {
      thinkingSupported = await engineManager.chatbotSupportsThinking();
    } catch {
      thinkingSupported = false;
    }
    try {
      chatTemplateSupported = await engineManager.chatbotHasChatTemplate();
    } catch {
      chatTemplateSupported = false;
    }
  }
  return {
    phase: raw.phase,
    modelId: raw.modelId,
    pipelineTag: raw.pipelineTag ?? null,
    ready: state.engineBootstrapped && raw.phase === 'idle' && Boolean(raw.modelId),
    modelPath: raw.modelId || null,
    thinkingSupported,
    chatTemplateSupported,
  };
}

function getEngineInitOptions() {
  return {
    resourcesRoot: getResourcesRoot(),
    outputsRoot: getOutputsRoot(),
    modelsCacheDir: getModelsRoot(),
    modelId: getEngineInitModelId(),
    onProgress: (info) => {
      emitInitProgress(info);
    },
  };
}

/**
 * Load the selected model after paths are configured.
 *
 * @param {string} modelId
 * @param {boolean} clearedMissing
 * @returns {Promise<object>}
 */
async function finishEngineBootstrapModelLoad(modelId, clearedMissing) {
  try {
    await engineManager.initialize(getEngineInitOptions());
    state.engineBootstrapped = true;
    emitInitProgress({ phase: 'loading', status: 'complete', modelId });
    return {
      modelId,
      fellBack: clearedMissing,
      clearedPreference: clearedMissing,
      loadFailed: false,
      pending: false,
    };
  } catch (loadErr) {
    await enterNoModelState();
    const message =
      loadErr && loadErr.message ? loadErr.message : String(loadErr || 'Failed to load model.');
    emitInitProgress({ status: 'error', message, clearPreference: true });
    return {
      modelId: null,
      fellBack: false,
      clearedPreference: true,
      loadFailed: true,
      message,
      pending: false,
    };
  }
}

module.exports = {
  getEngineInitModelId,
  persistSelectedModelId,
  enterNoModelState,
  mapEngineInitProgress,
  emitInitProgress,
  buildEngineStatusForRenderer,
  getEngineInitOptions,
  finishEngineBootstrapModelLoad,
};
