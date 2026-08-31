const fs = require('fs/promises');
const { shell } = require('electron');
const { ipcMain } = require('electron');
const engineManager = require('../../../engines/engineManager');
const {
  directoryContainsModelWeights,
  classifyHubRepoFiles,
  buildGgufAllowPatterns,
} = require('../../../engines/common/modelFormat');
const state = require('../state');
const { ok, fail } = require('../ipc/result');
const {
  assertValidHfRepoId,
  resolveModelsCacheAbsoluteDir,
  modelCacheFolderAbsPath,
  getModelsRoot,
} = require('../paths');
const {
  emitInitProgress,
  enterNoModelState,
  getEngineInitOptions,
  persistSelectedModelId,
} = require('./engineCore');
const { readModelPipelineTag } = require('../../../engines/common/pipelineTag');
const { t } = require('../../i18n');

/**
 * Discover Hub-style model ids under `cacheRoot` (`namespace/repo`).
 * Detects Transformers layouts (`config.json`) and GGUF-only repos (`*.gguf`).
 *
 * @param {string} cacheRoot
 * @returns {Promise<string[]>}
 */
async function listCachedHfModels(cacheRoot) {
  const path = require('path');
  const found = new Set();

  async function walk(currentDir) {
    let entries;
    try {
      entries = await fs.readdir(currentDir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const ent of entries) {
      const full = path.join(currentDir, ent.name);
      if (ent.isDirectory()) {
        await walk(full);
      } else if (ent.name === 'config.json' || /\.gguf$/i.test(ent.name)) {
        if (/^mmproj/i.test(ent.name)) {
          continue;
        }
        const parentDir = path.dirname(full);
        const rel = path.relative(cacheRoot, parentDir);
        if (!rel || rel.startsWith('..')) {
          continue;
        }
        const parts = rel.split(path.sep);
        if (parts.length >= 2) {
          found.add(parts.slice(0, 2).join('/'));
        } else {
          found.add(parts.join('/'));
        }
      }
    }
  }

  try {
    await fs.access(cacheRoot);
  } catch {
    return [];
  }
  await walk(cacheRoot);

  const withWeights = [];
  for (const id of found) {
    const modelRoot = path.join(cacheRoot, ...id.split('/'));
    if (await directoryContainsModelWeights(modelRoot)) {
      withWeights.push(id);
    }
  }

  return withWeights.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
}

/**
 * @param {string | null | undefined} modelId
 * @returns {Promise<boolean>}
 */
async function modelIdHasCachedWeights(modelId) {
  if (!modelId || typeof modelId !== 'string') {
    return false;
  }
  try {
    const abs = await resolveModelsCacheAbsoluteDir(modelId);
    return await directoryContainsModelWeights(abs);
  } catch {
    return false;
  }
}

/**
 * Clear the active default when its weights are not in the cache (no auto-pick of another model).
 *
 * @returns {Promise<boolean>} True when the preference was cleared.
 */
async function clearPersistedModelIfNotCached() {
  const modelId = state.selectedModelId;
  if (!modelId) {
    return false;
  }
  if (await modelIdHasCachedWeights(modelId)) {
    return false;
  }
  await persistSelectedModelId(null);
  return true;
}

/**
 * @param {string} modelId
 */
async function trashModelCacheFolder(modelId) {
  const abs = modelCacheFolderAbsPath(modelId);
  try {
    await fs.access(abs);
  } catch (err) {
    if (err && err.code === 'ENOENT') {
      return;
    }
    throw err;
  }

  const maxAttempts = 8;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      await shell.trashItem(abs);
      return;
    } catch (err) {
      const locked =
        err &&
        (err.code === 'EBUSY' ||
          err.code === 'EPERM' ||
          err.code === 'EACCES' ||
          /busy|locked|in use/i.test(String(err.message || '')));
      if (!locked || attempt === maxAttempts - 1) {
        try {
          await fs.rm(abs, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 });
          return;
        } catch (rmErr) {
          if (rmErr && rmErr.code === 'ENOENT') {
            return;
          }
          throw rmErr;
        }
      }
      await new Promise((resolve) => setTimeout(resolve, 400 * (attempt + 1)));
    }
  }
}

function registerModelsPrefsIpc() {
  ipcMain.handle('prefs:getSelectedModelId', async () => {
    try {
      return ok({ modelId: state.selectedModelId });
    } catch (err) {
      return fail(err, 'E_PREFS');
    }
  });

  ipcMain.handle('prefs:setSelectedModelId', async (_event, payload) => {
    try {
      const raw = payload && typeof payload.modelId === 'string' ? payload.modelId : '';
      await persistSelectedModelId(assertValidHfRepoId(raw));
      return ok({ modelId: state.selectedModelId });
    } catch (err) {
      return fail(err, 'E_PREFS');
    }
  });

  ipcMain.handle('prefs:listCachedModels', async () => {
    try {
      const modelsRoot = getModelsRoot();
      const ids = await listCachedHfModels(modelsRoot);
      const models = await Promise.all(
        ids.map(async (id) => ({
          id,
          pipelineTag: await readModelPipelineTag(modelsRoot, id),
        }))
      );
      return ok({ models });
    } catch (err) {
      return fail(err, 'E_PREFS');
    }
  });

  ipcMain.handle('prefs:moveCachedModelToTrash', async (_event, payload) => {
    try {
      const modelId = assertValidHfRepoId(payload && payload.modelId);
      const status = engineManager.getStatus();
      if (status.phase === 'generating' || status.phase === 'downloading' || status.phase === 'loading') {
        throw new Error(t('errors.modelsPrefs.waitUntilIdle'));
      }
      await resolveModelsCacheAbsoluteDir(modelId);
      const wasSelected = state.selectedModelId === modelId;
      const isLoaded = status.modelId === modelId;

      if (isLoaded) {
        if (wasSelected) {
          await enterNoModelState();
        } else {
          await engineManager.ejectModel();
          state.engineBootstrapped = false;
          emitInitProgress({ phase: 'unload', status: 'complete' });
        }
      } else if (wasSelected) {
        await persistSelectedModelId(null);
        state.engineBootstrapped = false;
      }

      await trashModelCacheFolder(modelId);
      return ok({ modelId, wasSelected });
    } catch (err) {
      return fail(err, 'E_PREFS');
    }
  });

  ipcMain.handle('models:probeHubRepo', async (_event, payload) => {
    try {
      const modelId = assertValidHfRepoId(payload && payload.modelId);
      const files = await engineManager.listHubModelFiles(modelId, {
        modelsCacheDir: getModelsRoot(),
      });
      const classified = classifyHubRepoFiles(files);
      return ok({
        modelId,
        kind: classified.kind,
        variants: classified.variants,
        files,
      });
    } catch (err) {
      return fail(err, 'E_PROBE');
    }
  });

  ipcMain.handle('models:downloadModel', async (event, payload) => {
    try {
      const modelId = assertValidHfRepoId(payload && payload.modelId);
      const sender = event.sender;
      let allowPatterns = Array.isArray(payload && payload.allowPatterns)
        ? payload.allowPatterns.filter((p) => typeof p === 'string' && p.trim())
        : undefined;
      const ggufVariant =
        payload && typeof payload.ggufVariant === 'string' ? payload.ggufVariant.trim() : undefined;

      if ((!allowPatterns || !allowPatterns.length) && ggufVariant) {
        const files = await engineManager.listHubModelFiles(modelId, {
          modelsCacheDir: getModelsRoot(),
        });
        const classified = classifyHubRepoFiles(files);
        const variant = classified.variants.find((v) => v.key === ggufVariant);
        if (!variant) {
          throw new Error(t('errors.modelsPrefs.ggufVariantNotFound', { variant: ggufVariant }));
        }
        allowPatterns = buildGgufAllowPatterns(files, variant);
      }

      const downloadPromise = engineManager.downloadModel({
        modelId,
        modelsCacheDir: getModelsRoot(),
        allowPatterns,
        ggufVariant,
        onProgress: (info) => {
          if (sender && !sender.isDestroyed()) {
            sender.send('models:downloadProgress', { modelId, event: info });
          }
        },
      });
      state.activePanelModelDownload = { modelId, promise: downloadPromise };
      try {
        await downloadPromise;
        return ok({ modelId });
      } finally {
        if (state.activePanelModelDownload && state.activePanelModelDownload.modelId === modelId) {
          state.activePanelModelDownload = null;
        }
      }
    } catch (err) {
      if (err && (err.code === 'DOWNLOAD_CANCELLED' || /cancelled/i.test(String(err.message || '')))) {
        return fail(err, 'E_DOWNLOAD_CANCELLED');
      }
      return fail(err, 'E_DOWNLOAD');
    }
  });

  ipcMain.handle('models:cancelModelDownload', async (_event, payload) => {
    try {
      const modelId = assertValidHfRepoId(payload && payload.modelId);
      const active =
        state.activePanelModelDownload && state.activePanelModelDownload.modelId === modelId
          ? state.activePanelModelDownload
          : null;
      let workerWasReset = false;
      if (active) {
        await engineManager.cancelModelDownload(modelId, { modelsCacheDir: getModelsRoot() });
        let downloadSettled = false;
        await Promise.race([
          active.promise
            .catch(() => {})
            .finally(() => {
              downloadSettled = true;
            }),
          new Promise((resolve) => setTimeout(resolve, 8000)),
        ]);
        if (!downloadSettled) {
          await engineManager.resetInferenceWorker();
          workerWasReset = true;
          state.activePanelModelDownload = null;
        }
      }
      await trashModelCacheFolder(modelId);
      if (workerWasReset && state.engineBootstrapped && state.selectedModelId) {
        try {
          if (await modelIdHasCachedWeights(state.selectedModelId)) {
            await engineManager.reinitialize(getEngineInitOptions());
          } else {
            await enterNoModelState();
          }
        } catch {
          await enterNoModelState({ emitProgress: false });
        }
      }
      return ok({ modelId, cancelled: true });
    } catch (err) {
      return fail(err, 'E_DOWNLOAD_CANCEL');
    }
  });
}

module.exports = {
  registerModelsPrefsIpc,
  modelIdHasCachedWeights,
  clearPersistedModelIfNotCached,
};
