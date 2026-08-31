'use strict';

const path = require('path');
const fs = require('fs/promises');
const fsSync = require('fs');
const { ipcMain } = require('electron');
const { getUserDataRoot } = require('../paths');
const { ok, fail } = require('../ipc/result');
const { t, isSupportedLanguage } = require('../../i18n');

const PREFERENCES_FILE = 'preferences.json';

const DEFAULT_SIDE_PANELS = Object.freeze({
  models: false,
  sessions: false,
  resources: false,
  outputs: false,
});

function defaultPreferences() {
  return {
    activeWorkspace: null,
    selectedModelId: null,
    sidePanelsCollapsed: { ...DEFAULT_SIDE_PANELS },
    reasoningEnabled: false,
    resubmitEnabled: false,
    language: null,
  };
}

/** @type {ReturnType<typeof defaultPreferences> | null} */
let cached = null;

function preferencesFilePath() {
  return path.join(getUserDataRoot(), PREFERENCES_FILE);
}

/**
 * Read the saved UI language without loading the full preferences cache.
 * Safe before `loadPreferences()` (used to set Chromium `--lang` pre-ready).
 * @returns {string | null}
 */
function readPersistedLanguageFromDisk() {
  try {
    const raw = JSON.parse(fsSync.readFileSync(preferencesFilePath(), 'utf8'));
    return isSupportedLanguage(raw.language) ? raw.language : null;
  } catch {
    return null;
  }
}

/**
 * @param {unknown} raw
 * @returns {ReturnType<typeof defaultPreferences>}
 */
function normalizePreferences(raw) {
  const prefs = defaultPreferences();
  if (!raw || typeof raw !== 'object') {
    return prefs;
  }
  const record = /** @type {Record<string, unknown>} */ (raw);

  if (typeof record.activeWorkspace === 'string' && record.activeWorkspace.trim()) {
    prefs.activeWorkspace = record.activeWorkspace.trim();
  } else {
    prefs.activeWorkspace = null;
  }

  if (record.selectedModelId === null) {
    prefs.selectedModelId = null;
  } else if (typeof record.selectedModelId === 'string') {
    const id = record.selectedModelId.trim();
    prefs.selectedModelId = /^[\w.-]+\/[\w.-]+$/.test(id) ? id : null;
  }

  if (record.sidePanelsCollapsed && typeof record.sidePanelsCollapsed === 'object') {
    const panels = /** @type {Record<string, unknown>} */ (record.sidePanelsCollapsed);
    prefs.sidePanelsCollapsed = {
      models: Boolean(panels.models),
      sessions: Boolean(panels.sessions),
      resources: Boolean(panels.resources),
      outputs: Boolean(panels.outputs),
    };
  }

  prefs.reasoningEnabled = Boolean(record.reasoningEnabled);
  prefs.resubmitEnabled = Boolean(record.resubmitEnabled);
  prefs.language = isSupportedLanguage(record.language) ? record.language : null;
  return prefs;
}

/**
 * @returns {ReturnType<typeof defaultPreferences>}
 */
function getPreferences() {
  if (!cached) {
    throw new Error(t('errors.preferences.notLoaded'));
  }
  return cached;
}

/**
 * @returns {ReturnType<typeof defaultPreferences>}
 */
function snapshotPreferences() {
  const prefs = getPreferences();
  return {
    activeWorkspace: prefs.activeWorkspace,
    selectedModelId: prefs.selectedModelId,
    sidePanelsCollapsed: { ...prefs.sidePanelsCollapsed },
    reasoningEnabled: prefs.reasoningEnabled,
    resubmitEnabled: prefs.resubmitEnabled,
    language: prefs.language,
  };
}

/**
 * @returns {Promise<ReturnType<typeof defaultPreferences>>}
 */
async function loadPreferences() {
  try {
    const raw = await fs.readFile(preferencesFilePath(), 'utf8');
    cached = normalizePreferences(JSON.parse(raw));
  } catch {
    cached = defaultPreferences();
  }
  return cached;
}

/**
 * @param {Partial<ReturnType<typeof defaultPreferences>>} partial
 * @returns {Promise<ReturnType<typeof defaultPreferences>>}
 */
async function updatePreferences(partial) {
  if (!cached) {
    throw new Error(t('errors.preferences.notLoaded'));
  }
  const merged = {
    ...cached,
    ...partial,
  };
  if (partial && partial.sidePanelsCollapsed) {
    merged.sidePanelsCollapsed = {
      ...cached.sidePanelsCollapsed,
      ...partial.sidePanelsCollapsed,
    };
  }
  cached = normalizePreferences(merged);
  await fs.writeFile(preferencesFilePath(), `${JSON.stringify(cached, null, 2)}\n`, 'utf8');
  return cached;
}

/**
 * @param {unknown} partial
 * @returns {Partial<ReturnType<typeof defaultPreferences>>}
 */
function pickUiPreferenceUpdates(partial) {
  if (!partial || typeof partial !== 'object') {
    return {};
  }
  const record = /** @type {Record<string, unknown>} */ (partial);
  /** @type {Partial<ReturnType<typeof defaultPreferences>>} */
  const updates = {};
  if ('reasoningEnabled' in record) {
    updates.reasoningEnabled = Boolean(record.reasoningEnabled);
  }
  if ('resubmitEnabled' in record) {
    updates.resubmitEnabled = Boolean(record.resubmitEnabled);
  }
  if (record.sidePanelsCollapsed && typeof record.sidePanelsCollapsed === 'object') {
    const panels = /** @type {Record<string, unknown>} */ (record.sidePanelsCollapsed);
    updates.sidePanelsCollapsed = {
      models: Boolean(panels.models),
      sessions: Boolean(panels.sessions),
      resources: Boolean(panels.resources),
      outputs: Boolean(panels.outputs),
    };
  }
  if (isSupportedLanguage(record.language)) {
    updates.language = record.language;
  }
  return updates;
}

function registerPreferencesIpc() {
  ipcMain.handle('prefs:get', async () => {
    try {
      return ok({ preferences: snapshotPreferences() });
    } catch (err) {
      return fail(err, 'E_PREFS');
    }
  });

  ipcMain.handle('prefs:update', async (_event, payload) => {
    try {
      const previousLanguage = cached ? cached.language : null;
      const updates = pickUiPreferenceUpdates(payload);
      await updatePreferences(updates);
      if ('language' in updates && updates.language !== previousLanguage) {
        const { applyResolvedLanguage, reloadAllWindows } = require('./i18n');
        applyResolvedLanguage();
        reloadAllWindows();
      }
      return ok({ preferences: snapshotPreferences() });
    } catch (err) {
      return fail(err, 'E_PREFS');
    }
  });
}

module.exports = {
  loadPreferences,
  getPreferences,
  snapshotPreferences,
  updatePreferences,
  readPersistedLanguageFromDisk,
  registerPreferencesIpc,
};
