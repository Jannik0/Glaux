'use strict';

const { app, ipcMain, session } = require('electron');
const {
  getI18nPayload,
  setCurrentLanguage,
  resolveLanguage,
  chromiumLocaleTag,
  primaryLanguageTag,
} = require('../../i18n');
const { getPreferences, readPersistedLanguageFromDisk } = require('./preferences');
const state = require('../state');

function getPreferredOsLocale() {
  try {
    if (typeof app.getPreferredSystemLanguages === 'function') {
      const preferred = app.getPreferredSystemLanguages();
      if (Array.isArray(preferred) && preferred[0]) {
        return preferred[0];
      }
    }
  } catch {
    /* ignore */
  }
  return process.env.LC_ALL || process.env.LANG || '';
}

function getOsLocale() {
  try {
    if (app.isReady() && typeof app.getLocale === 'function') {
      return app.getLocale() || getPreferredOsLocale();
    }
  } catch {
    /* ignore */
  }
  return getPreferredOsLocale();
}

function getPersistedLanguage() {
  try {
    const prefs = getPreferences();
    return prefs && prefs.language ? prefs.language : null;
  } catch {
    return readPersistedLanguageFromDisk();
  }
}

/**
 * Chromium reads `--lang` only before `ready`. Call immediately after app paths
 * are initialized so native dialogs and spellcheck match Glaux's language.
 */
function applyChromiumLangSwitch() {
  const language = resolveLanguage(getPersistedLanguage(), getPreferredOsLocale());
  app.commandLine.appendSwitch('lang', chromiumLocaleTag(language, getPreferredOsLocale()));
}

function applySpellCheckerLanguage(language) {
  if (!app.isReady()) {
    return;
  }
  const desired = chromiumLocaleTag(language, getPreferredOsLocale());
  let sess;
  try {
    sess = session.defaultSession;
  } catch {
    return;
  }
  if (!sess || typeof sess.setSpellCheckerLanguages !== 'function') {
    return;
  }
  const available = Array.isArray(sess.availableSpellCheckerLanguages)
    ? sess.availableSpellCheckerLanguages
    : [];
  const pick =
    available.find((code) => code.toLowerCase() === desired.toLowerCase()) ||
    available.find((code) => primaryLanguageTag(code) === language) ||
    (available.includes(language) ? language : null);
  if (!pick) {
    return;
  }
  try {
    sess.setSpellCheckerLanguages([pick]);
  } catch {
    /* pack missing after locale prune */
  }
}

function applyResolvedLanguage() {
  const persisted = getPersistedLanguage();
  const language = resolveLanguage(persisted, getOsLocale());
  setCurrentLanguage(language);
  applySpellCheckerLanguage(language);
  return language;
}

function reloadAllWindows() {
  const windows = [];
  const main = state.getMainWindow();
  if (main && !main.isDestroyed()) {
    windows.push(main);
  }
  for (const win of state.markdownEditorWindows.values()) {
    if (win && !win.isDestroyed()) {
      windows.push(win);
    }
  }
  for (const win of state.mediaViewerWindows.values()) {
    if (win && !win.isDestroyed()) {
      windows.push(win);
    }
  }
  for (const win of windows) {
    win.reload();
  }
}

function registerI18nIpc() {
  ipcMain.on('i18n:get', (event) => {
    applyResolvedLanguage();
    try {
      event.returnValue = JSON.parse(
        JSON.stringify(getI18nPayload(getPersistedLanguage(), getOsLocale())),
      );
    } catch {
      event.returnValue = {
        language: 'en',
        persistedLanguage: null,
        available: [],
        catalog: {},
        fallbackCatalog: {},
      };
    }
  });
}

module.exports = {
  applyResolvedLanguage,
  applyChromiumLangSwitch,
  reloadAllWindows,
  registerI18nIpc,
  getOsLocale,
  getPersistedLanguage,
};
