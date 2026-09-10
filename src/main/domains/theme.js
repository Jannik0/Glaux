'use strict';

const { nativeTheme, BrowserWindow, ipcMain } = require('electron');
const { resolveTheme, DEFAULT_THEME } = require('../theme');

const WINDOW_BACKGROUND = Object.freeze({
  dark: '#0f1011',
  light: '#dddff0',
});

function appearanceFromNativeTheme() {
  return nativeTheme.shouldUseDarkColors ? 'dark' : 'light';
}

function getWindowBackgroundColor() {
  return WINDOW_BACKGROUND[appearanceFromNativeTheme()];
}

function syncWindowBackgrounds() {
  const color = getWindowBackgroundColor();
  for (const win of BrowserWindow.getAllWindows()) {
    if (win && !win.isDestroyed()) {
      win.setBackgroundColor(color);
    }
  }
}

/**
 * @param {unknown} persisted
 */
function applyNativeTheme(persisted) {
  nativeTheme.themeSource = resolveTheme(persisted);
  syncWindowBackgrounds();
}

function registerThemeIpc() {
  ipcMain.on('theme:get', (event) => {
    let theme = DEFAULT_THEME;
    try {
      const { getPreferences } = require('./preferences');
      const prefs = getPreferences();
      theme = resolveTheme(prefs && prefs.theme);
    } catch {
      theme = DEFAULT_THEME;
    }
    event.returnValue = { theme };
  });

  nativeTheme.on('updated', () => {
    syncWindowBackgrounds();
  });
}

module.exports = {
  applyNativeTheme,
  getWindowBackgroundColor,
  registerThemeIpc,
};
