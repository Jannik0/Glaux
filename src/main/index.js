const path = require('path');
const { app, BrowserWindow, shell } = require('electron');
const engineManager = require('../../engines/engineManager');
const { APP_NAME, APP_ICON_PATH, initAppPaths, ensureModelsDirectory } = require('./paths');
const state = require('./state');
const { registerIpc } = require('./ipc/register');
const { ensureWorkspacesReady } = require('./domains/workspaces');
const { loadPreferences, getPreferences } = require('./domains/preferences');
const { applyResolvedLanguage, applyChromiumLangSwitch } = require('./domains/i18n');
const { getWindowBackgroundColor } = require('./domains/theme');
const { isForceCpu } = require('../../engines/common/gpuRuntime');

if (process.platform === 'win32') {
  app.setAppUserModelId(APP_NAME);
}

initAppPaths();
applyChromiumLangSwitch();
registerIpc();

if (isForceCpu()) {
  process.stderr.write(
    'GLAUX_FORCE_CPU is set; Hugging Face, llama.cpp, and transcribe.cpp will pin CPU.\n'
  );
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    title: APP_NAME,
    width: 1000,
    height: 700,
    autoHideMenuBar: true,
    icon: APP_ICON_PATH,
    backgroundColor: getWindowBackgroundColor(),
    webPreferences: {
      preload: path.join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.maximize();
  mainWindow.loadFile(path.join(__dirname, '../renderer', 'index.html'));

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http:') || url.startsWith('https:')) {
      void shell.openExternal(url);
    }
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    state.setMainWindow(null);
  });

  state.setMainWindow(mainWindow);
}

app.whenReady().then(async () => {
  await ensureModelsDirectory();
  await loadPreferences();
  applyResolvedLanguage();
  state.selectedModelId = getPreferences().selectedModelId;
  await ensureWorkspacesReady();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', (event) => {
  if (state.appQuitCleanupStarted) {
    return;
  }
  state.appQuitCleanupStarted = true;
  event.preventDefault();
  engineManager
    .shutdown()
    .catch((err) => {
      console.error('Engine shutdown failed:', err);
    })
    .finally(() => {
      app.exit(0);
    });
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    if (state.appQuitCleanupStarted) {
      return;
    }
    state.appQuitCleanupStarted = true;
    engineManager
      .shutdown()
      .catch(() => {})
      .finally(() => {
        app.exit(0);
      });
  });
}
