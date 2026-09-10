const path = require('path');
const fs = require('fs/promises');
const { pathToFileURL } = require('url');
const { ipcMain, BrowserWindow } = require('electron');
const { t } = require('../../i18n');
const state = require('../state');
const { getMediaKindFromFileName } = require('../mediaKinds');
const {
  APP_NAME,
  APP_ICON_PATH,
  resolvePanelFilePath,
  resolvePanelMediaPath,
} = require('../paths');
const { getWindowBackgroundColor } = require('./theme');

function getMarkdownEditorKey(panel, relativePath) {
  return `${panel}:${relativePath}`;
}

function assertMarkdownRelativePath(relativePath) {
  if (typeof relativePath !== 'string' || !relativePath.trim()) {
    throw new Error(t('errors.editors.filePathRequired'));
  }
  const lowerPath = relativePath.toLowerCase();
  if (!lowerPath.endsWith('.md') && !lowerPath.endsWith('.txt')) {
    throw new Error(t('errors.editors.onlyMarkdownOrText'));
  }
}

function getMarkdownContextFromEvent(event) {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win || !win.markdownContext) {
    throw new Error(t('errors.editors.markdownContextUnavailable'));
  }
  return win.markdownContext;
}

function handleMarkdownEditorClose(editorWindow, event) {
  if (editorWindow.markdownCloseConfirmed) {
    return;
  }

  event.preventDefault();
  editorWindow.webContents.send('markdown:attemptClose');
}

function openMarkdownEditorWindow(panel, relativePath) {
  assertMarkdownRelativePath(relativePath);
  const key = getMarkdownEditorKey(panel, relativePath);
  const existing = state.markdownEditorWindows.get(key);
  if (existing && !existing.isDestroyed()) {
    existing.focus();
    return;
  }

  const fileName = path.basename(relativePath);
  const editorWindow = new BrowserWindow({
    title: `${fileName} - ${APP_NAME}`,
    width: 960,
    height: 720,
    minWidth: 640,
    minHeight: 480,
    autoHideMenuBar: true,
    icon: APP_ICON_PATH,
    backgroundColor: getWindowBackgroundColor(),
    webPreferences: {
      preload: path.join(__dirname, '../../preload/preload-markdown.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  editorWindow.markdownContext = { panel, relativePath, fileName };
  editorWindow.loadFile(path.join(__dirname, '../../renderer', 'markdown-editor.html'));
  state.markdownEditorWindows.set(key, editorWindow);
  editorWindow.on('close', (event) => {
    void handleMarkdownEditorClose(editorWindow, event);
  });
  editorWindow.on('closed', () => {
    state.markdownEditorWindows.delete(key);
  });
}

function getMediaViewerKey(panel, relativePath) {
  return `${panel}:${relativePath}`;
}

function getMediaContextFromEvent(event) {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win || !win.mediaContext) {
    throw new Error(t('errors.editors.mediaContextUnavailable'));
  }
  return win.mediaContext;
}

async function openMediaViewerWindow(panel, relativePath) {
  const kind = getMediaKindFromFileName(path.basename(relativePath));
  if (!kind) {
    throw new Error(t('errors.editors.unsupportedMediaFileType'));
  }

  const key = getMediaViewerKey(panel, relativePath);
  const existing = state.mediaViewerWindows.get(key);
  if (existing && !existing.isDestroyed()) {
    existing.focus();
    return;
  }

  const absolutePath = resolvePanelMediaPath(panel, relativePath);
  const stats = await fs.stat(absolutePath).catch(() => null);
  if (!stats || !stats.isFile()) {
    throw new Error(t('errors.editors.fileDoesNotExist'));
  }

  const fileName = path.basename(relativePath);
  const mediaUrl = pathToFileURL(absolutePath).href;
  const windowSizes = {
    image: { width: 900, height: 700, minWidth: 320, minHeight: 240 },
    audio: { width: 480, height: 220, minWidth: 360, minHeight: 180 },
    video: { width: 960, height: 640, minWidth: 480, minHeight: 360 },
    pdf: { width: 900, height: 700, minWidth: 480, minHeight: 360 },
  };
  const size = windowSizes[kind];

  const viewerWindow = new BrowserWindow({
    title: `${fileName} - ${APP_NAME}`,
    width: size.width,
    height: size.height,
    minWidth: size.minWidth,
    minHeight: size.minHeight,
    autoHideMenuBar: true,
    icon: APP_ICON_PATH,
    backgroundColor: getWindowBackgroundColor(),
    webPreferences: {
      preload: path.join(__dirname, '../../preload/preload-media.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  viewerWindow.mediaContext = { panel, relativePath, fileName, kind, mediaUrl };
  viewerWindow.loadFile(path.join(__dirname, '../../renderer', 'media-viewer.html'));
  state.mediaViewerWindows.set(key, viewerWindow);
  viewerWindow.on('closed', () => {
    state.mediaViewerWindows.delete(key);
  });
}

function registerEditorsIpc() {
  ipcMain.handle('markdown:openEditor', async (_event, { panel, relativePath }) => {
    try {
      openMarkdownEditorWindow(panel, relativePath);
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.handle('markdown:getContext', async (event) => {
    try {
      const context = getMarkdownContextFromEvent(event);
      return { ok: true, context };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.handle('markdown:readFile', async (event) => {
    try {
      const { panel, relativePath } = getMarkdownContextFromEvent(event);
      const absolutePath = resolvePanelFilePath(panel, relativePath);
      const stats = await fs.stat(absolutePath).catch(() => null);
      if (!stats || !stats.isFile()) {
        throw new Error(t('errors.editors.fileDoesNotExist'));
      }
      const content = await fs.readFile(absolutePath, 'utf8');
      return { ok: true, content, fileName: path.basename(relativePath) };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.handle('markdown:writeFile', async (event, { content }) => {
    try {
      const { panel, relativePath } = getMarkdownContextFromEvent(event);
      const absolutePath = resolvePanelFilePath(panel, relativePath);
      const stats = await fs.stat(absolutePath).catch(() => null);
      if (!stats || !stats.isFile()) {
        throw new Error(t('errors.editors.fileDoesNotExist'));
      }
      await fs.writeFile(absolutePath, typeof content === 'string' ? content : '', 'utf8');
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.on('markdown:confirmClose', (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win || win.isDestroyed()) {
      return;
    }
    win.markdownCloseConfirmed = true;
    win.close();
  });

  ipcMain.handle('media:openViewer', async (_event, { panel, relativePath }) => {
    try {
      await openMediaViewerWindow(panel, relativePath);
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.handle('media:getContext', async (event) => {
    try {
      const context = getMediaContextFromEvent(event);
      return { ok: true, context };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });
}

module.exports = {
  registerEditorsIpc,
};
