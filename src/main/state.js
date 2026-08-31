/** @type {import('electron').BrowserWindow | null} */
let mainWindow = null;

/** @type {Map<string, import('electron').BrowserWindow>} */
const markdownEditorWindows = new Map();

/** @type {Map<string, import('electron').BrowserWindow>} */
const mediaViewerWindows = new Map();

/** Hub model id selected by the user; null when none. Mirrored in preferences.json. */
let selectedModelId = null;

/** Basename of the active session JSON file under sessionsRoot, or null for an unsaved session. */
let activeSessionFilename = null;

/** Name of the active workspace folder under Workspaces/, or null before boot. */
let activeWorkspaceName = null;

let engineBootstrapped = false;

/** Resolves when an in-flight startup model load finishes (success or failure). */
let engineBootstrapLoadPromise = null;

/** @type {{ modelId: string, promise: Promise<unknown> } | null} */
let activePanelModelDownload = null;

const activeStreamRequests = new Map();

let appQuitCleanupStarted = false;

function getMainWindow() {
  return mainWindow;
}

function setMainWindow(win) {
  mainWindow = win;
}

module.exports = {
  getMainWindow,
  setMainWindow,
  markdownEditorWindows,
  mediaViewerWindows,
  get selectedModelId() {
    return selectedModelId;
  },
  set selectedModelId(value) {
    selectedModelId = value;
  },
  get activeSessionFilename() {
    return activeSessionFilename;
  },
  set activeSessionFilename(value) {
    activeSessionFilename = value;
  },
  get activeWorkspaceName() {
    return activeWorkspaceName;
  },
  set activeWorkspaceName(value) {
    activeWorkspaceName = value;
  },
  get engineBootstrapped() {
    return engineBootstrapped;
  },
  set engineBootstrapped(value) {
    engineBootstrapped = value;
  },
  get engineBootstrapLoadPromise() {
    return engineBootstrapLoadPromise;
  },
  set engineBootstrapLoadPromise(value) {
    engineBootstrapLoadPromise = value;
  },
  get activePanelModelDownload() {
    return activePanelModelDownload;
  },
  set activePanelModelDownload(value) {
    activePanelModelDownload = value;
  },
  activeStreamRequests,
  get appQuitCleanupStarted() {
    return appQuitCleanupStarted;
  },
  set appQuitCleanupStarted(value) {
    appQuitCleanupStarted = value;
  },
};
