const path = require('path');
const fs = require('fs/promises');
const { app, shell } = require('electron');
const { getMediaKindFromFileName } = require('./mediaKinds');
const { t } = require('../i18n');
const {
  assertValidEntryName,
  assertValidOsFolderName,
  toPosixRelative,
  resolveWorkspacePath,
  isSubPath,
} = require('./pathSandbox');

const APP_NAME = 'Glaux';
const APP_ICON_PATH = (() => {
  const assetsDir = path.join(__dirname, '../../assets');
  if (process.platform === 'darwin') {
    return path.join(assetsDir, 'GlauxAI_Logo.icns');
  }
  if (process.platform === 'linux') {
    return path.join(assetsDir, 'GlauxAI_Logo.png');
  }
  return path.join(assetsDir, 'GlauxAI_Logo.ico');
})();
/** PNG works reliably with nativeImage.resize on every platform (tray / overlays). */
const APP_ICON_PNG_PATH = path.join(__dirname, '../../assets', 'GlauxAI_Logo.png');

let workspacesRoot;
let resourcesRoot;
let outputsRoot;
let modelsRoot;
let sessionsRoot;

function initAppPaths() {
  app.setName(APP_NAME);
  process.title = APP_NAME;
  app.setPath('userData', path.join(app.getPath('appData'), APP_NAME));
  workspacesRoot = path.join(app.getPath('userData'), 'Workspaces');
  modelsRoot = path.join(app.getPath('userData'), 'Models');
  resourcesRoot = null;
  outputsRoot = null;
  sessionsRoot = null;
}

function getUserDataRoot() {
  return app.getPath('userData');
}

function getWorkspacesRoot() {
  return workspacesRoot;
}

function getResourcesRoot() {
  return resourcesRoot;
}

function getOutputsRoot() {
  return outputsRoot;
}

function getModelsRoot() {
  return modelsRoot;
}

function getSessionsRoot() {
  return sessionsRoot;
}

/**
 * @param {unknown} raw
 * @returns {string}
 */
function assertValidHfRepoId(raw) {
  const s = typeof raw === 'string' ? raw.trim() : '';
  if (!s || s.length > 512) {
    throw new Error(t('errors.paths.enterValidModelId'));
  }
  if (!/^[\w.-]+\/[\w.-]+$/.test(s)) {
    throw new Error(t('errors.paths.invalidModelIdFormat'));
  }
  return s;
}

function assertValidSessionFilename(name) {
  const trimmed = assertValidEntryName(name);
  if (!trimmed.toLowerCase().endsWith('.json')) {
    return `${trimmed}.json`;
  }
  return trimmed;
}

/**
 * Resolve a workspace directory name under Workspaces/, confined to that root.
 * @param {unknown} name
 * @returns {{ name: string, absPath: string }}
 */
function resolveWorkspaceDir(name) {
  const safeName = assertValidEntryName(name);
  const absPath = path.resolve(workspacesRoot, safeName);
  const relativeToRoot = path.relative(workspacesRoot, absPath);
  if (
    relativeToRoot === '..' ||
    relativeToRoot.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativeToRoot) ||
    relativeToRoot.includes(path.sep)
  ) {
    throw new Error(t('errors.paths.invalidWorkspacePath'));
  }
  return { name: safeName, absPath };
}

/**
 * Point Sessions/Resources/Outputs roots at Workspaces/<name>/.
 * @param {unknown} name
 * @returns {string} validated workspace name
 */
function setActiveWorkspacePaths(name) {
  const { name: safeName, absPath } = resolveWorkspaceDir(name);
  resourcesRoot = path.join(absPath, 'Resources');
  outputsRoot = path.join(absPath, 'Outputs');
  sessionsRoot = path.join(absPath, 'Sessions');
  return safeName;
}

/**
 * @returns {Promise<string[]>}
 */
async function listWorkspaceNames() {
  await fs.mkdir(workspacesRoot, { recursive: true });
  const entries = await fs.readdir(workspacesRoot, { withFileTypes: true });
  return entries
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
}

function resolveResourcesPath(relativePath = '') {
  if (!resourcesRoot) {
    throw new Error(t('errors.paths.noActiveWorkspace'));
  }
  return resolveWorkspacePath(
    resourcesRoot,
    relativePath,
    t('errors.paths.pathEscapesResources')
  );
}

function resolveOutputsPath(relativePath = '') {
  if (!outputsRoot) {
    throw new Error(t('errors.paths.noActiveWorkspace'));
  }
  return resolveWorkspacePath(
    outputsRoot,
    relativePath,
    t('errors.paths.pathEscapesOutputs')
  );
}

function resolveSessionsPath(filename) {
  if (!sessionsRoot) {
    throw new Error(t('errors.paths.noActiveWorkspace'));
  }
  const safeName = assertValidSessionFilename(filename);
  const resolvedPath = path.resolve(sessionsRoot, safeName);
  const relativeToRoot = path.relative(sessionsRoot, resolvedPath);
  if (
    relativeToRoot === '..' ||
    relativeToRoot.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativeToRoot) ||
    relativeToRoot.includes(path.sep)
  ) {
    throw new Error(t('errors.paths.invalidSessionFilePath'));
  }
  return resolvedPath;
}

/**
 * Absolute model cache directory for a Hub-style id (`org/model`), confined under modelsRoot.
 *
 * @param {string} modelId
 * @returns {Promise<string>}
 */
async function resolveModelsCacheAbsoluteDir(modelId) {
  const id = assertValidHfRepoId(modelId);
  const abs = path.resolve(modelsRoot, ...id.split('/'));
  const rel = path.relative(modelsRoot, abs);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) {
    throw new Error(t('errors.paths.invalidModelFolderPath'));
  }
  let st;
  try {
    st = await fs.stat(abs);
  } catch (_e) {
    throw new Error(t('errors.paths.modelFolderNotFound'));
  }
  if (!st.isDirectory()) {
    throw new Error(t('errors.paths.modelPathNotDirectory'));
  }
  return abs;
}

/**
 * @param {string} modelId
 * @returns {string}
 */
function modelCacheFolderAbsPath(modelId) {
  const abs = path.resolve(modelsRoot, ...modelId.split('/'));
  const rel = path.relative(modelsRoot, abs);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) {
    throw new Error(t('errors.paths.invalidModelFolderPath'));
  }
  return abs;
}

async function ensureWorkspaceDirectories(name) {
  const { absPath } = resolveWorkspaceDir(name);
  await fs.mkdir(path.join(absPath, 'Resources'), { recursive: true });
  await fs.mkdir(path.join(absPath, 'Outputs'), { recursive: true });
  await fs.mkdir(path.join(absPath, 'Sessions'), { recursive: true });
}

async function ensureResourcesDirectory() {
  if (!resourcesRoot) {
    throw new Error(t('errors.paths.noActiveWorkspace'));
  }
  await fs.mkdir(resourcesRoot, { recursive: true });
}

async function ensureOutputsDirectory() {
  if (!outputsRoot) {
    throw new Error(t('errors.paths.noActiveWorkspace'));
  }
  await fs.mkdir(outputsRoot, { recursive: true });
}

async function ensureModelsDirectory() {
  await fs.mkdir(modelsRoot, { recursive: true });
}

async function ensureSessionsDirectory() {
  if (!sessionsRoot) {
    throw new Error(t('errors.paths.noActiveWorkspace'));
  }
  await fs.mkdir(sessionsRoot, { recursive: true });
}

async function ensureWorkspacesRootDirectory() {
  await fs.mkdir(workspacesRoot, { recursive: true });
}

async function pathExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch (_err) {
    return false;
  }
}

async function moveEntryToRecycleBin(entryPath) {
  const targetStats = await fs.stat(entryPath).catch(() => null);
  if (!targetStats) {
    throw new Error(t('errors.paths.entryDoesNotExist'));
  }

  await shell.trashItem(entryPath);
}

function resolvePanelFilePath(panel, relativePath) {
  if (typeof relativePath !== 'string' || !relativePath.trim()) {
    throw new Error(t('errors.paths.filePathRequired'));
  }
  const lowerPath = relativePath.toLowerCase();
  if (!lowerPath.endsWith('.md') && !lowerPath.endsWith('.txt')) {
    throw new Error(t('errors.paths.onlyMarkdownOrText'));
  }
  if (panel === 'resources') {
    return resolveResourcesPath(relativePath);
  }
  if (panel === 'outputs') {
    return resolveOutputsPath(relativePath);
  }
  throw new Error(t('errors.paths.unknownFilePanel'));
}

function resolvePanelMediaPath(panel, relativePath) {
  if (typeof relativePath !== 'string' || !relativePath.trim()) {
    throw new Error(t('errors.paths.filePathRequired'));
  }
  if (!getMediaKindFromFileName(path.basename(relativePath))) {
    throw new Error(t('errors.paths.unsupportedMediaFileType'));
  }
  if (panel === 'resources') {
    return resolveResourcesPath(relativePath);
  }
  if (panel === 'outputs') {
    return resolveOutputsPath(relativePath);
  }
  throw new Error(t('errors.paths.unknownFilePanel'));
}

module.exports = {
  APP_NAME,
  APP_ICON_PATH,
  APP_ICON_PNG_PATH,
  initAppPaths,
  getUserDataRoot,
  getWorkspacesRoot,
  getResourcesRoot,
  getOutputsRoot,
  getModelsRoot,
  getSessionsRoot,
  assertValidEntryName,
  assertValidOsFolderName,
  assertValidHfRepoId,
  assertValidSessionFilename,
  toPosixRelative,
  resolveWorkspaceDir,
  setActiveWorkspacePaths,
  listWorkspaceNames,
  resolveResourcesPath,
  resolveOutputsPath,
  resolveSessionsPath,
  resolveModelsCacheAbsoluteDir,
  modelCacheFolderAbsPath,
  ensureWorkspaceDirectories,
  ensureResourcesDirectory,
  ensureOutputsDirectory,
  ensureModelsDirectory,
  ensureSessionsDirectory,
  ensureWorkspacesRootDirectory,
  pathExists,
  moveEntryToRecycleBin,
  isSubPath,
  resolvePanelFilePath,
  resolvePanelMediaPath,
};
