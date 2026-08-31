'use strict';

const fs = require('fs/promises');
const { ipcMain } = require('electron');
const engineManager = require('../../../engines/engineManager');
const state = require('../state');
const { ok, fail } = require('../ipc/result');
const {
  assertValidEntryName,
  getModelsRoot,
  getResourcesRoot,
  getOutputsRoot,
  listWorkspaceNames,
  resolveWorkspaceDir,
  setActiveWorkspacePaths,
  ensureWorkspaceDirectories,
  ensureWorkspacesRootDirectory,
  moveEntryToRecycleBin,
  pathExists,
} = require('../paths');
const { getPreferences, updatePreferences } = require('./preferences');
const { t } = require('../../i18n');

const DEFAULT_WORKSPACE_NAME = 'Default';
const NEW_WORKSPACE_BASE = 'New Workspace';

/**
 * @returns {string | null}
 */
function readPersistedActiveWorkspaceName() {
  const name = getPreferences().activeWorkspace;
  return typeof name === 'string' && name.trim() ? name.trim() : null;
}

/**
 * @param {string} name
 * @returns {Promise<void>}
 */
async function persistActiveWorkspaceName(name) {
  await updatePreferences({ activeWorkspace: name });
}

/**
 * @param {string[]} names
 * @param {string} candidate
 * @returns {boolean}
 */
function workspaceNameTaken(names, candidate) {
  const lower = candidate.toLowerCase();
  return names.some((n) => n.toLowerCase() === lower);
}

/**
 * @param {string[]} existing
 * @returns {string}
 */
function allocateNewWorkspaceName(existing) {
  if (!workspaceNameTaken(existing, NEW_WORKSPACE_BASE)) {
    return NEW_WORKSPACE_BASE;
  }
  let i = 2;
  while (workspaceNameTaken(existing, `${NEW_WORKSPACE_BASE} ${i}`)) {
    i += 1;
  }
  return `${NEW_WORKSPACE_BASE} ${i}`;
}

function closeWorkspaceDependentWindows() {
  for (const win of state.markdownEditorWindows.values()) {
    if (win && !win.isDestroyed()) {
      win.markdownCloseConfirmed = true;
      win.close();
    }
  }
  for (const win of state.mediaViewerWindows.values()) {
    if (win && !win.isDestroyed()) {
      win.close();
    }
  }
}

/**
 * Keep engine attachment roots in sync with the active workspace.
 * Safe to call before bootstrap; bootstrap also reads current path getters.
 * @returns {Promise<void>}
 */
async function reconfigureEnginePathsIfReady() {
  const resourcesRoot = getResourcesRoot();
  const outputsRoot = getOutputsRoot();
  if (!resourcesRoot || !outputsRoot) {
    return;
  }
  try {
    await engineManager.configurePaths({
      resourcesRoot,
      outputsRoot,
      modelsCacheDir: getModelsRoot(),
    });
  } catch (err) {
    console.error('Failed to reconfigure engine paths for workspace:', err);
  }
}

/**
 * Activate a workspace: update roots, clear session, close editors, reconfigure engines.
 * @param {string} name
 * @returns {Promise<string>}
 */
async function activateWorkspace(name) {
  const safeName = setActiveWorkspacePaths(name);
  await ensureWorkspaceDirectories(safeName);
  state.activeWorkspaceName = safeName;
  state.activeSessionFilename = null;
  closeWorkspaceDependentWindows();
  await persistActiveWorkspaceName(safeName);
  await reconfigureEnginePathsIfReady();
  return safeName;
}

/**
 * @returns {Promise<{ workspaces: string[], active: string }>}
 */
async function getWorkspaceSnapshot() {
  const workspaces = await listWorkspaceNames();
  return {
    workspaces,
    active: state.activeWorkspaceName,
  };
}

/**
 * Boot-time: ensure at least Default exists and activate persisted or fallback.
 * @returns {Promise<string>}
 */
async function ensureWorkspacesReady() {
  await ensureWorkspacesRootDirectory();
  let names = await listWorkspaceNames();
  if (names.length === 0) {
    await ensureWorkspaceDirectories(DEFAULT_WORKSPACE_NAME);
    names = [DEFAULT_WORKSPACE_NAME];
  }

  const persisted = readPersistedActiveWorkspaceName();
  let target = null;
  if (persisted && workspaceNameTaken(names, persisted)) {
    target = names.find((n) => n.toLowerCase() === persisted.toLowerCase()) || persisted;
  } else {
    target = names[0];
  }

  return activateWorkspace(target);
}

async function listWorkspacesHandler() {
  const snapshot = await getWorkspaceSnapshot();
  return ok(snapshot);
}

async function getActiveWorkspaceHandler() {
  return ok({ name: state.activeWorkspaceName });
}

async function setActiveWorkspaceHandler(payload) {
  const name = assertValidEntryName(payload && payload.name);
  const names = await listWorkspaceNames();
  if (!workspaceNameTaken(names, name)) {
    throw new Error(t('errors.workspaces.doesNotExist'));
  }
  const existing = names.find((n) => n.toLowerCase() === name.toLowerCase()) || name;
  if (state.activeWorkspaceName && existing.toLowerCase() === state.activeWorkspaceName.toLowerCase()) {
    return ok(await getWorkspaceSnapshot());
  }
  await activateWorkspace(existing);
  return ok(await getWorkspaceSnapshot());
}

async function createWorkspaceHandler() {
  const names = await listWorkspaceNames();
  const newName = allocateNewWorkspaceName(names);
  await ensureWorkspaceDirectories(newName);
  await activateWorkspace(newName);
  const snapshot = await getWorkspaceSnapshot();
  return ok({ name: newName, ...snapshot });
}

async function renameWorkspaceHandler(payload) {
  const oldName = assertValidEntryName(payload && payload.oldName);
  const newName = assertValidEntryName(payload && payload.newName);
  const names = await listWorkspaceNames();
  if (!workspaceNameTaken(names, oldName)) {
    throw new Error(t('errors.workspaces.doesNotExist'));
  }
  const existingOld = names.find((n) => n.toLowerCase() === oldName.toLowerCase()) || oldName;
  if (existingOld.toLowerCase() === newName.toLowerCase()) {
    if (existingOld === newName) {
      return ok({ name: existingOld, ...(await getWorkspaceSnapshot()) });
    }
    // Same name ignoring case but different spelling — still rename on case-sensitive FS.
  } else if (workspaceNameTaken(names, newName)) {
    throw new Error(t('errors.workspaces.nameAlreadyExists'));
  }

  const { absPath: oldPath } = resolveWorkspaceDir(existingOld);
  const { absPath: newPath } = resolveWorkspaceDir(newName);
  if (!(await pathExists(oldPath))) {
    throw new Error(t('errors.workspaces.doesNotExist'));
  }
  if (oldPath !== newPath && (await pathExists(newPath))) {
    throw new Error(t('errors.workspaces.nameAlreadyExists'));
  }

  await fs.rename(oldPath, newPath);

  if (
    state.activeWorkspaceName &&
    state.activeWorkspaceName.toLowerCase() === existingOld.toLowerCase()
  ) {
    await activateWorkspace(newName);
  }

  return ok({ name: newName, ...(await getWorkspaceSnapshot()) });
}

async function trashWorkspaceHandler(payload) {
  const name = assertValidEntryName(payload && payload.name);
  const names = await listWorkspaceNames();
  if (names.length <= 1) {
    throw new Error(t('errors.workspaces.cannotDeleteOnly'));
  }
  if (!workspaceNameTaken(names, name)) {
    throw new Error(t('errors.workspaces.doesNotExist'));
  }
  const existing = names.find((n) => n.toLowerCase() === name.toLowerCase()) || name;
  const wasActive =
    Boolean(state.activeWorkspaceName) &&
    state.activeWorkspaceName.toLowerCase() === existing.toLowerCase();

  const { absPath } = resolveWorkspaceDir(existing);
  await moveEntryToRecycleBin(absPath);

  let nextActive = state.activeWorkspaceName;
  if (wasActive) {
    const remaining = (await listWorkspaceNames()).filter(
      (n) => n.toLowerCase() !== existing.toLowerCase()
    );
    if (remaining.length === 0) {
      throw new Error(t('errors.workspaces.cannotDeleteOnly'));
    }
    nextActive = await activateWorkspace(remaining[0]);
  }

  return ok({
    name: existing,
    wasActive,
    active: nextActive,
    ...(await getWorkspaceSnapshot()),
  });
}

function registerWorkspacesIpc() {
  ipcMain.handle('workspaces:list', async () => {
    try {
      return await listWorkspacesHandler();
    } catch (err) {
      return fail(err);
    }
  });

  ipcMain.handle('workspaces:getActive', async () => {
    try {
      return await getActiveWorkspaceHandler();
    } catch (err) {
      return fail(err);
    }
  });

  ipcMain.handle('workspaces:setActive', async (_event, payload) => {
    try {
      return await setActiveWorkspaceHandler(payload);
    } catch (err) {
      return fail(err);
    }
  });

  ipcMain.handle('workspaces:create', async () => {
    try {
      return await createWorkspaceHandler();
    } catch (err) {
      return fail(err);
    }
  });

  ipcMain.handle('workspaces:rename', async (_event, payload) => {
    try {
      return await renameWorkspaceHandler(payload);
    } catch (err) {
      return fail(err);
    }
  });

  ipcMain.handle('workspaces:trash', async (_event, payload) => {
    try {
      return await trashWorkspaceHandler(payload);
    } catch (err) {
      return fail(err);
    }
  });
}

module.exports = {
  registerWorkspacesIpc,
  ensureWorkspacesReady,
  activateWorkspace,
  getWorkspaceSnapshot,
};
