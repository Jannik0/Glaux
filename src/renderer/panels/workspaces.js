// Header workspace selector: switch, create, rename, and delete workspaces.

function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

const workspaceSelectEl = document.getElementById('workspace-select');
const workspaceCreateEl = document.getElementById('workspace-create');
const workspaceRenameEl = document.getElementById('workspace-rename');
const workspaceTrashEl = document.getElementById('workspace-trash');

/** @type {string[]} */
let workspaceNames = [];
/** @type {string | null} */
let activeWorkspaceName = null;
let workspaceSwitchInFlight = false;

/**
 * @param {string} value
 * @param {string[]} existingNames
 * @param {string | null} [allowName] name allowed even if present (current name on rename)
 * @returns {string | null} error message, or null if valid
 */
function validateWorkspaceNameInput(value, existingNames, allowName = null) {
  try {
    const safe = window.Glaux.TreePanel.validateEntryName(value);
    const lower = safe.toLowerCase();
    const allowLower = typeof allowName === 'string' ? allowName.toLowerCase() : null;
    if (
      existingNames.some((n) => n.toLowerCase() === lower) &&
      lower !== allowLower
    ) {
      return t('panels.workspaces.nameAlreadyExists');
    }
    return null;
  } catch (err) {
    return err && err.message ? err.message : String(err);
  }
}

function syncWorkspaceControlsDisabled() {
  const dis = engineBusy || engineLoading || Boolean(activeStream) || workspaceSwitchInFlight;
  if (workspaceSelectEl instanceof HTMLSelectElement) {
    workspaceSelectEl.disabled = dis;
  }
  if (workspaceCreateEl instanceof HTMLButtonElement) {
    workspaceCreateEl.disabled = dis;
  }
  if (workspaceRenameEl instanceof HTMLButtonElement) {
    workspaceRenameEl.disabled = dis;
  }
  if (workspaceTrashEl instanceof HTMLButtonElement) {
    workspaceTrashEl.disabled = dis || workspaceNames.length <= 1;
  }
}

/**
 * @param {{ workspaces?: string[], active?: string | null }} snapshot
 */
function applyWorkspaceSnapshot(snapshot) {
  workspaceNames = Array.isArray(snapshot.workspaces) ? snapshot.workspaces.slice() : [];
  activeWorkspaceName =
    typeof snapshot.active === 'string' && snapshot.active ? snapshot.active : null;
  renderWorkspaceSelect();
}

function renderWorkspaceSelect() {
  if (!(workspaceSelectEl instanceof HTMLSelectElement)) {
    return;
  }
  const previous = workspaceSelectEl.value;
  workspaceSelectEl.innerHTML = '';
  for (const name of workspaceNames) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    workspaceSelectEl.appendChild(option);
  }
  if (activeWorkspaceName && workspaceNames.includes(activeWorkspaceName)) {
    workspaceSelectEl.value = activeWorkspaceName;
  } else if (workspaceNames.length > 0) {
    workspaceSelectEl.value = workspaceNames[0];
    activeWorkspaceName = workspaceNames[0];
  } else if (previous) {
    workspaceSelectEl.value = previous;
  }
  syncWorkspaceControlsDisabled();
}

/**
 * Refresh sessions / resources / outputs / chat for the newly active workspace.
 * Does not use startNewSession()'s busy/stream guards — main already cleared the
 * active session when switching workspaces.
 */
async function refreshPanelsForWorkspaceChange() {
  if (window.api && window.api.sessions && typeof window.api.sessions.new === 'function') {
    try {
      await window.api.sessions.new();
    } catch (err) {
      console.error('Failed to reset session after workspace change:', err);
    }
  }
  activeSessionName = null;
  if (typeof clearChatMessages === 'function') {
    clearChatMessages();
  }
  if (typeof clearChatAttachments === 'function') {
    clearChatAttachments();
  }
  if (typeof inputEl !== 'undefined' && inputEl) {
    inputEl.value = '';
    if (typeof updateMessageInputHighlight === 'function') {
      updateMessageInputHighlight();
    }
  }
  if (typeof updateContextUsageIndicator === 'function') {
    void updateContextUsageIndicator({ refresh: true });
  }
  if (typeof sessionsEditingName !== 'undefined') {
    sessionsEditingName = null;
  }
  if (typeof refreshSessionsList === 'function') {
    await refreshSessionsList();
  }
  if (window.Glaux && window.Glaux.Resources && window.Glaux.Resources.panel) {
    await window.Glaux.Resources.panel.refreshTree();
  }
  if (window.Glaux && window.Glaux.Outputs && window.Glaux.Outputs.panel) {
    await window.Glaux.Outputs.panel.refreshTree();
  }
}

/**
 * @param {{ workspaces?: string[], active?: string | null }} snapshot
 * @param {{ refreshPanels?: boolean }} [opts]
 */
async function applyWorkspaceChange(snapshot, opts = {}) {
  applyWorkspaceSnapshot(snapshot);
  if (opts.refreshPanels !== false) {
    await refreshPanelsForWorkspaceChange();
  }
}

async function loadWorkspaces() {
  if (!(window.api && window.api.workspaces && typeof window.api.workspaces.list === 'function')) {
    return;
  }
  const snapshot = await window.api.workspaces.list();
  applyWorkspaceSnapshot(snapshot);
}

async function handleWorkspaceSelectChange() {
  if (!(workspaceSelectEl instanceof HTMLSelectElement)) {
    return;
  }
  const next = workspaceSelectEl.value;
  if (
    !next ||
    next === activeWorkspaceName ||
    workspaceSwitchInFlight ||
    engineBusy ||
    engineLoading ||
    activeStream
  ) {
    renderWorkspaceSelect();
    return;
  }
  if (!(window.api && window.api.workspaces)) {
    return;
  }
  workspaceSwitchInFlight = true;
  syncWorkspaceControlsDisabled();
  try {
    const snapshot = await window.api.workspaces.setActive(next);
    await applyWorkspaceChange(snapshot);
  } catch (err) {
    console.error('Failed to switch workspace:', err);
    renderWorkspaceSelect();
  } finally {
    workspaceSwitchInFlight = false;
    syncWorkspaceControlsDisabled();
  }
}

/**
 * @param {string} currentName
 * @returns {Promise<string | null>}
 */
async function promptRenameWorkspace(currentName) {
  const nextName = await window.Glaux.Dialogs.showNameDialog({
    title: t('panels.workspaces.renameTitle'),
    message: t('panels.workspaces.renameMessage'),
    initialValue: currentName,
    confirmLabel: t('panels.workspaces.renameConfirm'),
    existingNames: workspaceNames,
    validate: (value, existing) => validateWorkspaceNameInput(value, existing, currentName),
  });
  return nextName;
}

async function handleWorkspaceCreate() {
  if (
    workspaceSwitchInFlight ||
    engineBusy ||
    engineLoading ||
    activeStream ||
    !(window.api && window.api.workspaces)
  ) {
    return;
  }
  workspaceSwitchInFlight = true;
  syncWorkspaceControlsDisabled();
  try {
    const created = await window.api.workspaces.create();
    await applyWorkspaceChange(created);
    const renameTo = await promptRenameWorkspace(created.name || created.active);
    if (!renameTo || !created.name) {
      return;
    }
    if (renameTo === created.name) {
      return;
    }
    const renamed = await window.api.workspaces.rename(created.name, renameTo);
    await applyWorkspaceChange(renamed);
  } catch (err) {
    console.error('Failed to create workspace:', err);
    try {
      await loadWorkspaces();
    } catch (_e) {
      // ignore
    }
  } finally {
    workspaceSwitchInFlight = false;
    syncWorkspaceControlsDisabled();
  }
}

async function handleWorkspaceRename() {
  if (
    workspaceSwitchInFlight ||
    engineBusy ||
    engineLoading ||
    activeStream ||
    !activeWorkspaceName ||
    !(window.api && window.api.workspaces)
  ) {
    return;
  }
  const current = activeWorkspaceName;
  const renameTo = await promptRenameWorkspace(current);
  if (!renameTo || renameTo === current) {
    return;
  }
  workspaceSwitchInFlight = true;
  syncWorkspaceControlsDisabled();
  try {
    const renamed = await window.api.workspaces.rename(current, renameTo);
    await applyWorkspaceChange(renamed);
  } catch (err) {
    console.error('Failed to rename workspace:', err);
    try {
      await loadWorkspaces();
    } catch (_e) {
      // ignore
    }
  } finally {
    workspaceSwitchInFlight = false;
    syncWorkspaceControlsDisabled();
  }
}

async function handleWorkspaceTrash() {
  if (
    workspaceSwitchInFlight ||
    engineBusy ||
    engineLoading ||
    activeStream ||
    !activeWorkspaceName ||
    workspaceNames.length <= 1 ||
    !(window.api && window.api.workspaces)
  ) {
    return;
  }
  const name = activeWorkspaceName;
  const confirmed = await window.Glaux.Dialogs.showConfirmDialog({
    title: t('panels.workspaces.deleteConfirmTitle'),
    message: t('panels.workspaces.deleteConfirmMessage', { name }),
    confirmLabel: t('dialogs.delete'),
  });
  if (!confirmed) {
    return;
  }
  workspaceSwitchInFlight = true;
  syncWorkspaceControlsDisabled();
  try {
    const result = await window.api.workspaces.trash(name);
    await applyWorkspaceChange(result);
  } catch (err) {
    console.error('Failed to delete workspace:', err);
    try {
      await loadWorkspaces();
    } catch (_e) {
      // ignore
    }
  } finally {
    workspaceSwitchInFlight = false;
    syncWorkspaceControlsDisabled();
  }
}

function initializeWorkspacesPanel() {
  if (!(window.api && window.api.workspaces) || !(workspaceSelectEl instanceof HTMLSelectElement)) {
    return;
  }

  workspaceSelectEl.addEventListener('change', () => {
    void handleWorkspaceSelectChange();
  });
  workspaceCreateEl?.addEventListener('click', () => {
    void handleWorkspaceCreate();
  });
  workspaceRenameEl?.addEventListener('click', () => {
    void handleWorkspaceRename();
  });
  workspaceTrashEl?.addEventListener('click', () => {
    void handleWorkspaceTrash();
  });

  void loadWorkspaces();
}
