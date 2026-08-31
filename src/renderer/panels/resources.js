// Resources side panel: file tree backed by shared/treePanel.js, plus the
// features that are unique to Resources (uploading from the OS, creating
// documents, importing files dropped from the Outputs panel, and keyboard
// shortcuts shared between both tree panels).

function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

const resourcesTreeEl = document.getElementById('resources-tree');
const resourcesStatusEl = document.getElementById('resources-status');
const resourcesDropAreaEl = document.getElementById('resources-drop-area');
const resourcesDropzoneEl = document.getElementById('resources-dropzone');
const resourcesPanelEl = document.getElementById('resources-panel');
const resourcesContextMenuEl = document.getElementById('resources-context-menu');

/** Suspends side-panel polling while the user is actively dragging over the drop area. */
let dropAreaDragDepth = 0;

const resourcesPanel = window.Glaux.TreePanel.create({
  panelName: 'resources',
  api: window.api && window.api.resources,
  supportsCreateDocument: true,
  dataTransferType: 'text/x-resource-path',
  dragEffectAllowed: 'copyMove',
  elements: {
    treeEl: resourcesTreeEl,
    statusEl: resourcesStatusEl,
    panelEl: resourcesPanelEl,
    contextMenuEl: resourcesContextMenuEl,
    dropAreaEl: resourcesDropAreaEl,
  },
  renderEmptyState(hasTree, hasItems) {
    if (!hasTree) {
      resourcesTreeEl?.classList.add('hidden');
      resourcesDropzoneEl?.classList.remove('hidden');
      resourcesDropAreaEl?.classList.add('empty');
      return;
    }
    resourcesTreeEl?.classList.toggle('hidden', !hasItems);
    resourcesDropzoneEl?.classList.toggle('hidden', hasItems);
    resourcesDropAreaEl?.classList.toggle('empty', !hasItems);
  },
  onRenderComplete: () => updateMessageInputHighlight(),
  closePeerContextMenu: () => window.Glaux.Outputs.panel.closeContextMenu(),
  clearPeerSelection: () => window.Glaux.Outputs.panel.clearSelection(),
  getPeerDraggedPath: () => window.Glaux.Outputs.panel.draggedPath,
  onPeerDrop: async (peerPath, targetFolderPath) => {
    window.Glaux.Outputs.panel.draggedPath = '';
    await importOutputToResources(peerPath, targetFolderPath);
  },
  onExternalFilesDrop: async (paths, targetFolderPath) => {
    await uploadFilesIntoFolder(paths, targetFolderPath);
  },
  onExternalContentDrop: async (dataTransfer, targetFolderPath) => {
    window.Glaux.Dialogs.showProgressModal({
      title: t('panels.resources.preparing'),
      message: t('panels.resources.readingDroppedItems'),
    });
    const uploadPayload = await buildUploadPayloadFromDataTransfer(dataTransfer);
    if (uploadPayload.length) {
      await uploadFileContentsIntoFolder(uploadPayload, targetFolderPath);
    } else {
      window.Glaux.Dialogs.hideProgressModal();
      resourcesPanel.setStatus(t('panels.resources.noFilesInDrop'), true);
    }
  },
  onClearDropHighlight: () => {
    dropAreaDragDepth = 0;
  },
});

window.Glaux.Resources = { panel: resourcesPanel };
window.Glaux.MediaKinds.registerStatusSetter('resources', (message, isError) =>
  resourcesPanel.setStatus(message, isError)
);

function createProgressToken(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function formatProgressCurrent(current) {
  if (!current) {
    return '';
  }
  return current.length > 80 ? `${current.slice(0, 77)}...` : current;
}

let progressUnsubscribe = null;
let activeProgressToken = '';

async function chooseUploadConflictMode(conflictCount) {
  if (!conflictCount) {
    return 'error';
  }

  const choice = await window.Glaux.Dialogs.showChoiceDialog({
    title: t('panels.resources.conflictTitle'),
    message: t('panels.resources.conflictMessage', {
      count: conflictCount,
      conflictCount,
    }),
    options: [
      { id: 'cancel', label: t('dialogs.cancel'), danger: false },
      { id: 'skip', label: t('panels.resources.skipExisting'), danger: false },
      { id: 'overwrite', label: t('panels.resources.overwrite'), danger: true },
    ],
  });

  if (choice === 'overwrite') {
    return 'overwrite';
  }
  if (choice === 'skip') {
    return 'skip';
  }
  return 'cancel';
}

async function uploadFilesIntoFolder(filePaths, targetFolderPath) {
  if (!filePaths.length) {
    return;
  }

  resourcesPanel.setBusy(true);
  try {
    window.Glaux.Dialogs.showProgressModal({
      title: t('panels.resources.preparing'),
      message: t('panels.resources.checkingExistingItems'),
    });

    const analysis = await window.api.resources.getUploadConflictsForPaths(
      filePaths,
      targetFolderPath
    );
    const totalCount = analysis.totalCount || filePaths.length;
    let conflictMode = 'error';

    if (analysis.conflictCount > 0) {
      window.Glaux.Dialogs.hideProgressModal();
      conflictMode = await chooseUploadConflictMode(analysis.conflictCount);
      if (conflictMode === 'cancel') {
        resourcesPanel.setStatus(t('panels.resources.operationCancelled'), false);
        return;
      }
    }

    const progressToken = createProgressToken('upload-paths');
    activeProgressToken = progressToken;
    const estimatedUploadCount =
      conflictMode === 'skip'
        ? Math.max(0, totalCount - (analysis.conflictCount || 0))
        : totalCount;
    if (conflictMode === 'skip' && estimatedUploadCount === 0) {
      window.Glaux.Dialogs.hideProgressModal();
      resourcesPanel.setStatus(
        t('panels.resources.addedWithSkipped', {
          count: 0,
          uploadedCount: 0,
          skippedCount: analysis.conflictCount || totalCount,
        }),
        false
      );
      return;
    }
    window.Glaux.Dialogs.showProgressModal({
      title: t('panels.resources.addingItems'),
      message: t('panels.resources.progressCount', { completed: 0, total: estimatedUploadCount }),
    });
    await window.Glaux.Dialogs.waitForUiPaint();

    const uploadResult = await window.api.resources.uploadFilesWithMode(
      filePaths,
      targetFolderPath,
      conflictMode,
      progressToken
    );
    resourcesPanel.tree = uploadResult.tree;
    resourcesPanel.rebuildNodeIndex();
    resourcesPanel.keepExpandedFoldersValid();
    resourcesPanel.state.expandedFolderPaths.add(targetFolderPath || '');
    resourcesPanel.render();
    const uploadedCount =
      uploadResult?.stats && typeof uploadResult.stats.uploadedCount === 'number'
        ? uploadResult.stats.uploadedCount
        : filePaths.length;
    const skippedCount =
      uploadResult?.stats && typeof uploadResult.stats.skippedCount === 'number'
        ? uploadResult.stats.skippedCount
        : 0;
    if (skippedCount > 0) {
      resourcesPanel.setStatus(
        t('panels.resources.addedWithSkipped', {
          count: uploadedCount,
          uploadedCount,
          skippedCount,
        }),
        false
      );
    } else {
      resourcesPanel.setStatus(
        t('panels.resources.added', { count: uploadedCount, uploadedCount }),
        false
      );
    }
  } catch (err) {
    resourcesPanel.setStatus(err.message || String(err), true);
  } finally {
    window.Glaux.Dialogs.hideProgressModal();
    activeProgressToken = '';
    resourcesPanel.setBusy(false);
  }
}

async function uploadFileContentsIntoFolder(files, targetFolderPath) {
  if (!files.length) {
    return;
  }

  resourcesPanel.setBusy(true);
  try {
    window.Glaux.Dialogs.showProgressModal({
      title: t('panels.resources.preparing'),
      message: t('panels.resources.checkingExistingItems'),
    });

    const analysis = await window.api.resources.getUploadConflictsForFileContents(
      files,
      targetFolderPath
    );
    const totalCount = analysis.totalCount || files.length;
    let conflictMode = 'error';

    if (analysis.conflictCount > 0) {
      window.Glaux.Dialogs.hideProgressModal();
      conflictMode = await chooseUploadConflictMode(analysis.conflictCount);
      if (conflictMode === 'cancel') {
        resourcesPanel.setStatus(t('panels.resources.operationCancelled'), false);
        return;
      }
    }

    const progressToken = createProgressToken('upload-contents');
    activeProgressToken = progressToken;
    const estimatedUploadCount =
      conflictMode === 'skip'
        ? Math.max(0, totalCount - (analysis.conflictCount || 0))
        : totalCount;
    if (conflictMode === 'skip' && estimatedUploadCount === 0) {
      window.Glaux.Dialogs.hideProgressModal();
      resourcesPanel.setStatus(
        t('panels.resources.addedWithSkipped', {
          count: 0,
          uploadedCount: 0,
          skippedCount: analysis.conflictCount || totalCount,
        }),
        false
      );
      return;
    }
    window.Glaux.Dialogs.showProgressModal({
      title: t('panels.resources.addingItems'),
      message: t('panels.resources.progressCount', { completed: 0, total: estimatedUploadCount }),
    });
    await window.Glaux.Dialogs.waitForUiPaint();

    const uploadResult = await window.api.resources.uploadFileContentsWithMode(
      files,
      targetFolderPath,
      conflictMode,
      progressToken
    );
    resourcesPanel.tree = uploadResult.tree;
    resourcesPanel.rebuildNodeIndex();
    resourcesPanel.keepExpandedFoldersValid();
    resourcesPanel.state.expandedFolderPaths.add(targetFolderPath || '');
    resourcesPanel.render();
    const uploadedCount =
      uploadResult?.stats && typeof uploadResult.stats.uploadedCount === 'number'
        ? uploadResult.stats.uploadedCount
        : files.length;
    const skippedCount =
      uploadResult?.stats && typeof uploadResult.stats.skippedCount === 'number'
        ? uploadResult.stats.skippedCount
        : 0;
    if (skippedCount > 0) {
      resourcesPanel.setStatus(
        t('panels.resources.addedWithSkipped', {
          count: uploadedCount,
          uploadedCount,
          skippedCount,
        }),
        false
      );
    } else {
      resourcesPanel.setStatus(
        t('panels.resources.added', { count: uploadedCount, uploadedCount }),
        false
      );
    }
  } catch (err) {
    resourcesPanel.setStatus(err.message || String(err), true);
  } finally {
    window.Glaux.Dialogs.hideProgressModal();
    activeProgressToken = '';
    resourcesPanel.setBusy(false);
  }
}

async function buildUploadPayloadFromDataTransfer(dataTransfer) {
  if (!dataTransfer) {
    return [];
  }

  async function readFileSystemEntryFile(fileEntry) {
    return new Promise((resolve, reject) => {
      fileEntry.file(resolve, reject);
    });
  }

  async function readAllDirectoryEntries(directoryEntry) {
    const reader = directoryEntry.createReader();
    const entries = [];

    while (true) {
      const batch = await new Promise((resolve, reject) => {
        reader.readEntries(resolve, reject);
      });
      if (!batch.length) {
        break;
      }
      entries.push(...batch);
    }

    return entries;
  }

  async function collectEntryFiles(entry, parentPath, acc) {
    if (entry.isFile) {
      const file = await readFileSystemEntryFile(entry);
      if (!file || typeof file.arrayBuffer !== 'function') {
        return;
      }

      const relPath = parentPath ? `${parentPath}/${entry.name}` : entry.name;
      const fileBuffer = await file.arrayBuffer();
      acc.push({
        name: entry.name,
        relativePath: relPath,
        content: new Uint8Array(fileBuffer),
      });
      return;
    }

    if (entry.isDirectory) {
      const nextPath = parentPath ? `${parentPath}/${entry.name}` : entry.name;
      const children = await readAllDirectoryEntries(entry);
      for (const child of children) {
        await collectEntryFiles(child, nextPath, acc);
      }
    }
  }

  const payload = [];
  const entries = [];

  if (dataTransfer.items) {
    for (const item of dataTransfer.items) {
      if (!item || typeof item.webkitGetAsEntry !== 'function') {
        continue;
      }
      const entry = item.webkitGetAsEntry();
      if (entry) {
        entries.push(entry);
      }
    }
  }

  if (entries.length) {
    for (const entry of entries) {
      await collectEntryFiles(entry, '', payload);
    }
    if (payload.length) {
      return payload;
    }
  }

  if (!dataTransfer.files || !dataTransfer.files.length) {
    return payload;
  }

  for (const file of dataTransfer.files) {
    if (!file || !file.name || typeof file.arrayBuffer !== 'function') {
      continue;
    }

    const fileBuffer = await file.arrayBuffer();
    payload.push({
      name: file.name,
      relativePath: file.name,
      content: new Uint8Array(fileBuffer),
    });
  }

  return payload;
}

async function importOutputToResources(outputRelativePath, targetFolderPath) {
  resourcesPanel.setBusy(true);
  try {
    resourcesPanel.tree = await window.api.resources.importFromOutputs(
      outputRelativePath,
      targetFolderPath
    );
    resourcesPanel.rebuildNodeIndex();
    resourcesPanel.state.expandedFolderPaths.add(targetFolderPath || '');
    resourcesPanel.keepExpandedFoldersValid();
    resourcesPanel.render();
    resourcesPanel.setStatus(t('panels.resources.copiedFromOutputs'), false);
  } catch (err) {
    resourcesPanel.setStatus(err.message || String(err), true);
  } finally {
    resourcesPanel.setBusy(false);
  }
}

/** Shared by Resources and Outputs: both panels expose the same instance API. */
function initializeResourcesKeyboardShortcuts() {
  document.addEventListener('keydown', async (event) => {
    if (window.Glaux.Dialogs.isConfirmDialogActive() || window.Glaux.Dialogs.isNameDialogActive()) {
      return;
    }

    const outputsPanel = window.Glaux.Outputs.panel;

    if (activePanel === 'resources' && resourcesPanel.busy) {
      return;
    }
    if (activePanel === 'outputs' && outputsPanel.busy) {
      return;
    }

    const activeEl = document.activeElement;
    if (
      activeEl &&
      (activeEl.tagName === 'TEXTAREA' ||
        activeEl.tagName === 'INPUT' ||
        activeEl.tagName === 'SELECT')
    ) {
      return;
    }

    const currentPanel = activePanel === 'outputs' ? outputsPanel : resourcesPanel;

    if (event.key === 'Enter') {
      const node = currentPanel.getSelectedNode();
      if (window.Glaux.MediaKinds.tryOpenTreeFile(node, activePanel === 'outputs' ? 'outputs' : 'resources')) {
        event.preventDefault();
        currentPanel.closeContextMenu();
        return;
      }
    }

    if (event.key === 'F2') {
      const node = currentPanel.getSelectedNode();
      if (node && node.relativePath) {
        event.preventDefault();
        currentPanel.closeContextMenu();
        currentPanel.startInlineRename(node);
      }
      return;
    }

    if (event.key === 'Delete') {
      const node = currentPanel.getSelectedNode();
      if (node && node.relativePath) {
        event.preventDefault();
        currentPanel.closeContextMenu();
        await currentPanel.deleteEntry(node);
      }
    }
  });
}

function initializeResourcesPanel() {
  if (!(window.api && window.api.resources) || !resourcesTreeEl) {
    return;
  }

  resourcesPanelEl?.addEventListener('mousedown', () => {
    activePanel = 'resources';
  });

  resourcesDropAreaEl?.addEventListener('dragenter', (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropAreaDragDepth = 1;
    resourcesDropAreaEl.classList.add('drag-over');
  });
  resourcesDropAreaEl?.addEventListener('dragover', (event) => {
    event.preventDefault();
    event.stopPropagation();
    resourcesDropAreaEl.classList.add('drag-over');
    if (event.target instanceof Element && event.target.closest('.tree-row')) {
      return;
    }
    const dropEffect = resourcesPanel.draggedPath ? 'move' : 'copy';
    if (
      event.target instanceof Element &&
      resourcesTreeEl &&
      resourcesTreeEl.contains(event.target) &&
      !event.target.classList.contains('tree-root-drop-spacer')
    ) {
      event.dataTransfer.dropEffect = dropEffect;
      return;
    }
    resourcesPanel.setActiveDropTarget(null);
    event.dataTransfer.dropEffect = dropEffect;
  });
  resourcesDropAreaEl?.addEventListener('dragleave', (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (
      resourcesDropAreaEl &&
      event.relatedTarget instanceof Node &&
      resourcesDropAreaEl.contains(event.relatedTarget)
    ) {
      return;
    }
    resourcesPanel.clearDropHighlight();
  });

  resourcesDropAreaEl?.addEventListener('drop', async (event) => {
    resourcesPanel.closeContextMenu();
    if (event.target instanceof Element && event.target.closest('.tree-row')) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    resourcesPanel.clearDropHighlight();
    await resourcesPanel.resolveDrop(event, '');
  });

  resourcesTreeEl.addEventListener('click', () => {
    resourcesPanel.closeContextMenu();
  });

  resourcesPanel.initPanelChrome();

  if (progressUnsubscribe) {
    progressUnsubscribe();
  }
  progressUnsubscribe = window.api.resources.onUploadProgress((payload) => {
    if (!payload || payload.token !== activeProgressToken) {
      return;
    }
    const completed = Number(payload.completed || 0);
    const total = Number(payload.total || 0);
    const current = formatProgressCurrent(payload.current || '');
    window.Glaux.Dialogs.updateProgressModal({
      title: t('panels.resources.addingItems'),
      message: current
        ? t('panels.resources.progressCountWithCurrent', { completed, total, current })
        : t('panels.resources.progressCount', { completed, total }),
      completed,
      total,
    });
  });

  initializeResourcesKeyboardShortcuts();
  resourcesPanel.refreshTree();
}

function shouldSuspendResourcesPolling() {
  return (
    resourcesPanel.busy ||
    Boolean(resourcesPanel.editingEntryPath) ||
    Boolean(resourcesPanel.draggedPath) ||
    Boolean(window.Glaux.Outputs.panel.draggedPath) ||
    dropAreaDragDepth > 0 ||
    Boolean(resourcesContextMenuEl && !resourcesContextMenuEl.classList.contains('hidden')) ||
    window.Glaux.Dialogs.isConfirmDialogActive() ||
    window.Glaux.Dialogs.isNameDialogActive() ||
    window.Glaux.Dialogs.isProgressModalOpen()
  );
}
