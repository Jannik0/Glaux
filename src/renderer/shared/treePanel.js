// Generic factory behind the Resources and Outputs side panels.
// Resources and Outputs are near-identical file trees (list/select/rename/
// move/delete/context-menu/drag-drop); this module holds all of the logic
// that is genuinely shared, and panels/resources.js + panels/outputs.js
// configure the (small) set of real differences: create-document support,
// upload-from-OS-drag, importing from the peer panel, and native drag-out.
(function () {
  window.Glaux = window.Glaux || {};

  function t(key, vars) {
    return window.Glaux.i18n.t(key, vars);
  }

  const { Status } = window.Glaux;
  const { Dialogs } = window.Glaux;
  const { MediaKinds } = window.Glaux;

  function getParentFolderPath(entryPath) {
    if (!entryPath) {
      return '';
    }
    const lastSlash = entryPath.lastIndexOf('/');
    return lastSlash === -1 ? '' : entryPath.slice(0, lastSlash);
  }

  function validateEntryName(name) {
    if (typeof name !== 'string') {
      throw new Error(t('panels.tree.nameMustBeString'));
    }
    const trimmed = name.trim();
    if (!trimmed) {
      throw new Error(t('panels.tree.nameCannotBeEmpty'));
    }
    if (trimmed === '.' || trimmed === '..') {
      throw new Error(t('panels.tree.invalidName'));
    }
    if (trimmed.includes('/') || trimmed.includes('\\')) {
      throw new Error(t('panels.tree.nameCannotContainSeparators'));
    }
    return trimmed;
  }

  function canOfferOverwrite(err) {
    if (err && err.code === 'E_ALREADY_EXISTS') {
      return true;
    }
    const message = String((err && err.message) || err || '').toLowerCase();
    return message.includes('already exists') || message.includes('already contains');
  }

  async function runWithOptionalOverwrite(action, actionName) {
    try {
      await action(false);
      return true;
    } catch (err) {
      if (canOfferOverwrite(err)) {
        const shouldOverwrite = await Dialogs.showConfirmDialog({
          title: t('panels.tree.conflictTitle', { action: actionName }),
          message: t('panels.tree.conflictOverwriteMessage'),
          confirmLabel: t('panels.tree.overwrite'),
        });
        if (shouldOverwrite) {
          await action(true);
          return true;
        }
      }
      throw err;
    }
  }

  function flattenFolderPaths(node, acc = []) {
    if (!node || node.type !== 'folder') {
      return acc;
    }
    acc.push(node.relativePath);
    for (const child of node.children || []) {
      flattenFolderPaths(child, acc);
    }
    return acc;
  }

  function isInvalidMoveTarget(sourcePath, targetFolderPath) {
    if (!sourcePath) {
      return true;
    }
    if (sourcePath === targetFolderPath) {
      return true;
    }
    return Boolean(
      targetFolderPath && sourcePath && targetFolderPath.startsWith(`${sourcePath}/`)
    );
  }

  function getDefaultNewFolderName(nodesByPath, parentFolderPath) {
    const parentNode = nodesByPath.get(parentFolderPath || '') || nodesByPath.get('');
    const existingNames = new Set(
      ((parentNode && parentNode.children) || [])
        .filter((child) => child.type === 'folder')
        .map((child) => child.name.toLowerCase())
    );
    const baseName = 'New Folder';
    if (!existingNames.has(baseName.toLowerCase())) {
      return baseName;
    }
    let index = 2;
    while (existingNames.has(`${baseName} ${index}`.toLowerCase())) {
      index += 1;
    }
    return `${baseName} ${index}`;
  }

  function getDefaultNewDocumentName(nodesByPath, parentFolderPath) {
    const parentNode = nodesByPath.get(parentFolderPath || '') || nodesByPath.get('');
    const existingNames = new Set(
      ((parentNode && parentNode.children) || [])
        .filter((child) => child.type === 'file')
        .map((child) => child.name.toLowerCase())
    );
    const baseName = 'New Document.md';
    if (!existingNames.has(baseName.toLowerCase())) {
      return baseName;
    }
    let index = 2;
    while (existingNames.has(`New Document ${index}.md`.toLowerCase())) {
      index += 1;
    }
    return `New Document ${index}.md`;
  }

  /**
   * @param {{
   *   panelName: 'resources' | 'outputs',
   *   api: { listTree: Function, createFolder: Function, createFile?: Function, renameEntry: Function, moveEntry: Function, deleteEntry: Function },
   *   supportsCreateDocument?: boolean,
   *   dataTransferType: string,
   *   dragEffectAllowed: 'copyMove' | 'copy',
   *   onlyAcceptsOwnDrag?: boolean,
   *   elements: { treeEl: HTMLElement, statusEl: HTMLElement, panelEl: HTMLElement, contextMenuEl: HTMLElement, dropAreaEl?: HTMLElement },
   *   renderEmptyState: (hasTree: boolean, hasItems: boolean) => void,
   *   onRenderComplete?: () => void,
   *   closePeerContextMenu?: () => void,
   *   clearPeerSelection?: () => void,
   *   getPeerDraggedPath?: () => string,
   *   onPeerDrop?: (peerPath: string, targetFolderPath: string) => Promise<void>,
   *   onExternalFilesDrop?: (paths: string[], targetFolderPath: string) => Promise<void>,
   *   onExternalContentDrop?: (dataTransfer: DataTransfer, targetFolderPath: string) => Promise<void>,
   *   onClearDropHighlight?: () => void,
   *   onDragEndExtra?: () => void,
   * }} config
   */
  function createTreePanel(config) {
    const { elements } = config;

    const panel = {
      tree: null,
      selectedFolderPath: '',
      selectedEntryPath: '',
      selectedEntryType: 'folder',
      busy: false,
      draggedPath: '',
      editingEntryPath: '',
      editingEntryValue: '',
      activeDropTargetPath: null,
      expandedFolderPaths: new Set(['']),
      nodesByPath: new Map(),
      rowElsByPath: new Map(),
    };

    function setBusy(isBusy) {
      panel.busy = isBusy;
      if (isBusy) {
        closeContextMenu();
        cancelInlineRename();
      }
    }

    function setStatus(message, isError = false) {
      Status.setTimedPanelStatus(elements.statusEl, config.panelName, message, isError, 'error');
    }

    function getSelectedNode() {
      if (!panel.selectedEntryPath && panel.tree) {
        return panel.tree;
      }
      return panel.nodesByPath.get(panel.selectedEntryPath) || null;
    }

    function getActiveFolderPath() {
      if (!panel.selectedEntryPath) {
        return panel.selectedFolderPath || '';
      }
      if (panel.selectedEntryType === 'folder') {
        return panel.selectedEntryPath;
      }
      return getParentFolderPath(panel.selectedEntryPath);
    }

    function selectEntry(node) {
      panel.selectedEntryPath = node.relativePath;
      panel.selectedEntryType = node.type;
      panel.selectedFolderPath =
        node.type === 'folder' ? node.relativePath : getParentFolderPath(node.relativePath);
    }

    function clearSelection() {
      if (!panel.selectedEntryPath) {
        return;
      }
      panel.selectedEntryPath = '';
      panel.selectedEntryType = 'folder';
      render();
    }

    function closeContextMenu() {
      if (!elements.contextMenuEl) {
        return;
      }
      elements.contextMenuEl.classList.add('hidden');
      elements.contextMenuEl.textContent = '';
    }

    function cancelInlineRename() {
      panel.editingEntryPath = '';
      panel.editingEntryValue = '';
    }

    function rebuildNodeIndex() {
      panel.nodesByPath.clear();
      if (!panel.tree) {
        return;
      }
      const stack = [panel.tree];
      while (stack.length) {
        const next = stack.pop();
        panel.nodesByPath.set(next.relativePath, next);
        if (next.type === 'folder') {
          for (const child of next.children || []) {
            stack.push(child);
          }
        }
      }
    }

    function keepExpandedFoldersValid() {
      if (!panel.tree) {
        panel.expandedFolderPaths.clear();
        panel.expandedFolderPaths.add('');
        panel.selectedFolderPath = '';
        panel.selectedEntryPath = '';
        panel.selectedEntryType = 'folder';
        return;
      }
      const validFolders = new Set(flattenFolderPaths(panel.tree));
      for (const expandedPath of [...panel.expandedFolderPaths]) {
        if (!validFolders.has(expandedPath)) {
          panel.expandedFolderPaths.delete(expandedPath);
        }
      }
      if (!validFolders.has(panel.selectedFolderPath)) {
        panel.selectedFolderPath = '';
      }
      if (panel.selectedEntryPath && !panel.nodesByPath.has(panel.selectedEntryPath)) {
        panel.selectedEntryPath = '';
        panel.selectedEntryType = 'folder';
      }
      if (panel.editingEntryPath && !panel.nodesByPath.has(panel.editingEntryPath)) {
        cancelInlineRename();
      }
      if (!panel.expandedFolderPaths.has('')) {
        panel.expandedFolderPaths.add('');
      }
    }

    function updateSelectedPathAfterRelocation(oldPath, newPath) {
      if (panel.selectedEntryPath === oldPath) {
        panel.selectedEntryPath = newPath;
      } else if (oldPath && panel.selectedEntryPath.startsWith(`${oldPath}/`)) {
        panel.selectedEntryPath = `${newPath}${panel.selectedEntryPath.slice(oldPath.length)}`;
      }
      if (!panel.selectedFolderPath) {
        return;
      }
      if (panel.selectedFolderPath === oldPath) {
        panel.selectedFolderPath = newPath;
        return;
      }
      if (oldPath && panel.selectedFolderPath.startsWith(`${oldPath}/`)) {
        panel.selectedFolderPath = `${newPath}${panel.selectedFolderPath.slice(oldPath.length)}`;
      }
    }

    async function refreshTree(successMessage = '') {
      setBusy(true);
      try {
        panel.tree = await config.api.listTree();
        rebuildNodeIndex();
        keepExpandedFoldersValid();
        render();
        if (successMessage) {
          setStatus(successMessage, false);
        }
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        setBusy(false);
      }
    }

    /** Replace the tree wholesale (e.g. after an operation performed by another panel). */
    function setTree(nextTree) {
      panel.tree = nextTree;
      rebuildNodeIndex();
      keepExpandedFoldersValid();
      render();
    }

    async function createFolder(parentFolderPath, folderName) {
      let safeName;
      try {
        safeName = validateEntryName(
          typeof folderName === 'string' && folderName.trim()
            ? folderName
            : getDefaultNewFolderName(panel.nodesByPath, parentFolderPath)
        );
      } catch (err) {
        setStatus(err.message || String(err), true);
        return;
      }
      let createdNodeForRename = null;

      setBusy(true);
      try {
        panel.tree = await config.api.createFolder(parentFolderPath, safeName);
        rebuildNodeIndex();
        panel.expandedFolderPaths.add(parentFolderPath || '');
        keepExpandedFoldersValid();
        const createdPath = parentFolderPath ? `${parentFolderPath}/${safeName}` : safeName;
        createdNodeForRename = panel.nodesByPath.get(createdPath) || null;
        render();
        setStatus(t('panels.tree.createdFolder', { name: safeName }), false);
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        setBusy(false);
      }

      if (createdNodeForRename) {
        startInlineRename(createdNodeForRename);
      }
    }

    async function createDocument(parentFolderPath, fileName) {
      if (!config.supportsCreateDocument || typeof config.api.createFile !== 'function') {
        return;
      }
      let safeName;
      try {
        safeName = validateEntryName(
          typeof fileName === 'string' && fileName.trim()
            ? fileName
            : getDefaultNewDocumentName(panel.nodesByPath, parentFolderPath)
        );
      } catch (err) {
        setStatus(err.message || String(err), true);
        return;
      }

      if (!MediaKinds.isMarkdownFile(safeName)) {
        setStatus(t('panels.tree.documentMustEndWithMdOrTxt'), true);
        return;
      }

      let createdNodeForRename = null;

      setBusy(true);
      try {
        panel.tree = await config.api.createFile(parentFolderPath, safeName);
        rebuildNodeIndex();
        panel.expandedFolderPaths.add(parentFolderPath || '');
        keepExpandedFoldersValid();
        const createdPath = parentFolderPath ? `${parentFolderPath}/${safeName}` : safeName;
        createdNodeForRename = panel.nodesByPath.get(createdPath) || null;
        render();
        setStatus(t('panels.tree.createdDocument', { name: safeName }), false);
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        setBusy(false);
      }

      if (createdNodeForRename) {
        startInlineRename(createdNodeForRename);
      }
    }

    async function renameEntry(node, nextName) {
      let safeName;
      try {
        safeName = validateEntryName(nextName);
      } catch (err) {
        setStatus(err.message || String(err), true);
        return;
      }

      const oldPath = node.relativePath;
      const oldParts = oldPath ? oldPath.split('/') : [];
      oldParts[oldParts.length - 1] = safeName;
      const newPath = oldParts.join('/');

      setBusy(true);
      try {
        await runWithOptionalOverwrite(async (overwrite) => {
          panel.tree = await config.api.renameEntry(node.relativePath, safeName, overwrite);
        }, t('panels.tree.actionRename'));
        rebuildNodeIndex();
        updateSelectedPathAfterRelocation(oldPath, newPath);
        if (panel.editingEntryPath === oldPath) {
          cancelInlineRename();
        }
        keepExpandedFoldersValid();
        render();
        setStatus(t('panels.tree.renamedSuccessfully'), false);
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        setBusy(false);
      }
    }

    async function commitInlineRename(node, nextName) {
      if (!node || !node.relativePath) {
        cancelInlineRename();
        return;
      }

      const value = typeof nextName === 'string' ? nextName : '';
      if (value.trim() === node.name) {
        cancelInlineRename();
        render();
        return;
      }

      if (node.type === 'file') {
        const oldExtension = MediaKinds.getFileNameParts(node.name).extension.toLowerCase();
        const newExtension = MediaKinds.getFileNameParts(value.trim()).extension.toLowerCase();
        if (oldExtension !== newExtension) {
          const oldLabel = oldExtension || t('panels.tree.extensionNone');
          const newLabel = newExtension || t('panels.tree.extensionNone');
          const shouldProceed = await Dialogs.showConfirmDialog({
            title: t('panels.tree.changeExtensionTitle'),
            message: t('panels.tree.changeExtensionMessage', { oldLabel, newLabel }),
            confirmLabel: t('panels.tree.changeExtensionConfirm'),
          });
          if (!shouldProceed) {
            cancelInlineRename();
            render();
            return;
          }
        }
      }

      await renameEntry(node, value);
    }

    async function moveEntry(sourcePath, targetFolderPath) {
      if (isInvalidMoveTarget(sourcePath, targetFolderPath)) {
        setStatus(t('panels.tree.invalidMoveTarget'), true);
        return;
      }

      const sourceParts = sourcePath.split('/');
      const sourceName = sourceParts[sourceParts.length - 1];
      const newPath = targetFolderPath ? `${targetFolderPath}/${sourceName}` : sourceName;

      setBusy(true);
      try {
        await runWithOptionalOverwrite(async (overwrite) => {
          panel.tree = await config.api.moveEntry(sourcePath, targetFolderPath, overwrite);
        }, t('panels.tree.actionMove'));
        rebuildNodeIndex();
        updateSelectedPathAfterRelocation(sourcePath, newPath);
        panel.expandedFolderPaths.add(targetFolderPath || '');
        keepExpandedFoldersValid();
        render();
        setStatus(t('panels.tree.movedSuccessfully'), false);
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        setBusy(false);
      }
    }

    async function deleteEntry(node) {
      if (!node || !node.relativePath) {
        setStatus(t('panels.tree.rootCannotBeDeleted'), true);
        return;
      }

      const kind = node.type === 'folder' ? t('panels.tree.deleteFolder') : t('panels.tree.deleteFile');
      const confirmed = await Dialogs.showConfirmDialog({
        title: t('panels.tree.deleteTitle', { kind }),
        message: t('panels.tree.deleteMessage', { name: node.name }),
        confirmLabel: t('panels.tree.delete'),
      });
      if (!confirmed) {
        return;
      }

      Dialogs.showProgressModal({
        title: t('panels.tree.deletingTitle'),
        message: t('panels.tree.deletingMessage'),
      });
      Dialogs.updateProgressModal({ completed: 0, total: 1 });
      setBusy(true);
      try {
        panel.tree = await config.api.deleteEntry(node.relativePath);
        rebuildNodeIndex();
        if (panel.selectedEntryPath === node.relativePath) {
          panel.selectedEntryPath = '';
          panel.selectedEntryType = 'folder';
          panel.selectedFolderPath = getParentFolderPath(node.relativePath);
        }
        keepExpandedFoldersValid();
        render();
        Dialogs.updateProgressModal({ completed: 1, total: 1 });
        setStatus(t('panels.tree.deletedSuccessfully'), false);
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        Dialogs.hideProgressModal();
        setBusy(false);
      }
    }

    function focusInlineRenameInput(entryPath, entryType) {
      const renameInput = elements.treeEl?.querySelector(
        `input.tree-rename-input[data-entry-path="${entryPath}"]`
      );
      if (!renameInput) {
        return;
      }
      renameInput.focus();
      if (entryType === 'file') {
        const { baseName, extension } = MediaKinds.getFileNameParts(renameInput.value);
        if (extension) {
          renameInput.setSelectionRange(0, baseName.length);
          return;
        }
      }
      renameInput.select();
    }

    function startInlineRename(node) {
      if (!node || !node.relativePath || panel.busy) {
        return;
      }
      panel.editingEntryPath = node.relativePath;
      panel.editingEntryValue = node.name;
      selectEntry(node);
      closeContextMenu();
      render();
      requestAnimationFrame(() => {
        focusInlineRenameInput(node.relativePath, node.type);
      });
    }

    function addContextMenuItem(label, onClick) {
      if (!elements.contextMenuEl) {
        return;
      }
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'context-menu-item';
      button.textContent = label;
      button.addEventListener('click', async (event) => {
        event.preventDefault();
        closeContextMenu();
        await onClick();
      });
      elements.contextMenuEl.appendChild(button);
    }

    function openContextMenu(x, y, node) {
      if (!elements.contextMenuEl || panel.busy) {
        return;
      }

      elements.contextMenuEl.textContent = '';

      if (node.type === 'folder') {
        addContextMenuItem(t('panels.tree.newFolder'), async () => createFolder(node.relativePath));
        if (config.supportsCreateDocument) {
          addContextMenuItem(t('panels.tree.newDocument'), async () => createDocument(node.relativePath));
        }
      }

      if (node.relativePath) {
        addContextMenuItem(t('panels.tree.rename'), async () => startInlineRename(node));
        addContextMenuItem(t('panels.tree.delete'), async () => deleteEntry(node));
      }

      if (!elements.contextMenuEl.children.length) {
        closeContextMenu();
        return;
      }

      elements.contextMenuEl.classList.remove('hidden');
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const menuWidth = 180;
      const menuHeight = elements.contextMenuEl.offsetHeight || 120;
      const left = Math.min(x, viewportWidth - menuWidth - 8);
      const top = Math.min(y, viewportHeight - menuHeight - 8);
      elements.contextMenuEl.style.left = `${Math.max(8, left)}px`;
      elements.contextMenuEl.style.top = `${Math.max(8, top)}px`;
    }

    function toggleRowDropTarget(folderPath, shouldHighlight) {
      if (!folderPath) {
        return;
      }
      const targetRow = panel.rowElsByPath.get(folderPath);
      if (!targetRow) {
        return;
      }
      targetRow.classList.toggle('drop-target', Boolean(shouldHighlight));
    }

    function setActiveDropTarget(nextFolderPath) {
      const normalizedPath = typeof nextFolderPath === 'string' ? nextFolderPath : null;
      if (panel.activeDropTargetPath === normalizedPath) {
        return;
      }
      if (panel.activeDropTargetPath !== null) {
        toggleRowDropTarget(panel.activeDropTargetPath, false);
      }
      panel.activeDropTargetPath = normalizedPath;
      if (panel.activeDropTargetPath !== null) {
        toggleRowDropTarget(panel.activeDropTargetPath, true);
      }
    }

    function clearDropHighlight() {
      panel.activeDropTargetPath = null;
      elements.dropAreaEl?.classList.remove('drag-over');
      for (const rowEl of panel.rowElsByPath.values()) {
        rowEl.classList.remove('drop-target');
      }
      if (config.onClearDropHighlight) {
        config.onClearDropHighlight();
      }
    }

    /**
     * Shared drop-resolution pipeline used by both per-row drops and
     * "drop on empty area" (root) drops. Tries, in order: an internal move,
     * a drop coming from the peer panel, external OS files, then raw
     * dropped content (folders / non-file-backed drags).
     */
    async function resolveDrop(event, targetFolderPath) {
      const internalPath = event.dataTransfer.getData(config.dataTransferType);
      if (internalPath) {
        await moveEntry(internalPath, targetFolderPath);
        return;
      }

      const peerPath = config.getPeerDraggedPath ? config.getPeerDraggedPath() : '';
      if (peerPath && config.onPeerDrop) {
        await config.onPeerDrop(peerPath, targetFolderPath);
        return;
      }

      if (config.onExternalFilesDrop) {
        const externalPaths = window.Glaux.TreePanel.listExternalFilePaths(event.dataTransfer);
        if (externalPaths.length) {
          await config.onExternalFilesDrop(externalPaths, targetFolderPath);
          return;
        }
      }

      if (config.onExternalContentDrop) {
        await config.onExternalContentDrop(event.dataTransfer, targetFolderPath);
      }
    }

    function buildRowNode(node, depth) {
      const nodeEl = document.createElement('div');
      nodeEl.className = 'tree-node';
      const isEditing = Boolean(panel.editingEntryPath) && panel.editingEntryPath === node.relativePath;

      const rowEl = document.createElement('div');
      rowEl.className = `tree-row ${node.type}`;
      rowEl.style.paddingLeft = `${6 + depth * 14}px`;
      rowEl.draggable = node.relativePath !== '' && !isEditing;
      rowEl.setAttribute('data-entry-path', node.relativePath);
      rowEl.setAttribute('data-entry-type', node.type);
      panel.rowElsByPath.set(node.relativePath, rowEl);

      if (panel.selectedEntryPath && panel.selectedEntryPath === node.relativePath) {
        rowEl.classList.add('selected');
      }

      rowEl.addEventListener('dragstart', (event) => {
        if (!node.relativePath) {
          event.preventDefault();
          return;
        }
        panel.draggedPath = node.relativePath;
        event.dataTransfer.effectAllowed = config.dragEffectAllowed;
        event.dataTransfer.setData(config.dataTransferType, node.relativePath);
        event.dataTransfer.setData('text/plain', node.name);
      });

      const dropTargetFolderPath =
        node.type === 'folder' ? node.relativePath : getParentFolderPath(node.relativePath);

      rowEl.addEventListener('dragover', (event) => {
        if (config.onlyAcceptsOwnDrag && !panel.draggedPath) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        setActiveDropTarget(dropTargetFolderPath);
        event.dataTransfer.dropEffect = panel.draggedPath ? 'move' : 'copy';
      });

      rowEl.addEventListener('drop', async (event) => {
        event.preventDefault();
        event.stopPropagation();
        toggleRowDropTarget(dropTargetFolderPath, false);
        clearDropHighlight();
        await resolveDrop(event, dropTargetFolderPath);
      });

      rowEl.addEventListener('contextmenu', (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectEntry(node);
        render();
        openContextMenu(event.clientX, event.clientY, node);
      });

      const toggleEl = document.createElement('span');
      toggleEl.className = 'tree-toggle';
      if (node.type === 'folder' && (node.children || []).length > 0) {
        const isExpanded = panel.expandedFolderPaths.has(node.relativePath);
        toggleEl.textContent = isExpanded ? '▾' : '▸';
        toggleEl.addEventListener('click', (event) => {
          event.stopPropagation();
          if (isExpanded) {
            panel.expandedFolderPaths.delete(node.relativePath);
          } else {
            panel.expandedFolderPaths.add(node.relativePath);
          }
          render();
        });
      } else {
        toggleEl.textContent = '';
      }

      const iconEl = document.createElement('span');
      iconEl.className = 'tree-icon';
      const isExpandedFolder = node.type === 'folder' && panel.expandedFolderPaths.has(node.relativePath);
      iconEl.textContent =
        node.type === 'folder' ? (isExpandedFolder ? '📂' : '📁') : MediaKinds.getFileIcon(node.name);

      let nameEl;
      if (isEditing) {
        nameEl = document.createElement('input');
        nameEl.type = 'text';
        nameEl.className = 'tree-rename-input';
        nameEl.setAttribute('data-entry-path', node.relativePath);
        nameEl.value = panel.editingEntryValue || node.name;
        nameEl.addEventListener('mousedown', (event) => {
          event.stopPropagation();
        });
        nameEl.addEventListener('click', (event) => {
          event.stopPropagation();
        });
        nameEl.addEventListener('input', () => {
          panel.editingEntryValue = nameEl.value;
        });
        nameEl.addEventListener('keydown', async (event) => {
          event.stopPropagation();
          if (event.key === 'Enter') {
            event.preventDefault();
            await commitInlineRename(node, nameEl.value);
            return;
          }
          if (event.key === 'Escape') {
            event.preventDefault();
            cancelInlineRename();
            render();
          }
        });
        nameEl.addEventListener('blur', async () => {
          if (panel.editingEntryPath !== node.relativePath) {
            return;
          }
          await commitInlineRename(node, nameEl.value);
        });
      } else {
        nameEl = document.createElement('span');
        nameEl.className = 'tree-name';
        nameEl.textContent = node.name;
      }

      rowEl.addEventListener('click', (event) => {
        event.stopPropagation();
        selectEntry(node);
        if (node.type === 'folder') {
          panel.expandedFolderPaths.add(node.relativePath);
        }
        closeContextMenu();
        config.closePeerContextMenu?.();
        config.clearPeerSelection?.();
        render();
      });

      rowEl.addEventListener('dblclick', (event) => {
        event.stopPropagation();
        if (node.type === 'file') {
          MediaKinds.tryOpenTreeFile(node, config.panelName);
        }
      });

      rowEl.appendChild(toggleEl);
      rowEl.appendChild(iconEl);
      rowEl.appendChild(nameEl);
      nodeEl.appendChild(rowEl);

      if (node.type === 'folder' && panel.expandedFolderPaths.has(node.relativePath)) {
        const childrenContainer = document.createElement('div');
        childrenContainer.className = 'tree-children';
        for (const child of node.children || []) {
          childrenContainer.appendChild(buildRowNode(child, depth + 1));
        }
        nodeEl.appendChild(childrenContainer);
      }

      return nodeEl;
    }

    function render() {
      if (!elements.treeEl) {
        return;
      }
      panel.rowElsByPath.clear();
      elements.treeEl.textContent = '';

      if (!panel.tree) {
        config.renderEmptyState(false, false);
        config.onRenderComplete?.();
        return;
      }

      const children = panel.tree.children || [];
      const hasItems = children.length > 0;
      config.renderEmptyState(true, hasItems);

      for (const child of children) {
        elements.treeEl.appendChild(buildRowNode(child, 0));
      }
      const rootDropSpacer = document.createElement('div');
      rootDropSpacer.className = 'tree-root-drop-spacer';
      elements.treeEl.appendChild(rootDropSpacer);
      config.onRenderComplete?.();
    }

    /** Wires the panel-background context menu (right-click outside any row) and drag cleanup. */
    function initPanelChrome() {
      elements.panelEl?.addEventListener('contextmenu', (event) => {
        if (!panel.tree) {
          return;
        }
        if (event.target instanceof Element && event.target.closest('.tree-row')) {
          return;
        }
        event.preventDefault();
        const rootNode = panel.nodesByPath.get('') || panel.tree;
        if (rootNode) {
          selectEntry(rootNode);
          render();
          openContextMenu(event.clientX, event.clientY, rootNode);
        }
      });

      elements.panelEl?.addEventListener('dragend', () => {
        clearDropHighlight();
      });

      document.addEventListener('dragend', () => {
        if (panel.draggedPath) {
          panel.draggedPath = '';
          clearDropHighlight();
          config.onDragEndExtra?.();
        }
      });
      document.addEventListener(
        'drop',
        () => {
          requestAnimationFrame(() => {
            panel.draggedPath = '';
          });
        },
        true
      );
    }

    return {
      config,
      state: panel,
      get tree() {
        return panel.tree;
      },
      set tree(value) {
        panel.tree = value;
      },
      get draggedPath() {
        return panel.draggedPath;
      },
      set draggedPath(value) {
        panel.draggedPath = value;
      },
      get busy() {
        return panel.busy;
      },
      get editingEntryPath() {
        return panel.editingEntryPath;
      },
      nodesByPath: panel.nodesByPath,
      rowElsByPath: panel.rowElsByPath,
      setBusy,
      setStatus,
      getSelectedNode,
      getActiveFolderPath,
      selectEntry,
      clearSelection,
      closeContextMenu,
      cancelInlineRename,
      rebuildNodeIndex,
      keepExpandedFoldersValid,
      refreshTree,
      setTree,
      createFolder,
      createDocument,
      renameEntry,
      commitInlineRename,
      moveEntry,
      deleteEntry,
      startInlineRename,
      openContextMenu,
      setActiveDropTarget,
      toggleRowDropTarget,
      clearDropHighlight,
      resolveDrop,
      render,
      initPanelChrome,
    };
  }

  /** Best-effort resolution of a real filesystem path from a dropped File. */
  function resolveDroppedFilePath(file) {
    if (!file) {
      return '';
    }
    if (typeof file.path === 'string' && file.path.length > 0) {
      return file.path;
    }
    if (window.api && typeof window.api.getPathForFile === 'function') {
      const resolvedPath = window.api.getPathForFile(file);
      if (typeof resolvedPath === 'string' && resolvedPath.length > 0) {
        return resolvedPath;
      }
    }
    return '';
  }

  function fileUriToFsPath(uri) {
    let filePath = String(uri).replace(/^file:\/\//i, '');

    // Windows drive letter: file:///C:/Users/... → /C:/Users/...
    if (/^\/[a-zA-Z]:[\\/]/.test(filePath)) {
      filePath = filePath.slice(1);
      return decodeURIComponent(filePath).replace(/\//g, '\\');
    }

    // UNC: file://server/share/... → //server/share/...
    if (filePath.startsWith('//') || filePath.startsWith('\\\\')) {
      return decodeURIComponent(filePath).replace(/\//g, '\\');
    }

    const localhost = filePath.match(/^localhost(\/.*)$/i);
    if (localhost) {
      filePath = localhost[1];
    }

    filePath = decodeURIComponent(filePath);

    // Only force Windows separators when the host is Windows.
    const isWindows =
      (typeof window !== 'undefined' && window.api && window.api.platform === 'win32') ||
      /^[a-zA-Z]:[\\/]/.test(filePath);
    if (isWindows) {
      return filePath.replace(/\//g, '\\');
    }
    return filePath;
  }

  function listExternalFilePaths(dataTransfer) {
    if (!dataTransfer) {
      return [];
    }

    const filePaths = new Set();

    // Prefer item-based resolution first. Folder drops often expose one item
    // with a real filesystem path while also listing every nested file.
    if (dataTransfer.items) {
      for (const item of dataTransfer.items) {
        if (!item || typeof item.getAsFile !== 'function') {
          continue;
        }
        const itemFile = item.getAsFile();
        const resolvedPath = resolveDroppedFilePath(itemFile);
        if (resolvedPath) {
          filePaths.add(resolvedPath);
        }
      }
    }

    if (filePaths.size > 0) {
      return [...filePaths];
    }

    if (dataTransfer.files) {
      for (const file of dataTransfer.files) {
        const resolvedPath = resolveDroppedFilePath(file);
        if (resolvedPath) {
          filePaths.add(resolvedPath);
        }
      }
    }

    // Fallback for OS drags that only provide URI payloads.
    const uriList = dataTransfer.getData('text/uri-list');
    if (uriList) {
      const uriLines = uriList
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith('#'));

      for (const uri of uriLines) {
        if (!uri.toLowerCase().startsWith('file://')) {
          continue;
        }
        const filePath = fileUriToFsPath(uri);
        if (filePath) {
          filePaths.add(filePath);
        }
      }
    }

    // Additional fallback used by some Windows explorer drags.
    const plainTextData = dataTransfer.getData('text/plain');
    if (plainTextData) {
      const plainLines = plainTextData
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);

      for (const line of plainLines) {
        if (line.toLowerCase().startsWith('file://')) {
          const filePath = fileUriToFsPath(line);
          if (filePath) {
            filePaths.add(filePath);
          }
          continue;
        }

        if (/^[a-zA-Z]:\\/.test(line) || /^\\\\/.test(line)) {
          filePaths.add(line);
        }
      }
    }

    // Chromium sometimes provides "DownloadURL" payloads: mime:name:url
    const downloadUrlData = dataTransfer.getData('DownloadURL');
    if (downloadUrlData) {
      const parts = downloadUrlData.split(':');
      const maybeUrl = parts.slice(2).join(':').trim();
      if (maybeUrl.toLowerCase().startsWith('file://')) {
        const filePath = fileUriToFsPath(maybeUrl);
        if (filePath) {
          filePaths.add(filePath);
        }
      }
    }

    return [...filePaths];
  }

  window.Glaux.TreePanel = {
    create: createTreePanel,
    getParentFolderPath,
    validateEntryName,
    canOfferOverwrite,
    runWithOptionalOverwrite,
    flattenFolderPaths,
    isInvalidMoveTarget,
    listExternalFilePaths,
    resolveDroppedFilePath,
    fileUriToFsPath,
  };
})();
