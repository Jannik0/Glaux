// Outputs side panel: file tree backed by shared/treePanel.js. Outputs only
// supports internal moves (no upload, no "new document") plus exporting chat
// messages and importing files into Resources (handled from resources.js).

const outputsTreeEl = document.getElementById('outputs-tree');
const outputsStatusEl = document.getElementById('outputs-status');
const outputsPanelEl = document.getElementById('outputs-panel');
const outputsContextMenuEl = document.getElementById('outputs-context-menu');
const outputsEmptyEl = document.getElementById('outputs-empty');

const outputsPanel = window.Glaux.TreePanel.create({
  panelName: 'outputs',
  api: window.api && window.api.outputs,
  supportsCreateDocument: false,
  dataTransferType: 'text/x-output-path',
  dragEffectAllowed: 'copy',
  onlyAcceptsOwnDrag: true,
  elements: {
    treeEl: outputsTreeEl,
    statusEl: outputsStatusEl,
    panelEl: outputsPanelEl,
    contextMenuEl: outputsContextMenuEl,
  },
  renderEmptyState(hasTree, hasItems) {
    if (!hasTree) {
      outputsTreeEl?.classList.add('hidden');
      outputsEmptyEl?.classList.remove('hidden');
      return;
    }
    outputsTreeEl?.classList.toggle('hidden', !hasItems);
    outputsEmptyEl?.classList.toggle('hidden', hasItems);
  },
  onRenderComplete: () => updateMessageInputHighlight(),
  closePeerContextMenu: () => window.Glaux.Resources.panel.closeContextMenu(),
  clearPeerSelection: () => window.Glaux.Resources.panel.clearSelection(),
  // Outputs never accepts a drop from Resources or the OS — dragover simply
  // won't preventDefault unless this panel's own item is being dragged
  // (see `onlyAcceptsOwnDrag` above), so no peer/external hooks are needed.
  onDragEndExtra: () => window.Glaux.Resources.panel.clearDropHighlight(),
});

window.Glaux.Outputs = { panel: outputsPanel };
window.Glaux.MediaKinds.registerStatusSetter('outputs', (message, isError) =>
  outputsPanel.setStatus(message, isError)
);

function initializeOutputsPanel() {
  if (!(window.api && window.api.outputs) || !outputsTreeEl) {
    return;
  }

  outputsPanelEl?.addEventListener('mousedown', () => {
    activePanel = 'outputs';
  });

  const outputsTreeAreaEl = document.getElementById('outputs-tree-area');
  outputsTreeAreaEl?.addEventListener('dragover', (event) => {
    if (!outputsPanel.draggedPath) return;
    if (event.target instanceof Element && event.target.closest('.tree-row')) return;
    event.preventDefault();
    event.stopPropagation();
    if (
      event.target instanceof Element &&
      outputsTreeEl &&
      outputsTreeEl.contains(event.target) &&
      !event.target.classList.contains('tree-root-drop-spacer')
    ) {
      event.dataTransfer.dropEffect = 'move';
      return;
    }
    outputsPanel.setActiveDropTarget(null);
    event.dataTransfer.dropEffect = 'move';
  });

  outputsTreeAreaEl?.addEventListener('drop', async (event) => {
    if (event.target instanceof Element && event.target.closest('.tree-row')) return;
    event.preventDefault();
    event.stopPropagation();
    outputsPanel.clearDropHighlight();
    await outputsPanel.resolveDrop(event, '');
  });

  outputsTreeEl.addEventListener('click', () => {
    outputsPanel.closeContextMenu();
  });

  outputsPanel.initPanelChrome();

  outputsPanel.refreshTree();
}

function shouldSuspendOutputsPolling() {
  return (
    outputsPanel.busy ||
    Boolean(outputsPanel.editingEntryPath) ||
    Boolean(outputsPanel.draggedPath) ||
    Boolean(outputsContextMenuEl && !outputsContextMenuEl.classList.contains('hidden')) ||
    window.Glaux.Dialogs.isConfirmDialogActive() ||
    window.Glaux.Dialogs.isNameDialogActive() ||
    window.Glaux.Dialogs.isProgressModalOpen()
  );
}
