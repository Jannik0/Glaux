// Boot glue: side-panel collapse/poll chrome, the global dialog input trap,
// and the startup sequence (engine bootstrap + panel initialization). Every
// other script has already run by the time this file executes.

const SIDE_PANEL_COLLAPSED_WIDTH = '32px';

function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

/** @type {{ models: boolean, sessions: boolean, resources: boolean, outputs: boolean }} */
const sidePanelCollapsed = {
  models: false,
  sessions: false,
  resources: false,
  outputs: false,
};

const sidePanelConfig = [
  { key: 'models', el: () => modelsPanelEl },
  { key: 'sessions', el: () => sessionsPanelEl },
  { key: 'resources', el: () => resourcesPanelEl },
  { key: 'outputs', el: () => outputsPanelEl },
];

/**
 * @param {{ models?: boolean, sessions?: boolean, resources?: boolean, outputs?: boolean } | null | undefined} panels
 */
function applySidePanelCollapsedPreference(panels) {
  sidePanelCollapsed.models = Boolean(panels && panels.models);
  sidePanelCollapsed.sessions = Boolean(panels && panels.sessions);
  sidePanelCollapsed.resources = Boolean(panels && panels.resources);
  sidePanelCollapsed.outputs = Boolean(panels && panels.outputs);
}

function saveSidePanelCollapsedState() {
  if (!(window.api && typeof window.api.updatePreferences === 'function')) {
    return;
  }
  void window.api.updatePreferences({
    sidePanelsCollapsed: {
      models: sidePanelCollapsed.models,
      sessions: sidePanelCollapsed.sessions,
      resources: sidePanelCollapsed.resources,
      outputs: sidePanelCollapsed.outputs,
    },
  });
}

function getSidePanelExpandedWidth() {
  return window.matchMedia('(max-width: 1400px)').matches ? '200px' : '300px';
}

function applySidePanelLayout() {
  const appBody = document.querySelector('.app-body');
  if (!appBody) {
    return;
  }
  for (const { key, el } of sidePanelConfig) {
    const panelEl = el();
    const collapsed = sidePanelCollapsed[key];
    const label = t(`panels.${key}.title`);
    if (panelEl) {
      panelEl.classList.toggle('is-collapsed', collapsed);
    }
    appBody.style.setProperty(
      `--panel-${key}-width`,
      collapsed ? SIDE_PANEL_COLLAPSED_WIDTH : getSidePanelExpandedWidth(),
    );
    const btn = panelEl?.querySelector('.side-panel-collapse-btn');
    if (btn) {
      const collapseIcon = btn.dataset.collapseIcon || '‹';
      const expandIcon = btn.dataset.expandIcon || '›';
      btn.textContent = collapsed ? expandIcon : collapseIcon;
      btn.setAttribute('aria-expanded', String(!collapsed));
      btn.setAttribute(
        'aria-label',
        collapsed ? t('panels.expandAria', { label }) : t('panels.collapseAria', { label }),
      );
      btn.title = collapsed ? t('panels.expand') : t('panels.collapse');
    }
  }
}

/**
 * @param {'models' | 'sessions' | 'resources' | 'outputs'} key
 */
function toggleSidePanel(key) {
  if (!Object.prototype.hasOwnProperty.call(sidePanelCollapsed, key)) {
    return;
  }
  sidePanelCollapsed[key] = !sidePanelCollapsed[key];
  saveSidePanelCollapsedState();
  applySidePanelLayout();
}

function initializeSidePanelCollapse() {
  applySidePanelLayout();
  window.matchMedia('(max-width: 1400px)').addEventListener('change', applySidePanelLayout);
  document.querySelectorAll('.side-panel-collapse-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.panel;
      if (
        key === 'models' ||
        key === 'sessions' ||
        key === 'resources' ||
        key === 'outputs'
      ) {
        toggleSidePanel(key);
      }
    });
  });
}

const SIDE_PANEL_POLL_MS = 5000;

function tickSidePanelPoll() {
  if (document.hidden) {
    return;
  }
  if (!shouldSuspendResourcesPolling()) {
    void window.Glaux.Resources.panel.refreshTree();
  }
  if (!shouldSuspendOutputsPolling()) {
    void window.Glaux.Outputs.panel.refreshTree();
  }
}

function startSidePanelPolling() {
  setInterval(tickSidePanelPoll, SIDE_PANEL_POLL_MS);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) tickSidePanelPoll();
  });
}

function initializeThemeSelect() {
  const select = document.getElementById('theme-select');
  if (!(select instanceof HTMLSelectElement)) {
    return;
  }
  let selected =
    window.api && typeof window.api.getThemePreference === 'function'
      ? window.api.getThemePreference()
      : 'system';
  if (selected !== 'light' && selected !== 'dark' && selected !== 'system') {
    selected = 'system';
  }
  select.value = selected;
  select.addEventListener('change', () => {
    const next = select.value;
    if (next !== 'system' && next !== 'light' && next !== 'dark') {
      return;
    }
    if (next === selected) {
      return;
    }
    selected = next;
    if (window.api && typeof window.api.updatePreferences === 'function') {
      void window.api.updatePreferences({ theme: next });
    }
  });
}

function initializeLanguageSelect() {
  const select = document.getElementById('language-select');
  const i18n = window.Glaux && window.Glaux.i18n;
  if (!(select instanceof HTMLSelectElement) || !i18n) {
    return;
  }
  select.textContent = '';
  for (const lang of i18n.available) {
    const option = document.createElement('option');
    option.value = lang.code;
    option.textContent = lang.nativeName;
    select.appendChild(option);
  }
  select.value = i18n.language;
  select.addEventListener('change', () => {
    const next = select.value;
    if (!next || next === i18n.language) {
      return;
    }
    if (window.api && typeof window.api.updatePreferences === 'function') {
      void window.api.updatePreferences({ language: next });
    }
  });
}

/**
 * Global chrome that spans both tree panels and the chat context menu: clicking
 * outside a panel clears its selection, right-clicking anywhere closes any open
 * context menu, and losing window focus closes menus too.
 */
function initializeGlobalPanelChrome() {
  document.addEventListener('click', (event) => {
    window.Glaux.Resources.panel.closeContextMenu();
    window.Glaux.Outputs.panel.closeContextMenu();
    closeChatContextMenu();

    if (!(event.target instanceof Element)) {
      window.Glaux.Resources.panel.clearSelection();
      window.Glaux.Outputs.panel.clearSelection();
      return;
    }

    if (
      event.target.closest('#confirm-overlay') ||
      event.target.closest('#repo-check-overlay') ||
      event.target.closest('#model-load-overlay') ||
      event.target.closest('#gguf-variant-overlay') ||
      event.target.closest('#app-progress-overlay')
    ) {
      return;
    }

    const inResourcesPanel = event.target.closest('#resources-panel');
    const inOutputsPanel = event.target.closest('#outputs-panel');

    if (inResourcesPanel) {
      window.Glaux.Outputs.panel.clearSelection();
      if (!event.target.closest('.tree-row') && !event.target.closest('.tree-rename-input')) {
        window.Glaux.Resources.panel.clearSelection();
      }
      return;
    }

    if (inOutputsPanel) {
      window.Glaux.Resources.panel.clearSelection();
      if (!event.target.closest('.tree-row') && !event.target.closest('.tree-rename-input')) {
        window.Glaux.Outputs.panel.clearSelection();
      }
      return;
    }

    window.Glaux.Resources.panel.clearSelection();
    window.Glaux.Outputs.panel.clearSelection();
  });

  document.addEventListener(
    'contextmenu',
    () => {
      window.Glaux.Resources.panel.closeContextMenu();
      window.Glaux.Outputs.panel.closeContextMenu();
      closeChatContextMenu();
    },
    true
  );

  window.addEventListener('blur', () => {
    window.Glaux.Resources.panel.closeContextMenu();
    closeChatContextMenu();
  });
}

/**
 * Traps keyboard/mouse/drag input to whichever modal is open (confirm, repo
 * check, model load, GGUF variant picker, or progress) so the rest of the app
 * is inert behind it.
 */
function initializeGlobalDialogInputTrap() {
  const { Dialogs } = window.Glaux;
  const {
    confirmOverlayEl,
    confirmCancelEl,
    confirmAltEl,
    confirmConfirmEl,
    nameOverlayEl,
    nameInputEl,
    progressOverlayEl,
  } = Dialogs.elements;

  function isProgressLikeModalOpen() {
    return (
      Boolean(progressOverlayEl && !progressOverlayEl.classList.contains('hidden')) ||
      (typeof isRepoCheckModalActive === 'function' && isRepoCheckModalActive()) ||
      (typeof isModelLoadModalActive === 'function' && isModelLoadModalActive())
    );
  }

  function isBlockingDialogOpen() {
    return Dialogs.isConfirmDialogActive() || Dialogs.isNameDialogActive() || isProgressLikeModalOpen();
  }

  document.addEventListener(
    'keydown',
    (event) => {
      if (typeof isRepoCheckModalActive === 'function' && isRepoCheckModalActive()) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (typeof isModelLoadModalActive === 'function' && isModelLoadModalActive()) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (isGgufVariantDialogActive()) {
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          settleGgufVariantDialog(null);
          return;
        }
        if (event.key === 'Enter') {
          event.preventDefault();
          event.stopPropagation();
          const value = ggufVariantSelectEl ? ggufVariantSelectEl.value : null;
          settleGgufVariantDialog(value || null);
          return;
        }
        // Allow typing/arrows in the select.
        if (event.target === ggufVariantSelectEl) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (Dialogs.isNameDialogActive()) {
        if (event.target === nameInputEl) {
          return;
        }
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          Dialogs.closeNameDialog(null);
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (!Dialogs.isConfirmDialogActive() && !isProgressLikeModalOpen()) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    },
    true
  );
  document.addEventListener(
    'keypress',
    (event) => {
      if (Dialogs.isNameDialogActive() && event.target === nameInputEl) {
        return;
      }
      if (!isBlockingDialogOpen()) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    },
    true
  );
  document.addEventListener(
    'keyup',
    (event) => {
      if (Dialogs.isNameDialogActive() && event.target === nameInputEl) {
        return;
      }
      if (!isBlockingDialogOpen()) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    },
    true
  );
  document.addEventListener(
    'focusin',
    (event) => {
      if (Dialogs.isNameDialogActive()) {
        if (
          event.target !== nameInputEl &&
          event.target !== Dialogs.elements.nameCancelEl &&
          event.target !== Dialogs.elements.nameConfirmEl
        ) {
          nameInputEl?.focus();
        }
        return;
      }
      if (!Dialogs.isConfirmDialogActive()) {
        return;
      }
      if (
        event.target === confirmCancelEl ||
        event.target === confirmAltEl ||
        event.target === confirmConfirmEl
      ) {
        if (document.activeElement instanceof HTMLElement) {
          document.activeElement.blur();
        }
      }
    },
    true
  );
  document.addEventListener(
    'mousedown',
    (event) => {
      const isProgressOpen = isProgressLikeModalOpen();
      const isGgufOpen = Boolean(
        ggufVariantOverlayEl && !ggufVariantOverlayEl.classList.contains('hidden')
      );
      const isRepoCheckOpen =
        typeof isRepoCheckModalActive === 'function' && isRepoCheckModalActive();
      const isModelLoadOpen =
        typeof isModelLoadModalActive === 'function' && isModelLoadModalActive();
      const isNameOpen = Dialogs.isNameDialogActive();
      if (!Dialogs.isConfirmDialogActive() && !isProgressOpen && !isGgufOpen && !isNameOpen) {
        return;
      }
      const activeOverlay = Dialogs.isConfirmDialogActive()
        ? confirmOverlayEl
        : isNameOpen
          ? nameOverlayEl
          : isRepoCheckOpen
            ? repoCheckOverlayEl
            : isModelLoadOpen
              ? modelLoadOverlayEl
              : isGgufOpen
                ? ggufVariantOverlayEl
                : progressOverlayEl;
      if (activeOverlay && !activeOverlay.contains(event.target)) {
        event.preventDefault();
        event.stopPropagation();
      }
    },
    true
  );
  document.addEventListener(
    'click',
    (event) => {
      const isProgressOpen = isProgressLikeModalOpen();
      const isNameOpen = Dialogs.isNameDialogActive();
      if (!Dialogs.isConfirmDialogActive() && !isProgressOpen && !isNameOpen) {
        return;
      }
      const activeOverlay = Dialogs.isConfirmDialogActive()
        ? confirmOverlayEl
        : isNameOpen
          ? nameOverlayEl
          : typeof isRepoCheckModalActive === 'function' && isRepoCheckModalActive()
            ? repoCheckOverlayEl
            : typeof isModelLoadModalActive === 'function' && isModelLoadModalActive()
              ? modelLoadOverlayEl
              : progressOverlayEl;
      if (activeOverlay && !activeOverlay.contains(event.target)) {
        event.preventDefault();
        event.stopPropagation();
      }
    },
    true
  );
  document.addEventListener(
    'contextmenu',
    (event) => {
      const isProgressOpen = isProgressLikeModalOpen();
      const isNameOpen = Dialogs.isNameDialogActive();
      if (!Dialogs.isConfirmDialogActive() && !isProgressOpen && !isNameOpen) {
        return;
      }
      const activeOverlay = Dialogs.isConfirmDialogActive()
        ? confirmOverlayEl
        : isNameOpen
          ? nameOverlayEl
          : typeof isRepoCheckModalActive === 'function' && isRepoCheckModalActive()
            ? repoCheckOverlayEl
            : typeof isModelLoadModalActive === 'function' && isModelLoadModalActive()
              ? modelLoadOverlayEl
              : progressOverlayEl;
      if (activeOverlay && !activeOverlay.contains(event.target)) {
        event.preventDefault();
        event.stopPropagation();
      }
    },
    true
  );
  document.addEventListener(
    'dragstart',
    (event) => {
      if (!isBlockingDialogOpen()) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    },
    true
  );
  document.addEventListener(
    'drop',
    (event) => {
      if (!isBlockingDialogOpen()) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    },
    true
  );
  document.addEventListener(
    'submit',
    (event) => {
      if (!isBlockingDialogOpen()) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    },
    true
  );
}

initializeGlobalPanelChrome();
initializeGlobalDialogInputTrap();
initializeThemeSelect();
initializeLanguageSelect();

void (async () => {
  if (typeof window.api.onInitProgress === 'function') {
    window.api.onInitProgress((info) => {
      if (!info || typeof info !== 'object' || !info.status) return;
      if (info.status === 'loadStage' && typeof info.message === 'string' && info.message.trim()) {
        setEngineLoading(true);
        modelProgressHeading = info.message.trim();
        showModelLoadModal({ message: modelProgressHeading });
        return;
      }
      if (info.status === 'loadProgress') {
        setEngineLoading(true);
        const percent =
          typeof info.percent === 'number' && Number.isFinite(info.percent)
            ? Math.max(0, Math.min(100, Math.round(info.percent)))
            : null;
        const rawMessage =
          typeof info.message === 'string' && info.message.trim()
            ? info.message.trim()
            : t('chat.loadingEllipsis');
        // Drop trailing " N%" from the status text; the bar shows percent.
        const message = rawMessage.replace(/\s+\d+%\s*$/, '');
        modelProgressHeading = message;
        updateModelLoadModal({
          message,
          ...(percent !== null ? { percent } : {}),
        });
        return;
      }
      if (info.status === 'downloadStart') {
        modelProgressHeading = t('chat.downloadingModel');
        setLoadingStatusMessage(modelProgressHeading);
        return;
      }
      if (info.status === 'downloadDone') {
        setEngineLoading(true);
        modelProgressHeading = t('chat.loadingModel');
        void refreshModelsCacheList();
        showModelLoadModal({
          message: info.modelId ? t('chat.loadingNamed', { modelId: info.modelId }) : t('chat.loadingEllipsis'),
        });
        return;
      }
      if (info.status === 'ready') {
        setEngineLoading(false);
        hideModelLoadModal();
        modelProgressHeading = t('chat.modelReady');
        void refreshModelsCacheList();
        void (async () => {
          let message = t('chat.modelReady');
          if (typeof window.api.getStatus === 'function') {
            try {
              await syncActiveEnginePipelineTag();
              const status = await window.api.getStatus();
              applyReasoningSupportFromStatus(status);
              applyChatTemplateSupportFromStatus(status);
              message = formatModelReadyStatusMessage();
            } catch {
              /* keep default */
            }
          }
          setLoadingStatusMessage(message);
        })();
        return;
      }
      if (info.status === 'ejected') {
        setEngineLoading(false);
        hideModelLoadModal();
        setReady(false);
        chatTemplateSupported = false;
        setContextUsageLabel('');
        activeEnginePipelineTag = null;
        modelsPanelSelectedId = null;
        showNoModelLoadedMessage();
        void refreshModelsCacheList();
        void pollStatus();
        return;
      }
      if (info.status === 'error') {
        setEngineLoading(false);
        hideModelLoadModal();
        const wasDownloading = modelProgressHeading === t('chat.downloadingModel');
        if (wasDownloading) {
          setLoadingStatusMessage(
            `${t('chat.downloadFailed')}: ${info.message || t('chat.unknownErrorLower')}`,
          );
        } else {
          setLoadingStatusMessage(
            t('chat.failedToLoadModel', { message: info.message || t('chat.unknownErrorLower') }),
          );
        }
        if (info.clearPreference) {
          modelsPanelSelectedId = null;
          void refreshModelsCacheList();
        }
      }
    });
  }

  initializeWorkspacesPanel();
  initializeResourcesPanel();
  initializeOutputsPanel();
  initializeModelsPanel();
  initializeSessionsPanel();
  initializeChatMessageContextMenu();
  startSidePanelPolling();
  initializeDropReject(messageInputShellEl);
  initializeChatAttachmentsDrop();
  renderChatAttachments();

  pollStatus();

  void (async () => {
    if (window.api && typeof window.api.getPreferences === 'function') {
      try {
        const preferences = await window.api.getPreferences();
        applySidePanelCollapsedPreference(preferences.sidePanelsCollapsed);
        applyReasoningEnabledPreference(preferences.reasoningEnabled);
        applyResubmitEnabledPreference(preferences.resubmitEnabled);
      } catch {
        /* keep defaults */
      }
    }

    initializeReasoningToggle();
    initializeResubmitToggle();
    initializeSidePanelCollapse();

    if (!(window.api && typeof window.api.bootstrapEngine === 'function')) {
      return;
    }
    try {
      const { modelId, clearedPreference, loadFailed, message, pending } =
        await window.api.bootstrapEngine();
      if (clearedPreference) {
        modelsPanelSelectedId = null;
      }
      if (loadFailed && message) {
        reportEngineLoadError(message);
      } else if (!modelId) {
        modelsPanelSelectedId = null;
        showNoModelLoadedMessage();
      } else {
        modelsPanelSelectedId = modelId;
        if (pending) {
          setEngineLoading(true);
        }
      }
      void refreshModelsCacheList();
      if (pending) {
        return;
      }
    } catch (err) {
      const msg = err && err.message ? err.message : String(err || t('chat.engineFailedToStart'));
      reportEngineLoadError(msg);
    }
  })();
})();

updateMessageInputHighlight();
