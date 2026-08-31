// Models side panel: cached-model list, download/eject/reload, repository
// check modal, model-load progress modal, and the GGUF-variant picker dialog.
// Uses shared/dialogs.js + shared/status.js and the cross-panel state declared
// in state.js.

function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

const modelsCacheListEl = document.getElementById('models-cache-list');
const modelReloadEl = document.getElementById('model-reload');
const modelEjectEl = document.getElementById('model-eject');
const modelsAddInputEl = document.getElementById('models-add-input');
const modelsAddBtnEl = document.getElementById('models-add');
const modelsPanelEl = document.getElementById('models-panel');

const repoCheckOverlayEl = document.getElementById('repo-check-overlay');
const repoCheckMessageEl = document.getElementById('repo-check-message');
const modelLoadOverlayEl = document.getElementById('model-load-overlay');
const modelLoadMessageEl = document.getElementById('model-load-message');
const modelLoadBarEl = document.getElementById('model-load-bar');
const ggufVariantOverlayEl = document.getElementById('gguf-variant-overlay');
const ggufVariantMessageEl = document.getElementById('gguf-variant-message');
const ggufVariantSelectEl = document.getElementById('gguf-variant-select');
const ggufVariantCancelEl = document.getElementById('gguf-variant-cancel');
const ggufVariantConfirmEl = document.getElementById('gguf-variant-confirm');
/** @type {((value: string | null) => void) | null} */
let ggufVariantResolver = null;

/** True while add-model probe / variant picker / download is in progress. */
let modelAddFlowInFlight = false;

/**
 * Checkbox selection in the models panel. Cleared on eject along with the persisted selection.
 * @type {string | null}
 */
let modelsPanelSelectedId = null;

/** @returns {Promise<string | null>} */
async function getModelsPanelSelectedId() {
  if (!modelReady) {
    return modelsPanelSelectedId;
  }
  const id = await getEffectiveSelectedModelId();
  modelsPanelSelectedId = id;
  return id;
}

async function getEffectiveSelectedModelId() {
  if (window.api && typeof window.api.getSelectedModelId === 'function') {
    try {
      return await window.api.getSelectedModelId();
    } catch {
      return null;
    }
  }
  return null;
}

/** @type {Map<string, { loaded: number, total: number }>} */
const downloadingModels = new Map();

/** @type {Set<string>} */
const downloadStopRequested = new Set();

function isValidHfRepoId(raw) {
  const s = typeof raw === 'string' ? raw.trim() : '';
  return s.length > 0 && s.length <= 512 && /^[\w.-]+\/[\w.-]+$/.test(s);
}

/** @param {number} bytes */
function formatModelDownloadBytesCompact(bytes) {
  const n = Math.max(0, Number(bytes) || 0);
  if (!Number.isFinite(n)) {
    return '?';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let u = 0;
  let v = n;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u += 1;
  }
  const digits = u === 0 ? 0 : v >= 100 ? 0 : v >= 10 ? 1 : 2;
  const num = digits === 0 ? Math.round(v) : Number(v.toFixed(digits));
  return `${num}${units[u]}`;
}

/** @param {number} loaded @param {number} total */
function formatModelDownloadSizeLabel(loaded, total) {
  const loadedStr = formatModelDownloadBytesCompact(loaded);
  if (!total || total <= 0) {
    return `${loadedStr}/…`;
  }
  return `${loadedStr}/${formatModelDownloadBytesCompact(total)}`;
}

function patchModelRowDownloadSize(modelId) {
  if (!modelsCacheListEl) {
    return;
  }
  const esc = typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(modelId) : modelId;
  const row = modelsCacheListEl.querySelector(`.models-cache-row[data-model-id="${esc}"]`);
  if (!row) {
    return;
  }
  const sizeEl = row.querySelector('.models-cache-download-size');
  if (!sizeEl) {
    return;
  }
  const st = downloadingModels.get(modelId);
  const loaded = st?.loaded ?? 0;
  const total = st?.total ?? 0;
  const label = formatModelDownloadSizeLabel(loaded, total);
  sizeEl.textContent = label;
  sizeEl.setAttribute('aria-label', t('panels.models.downloadedAria', { label }));
}

function syncModelPanelDisabled() {
  const switchBlocked = engineBusy || engineLoading;
  const engineActionDis = engineBusy || engineLoading || !modelReady;
  const downloadInFlight = downloadingModels.size > 0 || modelAddFlowInFlight;
  modelsCacheListEl?.querySelectorAll('input[name="model-pick"]').forEach((cb) => {
    const row = cb.closest('.models-cache-row');
    const id = cb.value;
    const isDownloading =
      row?.classList.contains('is-downloading') || downloadingModels.has(id);
    // Allow picking a cached model when idle with no model loaded (first selection).
    cb.disabled = switchBlocked || isDownloading;
  });
  if (modelReloadEl) modelReloadEl.disabled = engineActionDis;
  if (modelEjectEl) modelEjectEl.disabled = engineActionDis;
  if (modelsAddBtnEl) modelsAddBtnEl.disabled = engineBusy || engineLoading || downloadInFlight;
  if (modelsAddInputEl) modelsAddInputEl.disabled = engineBusy || engineLoading || downloadInFlight;
  modelsCacheListEl?.querySelectorAll('.models-cache-trash').forEach((btn) => {
    if (btn instanceof HTMLButtonElement) {
      btn.disabled = engineBusy || engineLoading;
    }
  });
  modelsCacheListEl?.querySelectorAll('.models-cache-stop').forEach((btn) => {
    if (btn instanceof HTMLButtonElement) {
      btn.disabled = engineBusy;
    }
  });
}

const MODEL_TRASH_ICON_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>';

const MODEL_STOP_ICON_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>';

/** @type {Map<string, string | null>} */
const modelPipelineTagsById = new Map();

function normalizeCachedModelEntry(entry) {
  if (typeof entry === 'string') {
    return { id: entry, pipelineTag: null };
  }
  return {
    id: entry.id,
    pipelineTag: entry.pipelineTag ?? null,
  };
}

function cachedModelId(entry) {
  return typeof entry === 'string' ? entry : entry.id;
}

async function renderModelsCacheList(models, selectedId) {
  if (!modelsCacheListEl) {
    return;
  }
  const modelMap = new Map();
  for (const entry of models) {
    const normalized = normalizeCachedModelEntry(entry);
    modelMap.set(normalized.id, normalized.pipelineTag);
    modelPipelineTagsById.set(normalized.id, normalized.pipelineTag);
  }
  for (const id of downloadingModels.keys()) {
    if (!modelMap.has(id)) {
      modelMap.set(id, null);
    }
  }
  const allModels = [...modelMap.entries()].sort((a, b) =>
    a[0].localeCompare(b[0], undefined, { sensitivity: 'base' })
  );

  modelsCacheListEl.innerHTML = '';
  if (!allModels.length) {
    const empty = document.createElement('div');
    empty.className = 'models-cache-empty';
    empty.textContent = t('panels.models.empty');
    modelsCacheListEl.appendChild(empty);
    return;
  }
  for (const [id, pipelineTag] of allModels) {
    const isDownloading = downloadingModels.has(id);
    const row = document.createElement('div');
    row.className = 'models-cache-row';
    if (isDownloading) {
      row.classList.add('is-downloading');
    }
    row.dataset.modelId = id;

    const labelWrap = document.createElement('label');
    labelWrap.className = 'models-cache-row-main';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.name = 'model-pick';
    cb.value = id;
    cb.checked = id === selectedId;
    if (isDownloading) {
      cb.disabled = true;
    }

    const textWrap = document.createElement('span');
    textWrap.className = 'models-cache-row-text';

    const infoWrap = document.createElement('span');
    infoWrap.className = 'models-cache-row-info';

    const span = document.createElement('span');
    span.className = 'models-cache-row-label';
    span.textContent = id;
    infoWrap.appendChild(span);

    if (pipelineTag) {
      const pipelineEl = document.createElement('span');
      pipelineEl.className = 'models-cache-row-pipeline';
      pipelineEl.textContent = pipelineTag;
      infoWrap.appendChild(pipelineEl);
    }

    textWrap.appendChild(infoWrap);

    if (isDownloading) {
      const sizeEl = document.createElement('span');
      sizeEl.className = 'models-cache-download-size';
      sizeEl.setAttribute('aria-live', 'polite');
      sizeEl.textContent = '0B/…';
      textWrap.appendChild(sizeEl);
    }

    labelWrap.appendChild(cb);
    labelWrap.appendChild(textWrap);
    row.appendChild(labelWrap);

    const actionBtn = document.createElement('button');
    actionBtn.type = 'button';
    actionBtn.setAttribute('data-model-id', id);
    if (isDownloading) {
      actionBtn.className = 'models-cache-stop';
      actionBtn.setAttribute('aria-label', t('panels.models.stopDownloadAria', { modelId: id }));
      actionBtn.title = t('panels.models.stopDownload');
      actionBtn.innerHTML = MODEL_STOP_ICON_SVG;
    } else {
      actionBtn.className = 'models-cache-trash';
      actionBtn.setAttribute('aria-label', t('panels.models.deleteAria', { modelId: id }));
      actionBtn.title = t('panels.models.delete');
      actionBtn.innerHTML = MODEL_TRASH_ICON_SVG;
    }
    row.appendChild(actionBtn);

    modelsCacheListEl.appendChild(row);
    if (isDownloading) {
      patchModelRowDownloadSize(id);
    }
  }
  syncModelPanelDisabled();
}

async function handleCancelModelDownload(modelId) {
  if (!(window.api && typeof window.api.cancelModelDownload === 'function')) {
    return;
  }
  downloadStopRequested.add(modelId);
  setLoadingStatusMessage(t('panels.models.stoppingDownload'));
  try {
    await window.api.cancelModelDownload(modelId);
    downloadingModels.delete(modelId);
    await refreshModelsCacheList();
  } catch (err) {
    const msg = err && err.message ? err.message : t('panels.models.couldNotStopDownloadFallback');
    setLoadingStatusMessage(t('panels.models.couldNotStopDownload', { message: msg }));
    downloadingModels.delete(modelId);
    await refreshModelsCacheList();
    downloadStopRequested.delete(modelId);
    return;
  }
  downloadStopRequested.delete(modelId);
  setLoadingStatusMessage('');
  void pollStatus();
}

async function handleMoveCachedModelToTrash(modelId) {
  if (!(window.api && typeof window.api.moveCachedModelToTrash === 'function')) {
    return;
  }
  const confirmed = await window.Glaux.Dialogs.showConfirmDialog({
    title: t('panels.models.deleteConfirmTitle'),
    message: t('panels.models.deleteConfirmMessage', { modelId }),
    confirmLabel: t('panels.models.delete'),
  });
  if (!confirmed) {
    return;
  }
  try {
    const { wasSelected } = await window.api.moveCachedModelToTrash(modelId);
    if (wasSelected) {
      modelsPanelSelectedId = null;
    }
    await refreshModelsCacheList();
    void pollStatus();
  } catch (err) {
    const msg = err && err.message ? err.message : t('panels.models.couldNotMoveToRecycleBin');
    setLoadingStatusMessage(t('panels.models.couldNotDeleteModel', { message: msg }));
    void refreshModelsCacheList();
  }
}

function handleModelDownloadProgress(modelId, event) {
  if (!event || typeof event !== 'object') {
    return;
  }
  if (event.phase === 'download' && event.status === 'starting' && event.modelId === modelId) {
    downloadingModels.set(modelId, { loaded: 0, total: 0 });
    patchModelRowDownloadSize(modelId);
    return;
  }
  const file = typeof event.file === 'string' ? event.file : '';
  if (file && file !== modelId && file !== 'download') {
    return;
  }
  if (event.status === 'initiate') {
    downloadingModels.set(modelId, { loaded: 0, total: 0 });
    patchModelRowDownloadSize(modelId);
    return;
  }
  if (event.status === 'progress') {
    const loaded = Number(event.loaded);
    const total = Number(event.total);
    if (Number.isFinite(loaded) && Number.isFinite(total)) {
      downloadingModels.set(modelId, { loaded, total });
      patchModelRowDownloadSize(modelId);
    }
  }
}

async function runDownloadModelFlow() {
  if (!(window.api && typeof window.api.downloadModel === 'function')) {
    return;
  }
  if (modelAddFlowInFlight) {
    return;
  }
  const raw = modelsAddInputEl ? modelsAddInputEl.value.trim() : '';
  if (!raw) {
    setLoadingStatusMessage(t('panels.models.enterValidModelId'));
    return;
  }
  if (!isValidHfRepoId(raw)) {
    setLoadingStatusMessage(t('panels.models.invalidModelIdFormat'));
    return;
  }
  if (downloadingModels.has(raw)) {
    return;
  }
  let cached = [];
  try {
    cached = await window.api.listCachedModels();
  } catch {
    cached = [];
  }
  if (cached.some((entry) => cachedModelId(entry) === raw)) {
    setLoadingStatusMessage(t('panels.models.alreadyInCache'));
    return;
  }

  modelAddFlowInFlight = true;
  syncModelPanelDisabled();

  /** @type {{ allowPatterns?: string[], ggufVariant?: string }} */
  const downloadOpts = {};
  try {
    if (typeof window.api.probeHubRepo === 'function') {
      showRepoCheckModal(raw);
      try {
        await window.Glaux.Dialogs.waitForUiPaint();
        const probe = await window.api.probeHubRepo(raw);
        hideRepoCheckModal();
        if (probe.kind === 'gguf') {
          if (!probe.variants.length) {
            setLoadingStatusMessage(t('panels.models.noGgufVariants'));
            return;
          }
          const chosenKey = await showGgufVariantDialog({
            modelId: raw,
            variants: probe.variants,
          });
          if (!chosenKey) {
            setLoadingStatusMessage('');
            return;
          }
          const variant = probe.variants.find((v) => v.key === chosenKey);
          downloadOpts.ggufVariant = chosenKey;
          if (variant && Array.isArray(variant.files)) {
            // Prefer server-side rebuild from variant key; still pass files when available.
            downloadOpts.allowPatterns = undefined;
          }
        }
      } catch (err) {
        hideRepoCheckModal();
        const msg = err && err.message ? err.message : t('panels.models.couldNotInspectRepo');
        setLoadingStatusMessage(t('panels.models.downloadFailed', { message: msg }));
        return;
      }
    }

    downloadingModels.set(raw, { loaded: 0, total: 0 });
    setLoadingStatusMessage('');
    await renderModelsCacheList(cached, await getModelsPanelSelectedId());
    syncModelPanelDisabled();
    try {
      await window.api.downloadModel(raw, {
        onProgress: (event) => handleModelDownloadProgress(raw, event),
        ggufVariant: downloadOpts.ggufVariant,
        allowPatterns: downloadOpts.allowPatterns,
      });
      downloadingModels.delete(raw);
      if (modelsAddInputEl) {
        modelsAddInputEl.value = '';
      }
      await refreshModelsCacheList();
    } catch (err) {
      downloadingModels.delete(raw);
      const stopped =
        downloadStopRequested.has(raw) ||
        (err &&
          (err.code === 'E_DOWNLOAD_CANCELLED' || /cancelled/i.test(String(err.message || ''))));
      if (stopped) {
        setLoadingStatusMessage('');
        void pollStatus();
        await refreshModelsCacheList();
        return;
      }
      const msg = err && err.message ? err.message : t('panels.models.downloadFailedFallback');
      setLoadingStatusMessage(t('panels.models.downloadFailed', { message: msg }));
      await refreshModelsCacheList();
    }
  } finally {
    hideRepoCheckModal();
    modelAddFlowInFlight = false;
    syncModelPanelDisabled();
  }
}

/**
 * @param {string} modelId
 */
function showRepoCheckModal(modelId) {
  if (!repoCheckOverlayEl) {
    return;
  }
  if (repoCheckMessageEl) {
    repoCheckMessageEl.textContent = t('panels.models.inspectingFiles', { modelId });
  }
  repoCheckOverlayEl.classList.remove('hidden');
  if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur();
  }
}

function hideRepoCheckModal() {
  if (repoCheckOverlayEl) {
    repoCheckOverlayEl.classList.add('hidden');
  }
}

function isRepoCheckModalActive() {
  return Boolean(repoCheckOverlayEl && !repoCheckOverlayEl.classList.contains('hidden'));
}

/**
 * @param {{ message?: string, percent?: number }} [opts]
 */
function showModelLoadModal(opts = {}) {
  if (!modelLoadOverlayEl) {
    return;
  }
  const message =
    typeof opts.message === 'string' && opts.message.trim()
      ? opts.message.trim()
      : t('panels.models.loadingEllipsis');
  if (modelLoadMessageEl) {
    modelLoadMessageEl.textContent = message;
  }
  if (modelLoadBarEl) {
    const rawPercent = Number(opts.percent);
    if (Number.isFinite(rawPercent)) {
      const percent = Math.max(0, Math.min(100, Math.round(rawPercent)));
      modelLoadBarEl.classList.remove('is-indeterminate');
      modelLoadBarEl.style.width = `${percent}%`;
    } else {
      modelLoadBarEl.classList.add('is-indeterminate');
      modelLoadBarEl.style.width = '';
    }
  }
  modelLoadOverlayEl.classList.remove('hidden');
  if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur();
  }
}

/**
 * @param {{ message?: string, percent?: number }} [opts]
 */
function updateModelLoadModal(opts = {}) {
  if (!modelLoadOverlayEl || modelLoadOverlayEl.classList.contains('hidden')) {
    showModelLoadModal(opts);
    return;
  }
  if (typeof opts.message === 'string' && modelLoadMessageEl) {
    modelLoadMessageEl.textContent = opts.message;
  }
  if (modelLoadBarEl && typeof opts.percent === 'number' && Number.isFinite(opts.percent)) {
    const percent = Math.max(0, Math.min(100, Math.round(opts.percent)));
    modelLoadBarEl.classList.remove('is-indeterminate');
    modelLoadBarEl.style.width = `${percent}%`;
  }
}

function hideModelLoadModal() {
  if (modelLoadOverlayEl) {
    modelLoadOverlayEl.classList.add('hidden');
  }
  if (modelLoadBarEl) {
    modelLoadBarEl.classList.add('is-indeterminate');
    modelLoadBarEl.style.width = '';
  }
}

function isModelLoadModalActive() {
  return Boolean(modelLoadOverlayEl && !modelLoadOverlayEl.classList.contains('hidden'));
}

/**
 * @param {{ modelId: string, variants: Array<{ key: string, label: string, files?: string[], size?: number }> }} opts
 * @returns {Promise<string | null>} Selected variant key, or null if cancelled.
 */
function showGgufVariantDialog({ modelId, variants }) {
  if (
    !ggufVariantOverlayEl ||
    !ggufVariantMessageEl ||
    !ggufVariantSelectEl ||
    !ggufVariantCancelEl ||
    !ggufVariantConfirmEl
  ) {
    return Promise.resolve(variants[0] ? variants[0].key : null);
  }

  ggufVariantMessageEl.textContent = t('panels.models.ggufVariantMessage', { modelId });
  ggufVariantSelectEl.innerHTML = '';
  for (const variant of variants) {
    const option = document.createElement('option');
    option.value = variant.key;
    option.textContent = variant.label || variant.key;
    ggufVariantSelectEl.appendChild(option);
  }
  // Prefer a mid-size default when present.
  const preferred = ['Q4_K_M', 'Q5_K_M', 'Q4_K_S', 'Q6_K', 'Q8_0'];
  for (const key of preferred) {
    if (variants.some((v) => v.key === key)) {
      ggufVariantSelectEl.value = key;
      break;
    }
  }
  ggufVariantOverlayEl.classList.remove('hidden');
  if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur();
  }
  ggufVariantSelectEl.focus();

  return new Promise((resolve) => {
    ggufVariantResolver = resolve;
  });
}

function settleGgufVariantDialog(value) {
  if (ggufVariantOverlayEl) {
    ggufVariantOverlayEl.classList.add('hidden');
  }
  const resolve = ggufVariantResolver;
  ggufVariantResolver = null;
  if (resolve) {
    resolve(value);
  }
}

function isGgufVariantDialogActive() {
  return Boolean(ggufVariantOverlayEl && !ggufVariantOverlayEl.classList.contains('hidden'));
}

async function refreshModelsCacheList() {
  if (!(window.api && typeof window.api.listCachedModels === 'function') || !modelsCacheListEl) {
    return;
  }
  try {
    const models = await window.api.listCachedModels();
    const selectedId = await getModelsPanelSelectedId();
    await renderModelsCacheList(models, selectedId);
  } catch (_err) {
    modelsCacheListEl.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'models-cache-empty';
    empty.textContent = t('panels.models.cacheReadError');
    modelsCacheListEl.appendChild(empty);
  }
}

async function applySelectedCachedModel(modelId) {
  if (engineBusy || engineLoading) {
    await refreshModelsCacheList();
    return;
  }
  if (downloadingModels.has(modelId)) {
    setLoadingStatusMessage(t('panels.models.waitUntilDownloaded'));
    await refreshModelsCacheList();
    return;
  }
  setEngineLoading(true);
  setReady(false);
  modelsPanelSelectedId = modelId;
  modelProgressHeading = t('panels.models.loadingModelTitle');
  showModelLoadModal({ message: t('panels.models.loadingNamed', { modelId }) });
  try {
    await window.api.setSelectedModelId(modelId);
    const { clearedPreference } = await window.api.reinitializeEngine();
    if (clearedPreference) {
      modelsPanelSelectedId = null;
    }
    await refreshModelsCacheList();
    void pollStatus();
  } catch (err) {
    const msg = err && err.message ? err.message : t('panels.models.failedToSwitchModel');
    reportEngineLoadError(msg);
    void refreshModelsCacheList();
    void pollStatus();
  }
}

function initializeModelsPanel() {
  initializeDropReject(modelsAddInputEl);

  modelsPanelEl?.addEventListener('mousedown', () => {
    activePanel = 'models';
  });

  if (!(window.api && typeof window.api.bootstrapEngine === 'function') || !modelsCacheListEl) {
    syncModelPanelDisabled();
    return;
  }

  void refreshModelsCacheList();

  modelsCacheListEl.addEventListener('change', (e) => {
    const input = e.target;
    if (!(input instanceof HTMLInputElement) || input.name !== 'model-pick') {
      return;
    }
    if (engineBusy || engineLoading) {
      input.checked = input.value === modelsPanelSelectedId;
      return;
    }
    if (downloadingModels.has(input.value)) {
      input.checked = false;
      return;
    }
    if (!input.checked) {
      input.checked = true;
      return;
    }
    modelsCacheListEl.querySelectorAll('input[name="model-pick"]').forEach((cb) => {
      if (cb !== input) cb.checked = false;
    });
    void applySelectedCachedModel(input.value);
  });

  modelsCacheListEl.addEventListener('click', (e) => {
    const btn =
      e.target &&
      /** @type {HTMLElement} */ (e.target).closest('.models-cache-trash, .models-cache-stop');
    if (!(btn instanceof HTMLButtonElement)) {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    const id = btn.getAttribute('data-model-id');
    if (!id) {
      return;
    }
    if (btn.classList.contains('models-cache-stop')) {
      void handleCancelModelDownload(id);
      return;
    }
    void handleMoveCachedModelToTrash(id);
  });

  modelReloadEl?.addEventListener('click', async () => {
    if (engineBusy || engineLoading) {
      return;
    }
    setEngineLoading(true);
    setReady(false);
    modelProgressHeading = t('panels.models.loadingModelTitle');
    const reloadId = modelsPanelSelectedId || (await getEffectiveSelectedModelId());
    showModelLoadModal({
      message: reloadId
        ? t('panels.models.loadingNamed', { modelId: reloadId })
        : t('panels.models.loadingEllipsis'),
    });
    try {
      const { clearedPreference } = await window.api.reinitializeEngine();
      if (clearedPreference) {
        modelsPanelSelectedId = null;
      }
      await refreshModelsCacheList();
      void pollStatus();
    } catch (err) {
      const msg = err && err.message ? err.message : t('panels.models.reloadFailed');
      reportEngineLoadError(msg);
      void pollStatus();
    }
  });

  modelEjectEl?.addEventListener('click', async () => {
    try {
      await window.api.ejectEngine();
      setReady(false);
      modelsPanelSelectedId = null;
      showNoModelLoadedMessage();
      await refreshModelsCacheList();
      void pollStatus();
    } catch (err) {
      const msg = err && err.message ? err.message : t('panels.models.ejectFailedFallback');
      setLoadingStatusMessage(t('panels.models.ejectFailed', { message: msg }));
      void pollStatus();
    }
  });

  modelsAddBtnEl?.addEventListener('click', () => {
    void runDownloadModelFlow();
  });

  modelsAddInputEl?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      void runDownloadModelFlow();
    }
  });

  ggufVariantCancelEl?.addEventListener('mousedown', (event) => {
    event.preventDefault();
  });
  ggufVariantConfirmEl?.addEventListener('mousedown', (event) => {
    event.preventDefault();
  });
  ggufVariantCancelEl?.addEventListener('click', () => {
    settleGgufVariantDialog(null);
  });
  ggufVariantConfirmEl?.addEventListener('click', () => {
    const value = ggufVariantSelectEl ? ggufVariantSelectEl.value : null;
    settleGgufVariantDialog(value || null);
  });

  syncModelPanelDisabled();
}
