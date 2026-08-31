function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

const titleEl = document.getElementById('editor-title');
const statusEl = document.getElementById('editor-status');
const saveBtn = document.getElementById('save-btn');
const previewToggleBtn = document.getElementById('preview-toggle-btn');
const toolbarEl = document.getElementById('editor-toolbar');
const loadingEl = document.getElementById('editor-loading');
const mainEl = document.getElementById('editor-main');
const previewPaneEl = document.getElementById('preview-pane');
const textareaEl = document.getElementById('editor-textarea');
const previewEl = document.getElementById('editor-preview');
const confirmOverlayEl = document.getElementById('confirm-overlay');
const confirmTitleEl = document.getElementById('confirm-title');
const confirmMessageEl = document.getElementById('confirm-message');
const confirmCancelEl = document.getElementById('confirm-cancel');
const confirmAltEl = document.getElementById('confirm-alt');
const confirmConfirmEl = document.getElementById('confirm-confirm');

let savedContent = '';
let isDirty = false;
let isSaving = false;
let previewVisible = true;
let fileName = 'document.md';
let confirmResolver = null;
let confirmChoiceIds = { cancel: 'cancel', alt: 'alt', confirm: 'confirm' };

function setStatus(message, type = '') {
  statusEl.textContent = message || '';
  statusEl.className = 'editor-status';
  if (type) {
    statusEl.classList.add(type);
  }
}

function updateDirtyState() {
  isDirty = textareaEl.value !== savedContent;
  titleEl.classList.toggle('is-dirty', isDirty);
  if (!isSaving) {
    setStatus(isDirty ? t('editor.unsavedChanges') : '');
  }
}

function renderPreview() {
  const text = textareaEl.value;
  previewEl.textContent = '';
  if (!text) {
    return;
  }
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
    previewEl.textContent = text;
    return;
  }
  try {
    const html = marked.parse(text, {
      async: false,
      gfm: true,
      breaks: true,
    });
    previewEl.innerHTML = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
  } catch (err) {
    previewEl.textContent = text;
  }
}

function setPreviewVisible(visible) {
  previewVisible = visible;
  previewPaneEl.classList.toggle('hidden', !visible);
  mainEl.classList.toggle('split', visible);
  previewToggleBtn.classList.toggle('active', visible);
  previewToggleBtn.textContent = visible ? t('editor.hidePreview') : t('editor.showPreview');
  if (visible) {
    renderPreview();
  }
}

function replaceSelection(replacement, selectStart, selectEnd) {
  const start = textareaEl.selectionStart;
  const end = textareaEl.selectionEnd;
  const before = textareaEl.value.slice(0, start);
  const after = textareaEl.value.slice(end);
  textareaEl.value = before + replacement + after;
  const nextStart = typeof selectStart === 'number' ? start + selectStart : start + replacement.length;
  const nextEnd = typeof selectEnd === 'number' ? start + selectEnd : nextStart;
  textareaEl.setSelectionRange(nextStart, nextEnd);
  textareaEl.focus();
  updateDirtyState();
  renderPreview();
}

function wrapSelection(prefix, suffix = prefix, placeholder = '') {
  const start = textareaEl.selectionStart;
  const end = textareaEl.selectionEnd;
  const selected = textareaEl.value.slice(start, end) || placeholder;
  const replacement = prefix + selected + suffix;
  replaceSelection(
    replacement,
    prefix.length,
    prefix.length + selected.length
  );
}

function prefixSelectedLines(prefix) {
  const start = textareaEl.selectionStart;
  const end = textareaEl.selectionEnd;
  const value = textareaEl.value;
  const lineStart = value.lastIndexOf('\n', start - 1) + 1;
  const lineEndIndex = value.indexOf('\n', end);
  const lineEnd = lineEndIndex === -1 ? value.length : lineEndIndex;
  const block = value.slice(lineStart, lineEnd);
  const prefixed = block
    .split('\n')
    .map((line) => (line.length ? `${prefix}${line}` : prefix.trimEnd()))
    .join('\n');

  textareaEl.value = value.slice(0, lineStart) + prefixed + value.slice(lineEnd);
  textareaEl.setSelectionRange(lineStart, lineStart + prefixed.length);
  textareaEl.focus();
  updateDirtyState();
  renderPreview();
}

function applyFormatting(action) {
  switch (action) {
    case 'bold':
      wrapSelection('**', '**', t('editor.placeholders.bold'));
      break;
    case 'italic':
      wrapSelection('*', '*', t('editor.placeholders.italic'));
      break;
    case 'strikethrough':
      wrapSelection('~~', '~~', t('editor.placeholders.strikethrough'));
      break;
    case 'h1':
      prefixSelectedLines('# ');
      break;
    case 'h2':
      prefixSelectedLines('## ');
      break;
    case 'h3':
      prefixSelectedLines('### ');
      break;
    case 'ul':
      prefixSelectedLines('- ');
      break;
    case 'ol':
      prefixSelectedLines('1. ');
      break;
    case 'quote':
      prefixSelectedLines('> ');
      break;
    case 'link':
      wrapSelection('[', '](url)', t('editor.placeholders.link'));
      break;
    case 'code':
      wrapSelection('`', '`', t('editor.placeholders.code'));
      break;
    case 'codeblock':
      wrapSelection('```\n', '\n```', t('editor.placeholders.code'));
      break;
    default:
      break;
  }
}

async function saveDocument() {
  if (isSaving) {
    return false;
  }
  isSaving = true;
  setStatus(t('editor.saving'));
  try {
    await window.markdownApi.writeFile(textareaEl.value);
    savedContent = textareaEl.value;
    updateDirtyState();
    setStatus(t('editor.saved'), 'saved');
    setTimeout(() => {
      if (!isDirty) {
        setStatus('');
      }
    }, 2000);
    return true;
  } catch (err) {
    setStatus(err.message || String(err), 'error');
    return false;
  } finally {
    isSaving = false;
  }
}

function closeConfirmDialog(result) {
  if (!confirmOverlayEl) {
    return;
  }
  confirmOverlayEl.classList.add('hidden');
  const resolver = confirmResolver;
  confirmResolver = null;
  if (resolver) {
    requestAnimationFrame(() => {
      resolver(result);
    });
  }
}

function isConfirmDialogActive() {
  return Boolean(confirmOverlayEl && !confirmOverlayEl.classList.contains('hidden'));
}

function showChoiceDialog({ title, message, options }) {
  if (
    !confirmOverlayEl ||
    !confirmTitleEl ||
    !confirmMessageEl ||
    !confirmCancelEl ||
    !confirmConfirmEl ||
    !confirmAltEl
  ) {
    return Promise.resolve('cancel');
  }

  const safeOptions =
    Array.isArray(options) && options.length
      ? options
        : [
          { id: 'cancel', label: t('editor.cancel'), danger: false },
          { id: 'confirm', label: t('editor.confirm'), danger: true },
        ];

  const firstOption = safeOptions[0];
  const secondOption = safeOptions[1] || null;
  const thirdOption = safeOptions[2] || null;
  const hasThreeOptions = Boolean(thirdOption);
  confirmChoiceIds = {
    cancel: firstOption?.id || 'cancel',
    alt: hasThreeOptions ? secondOption?.id || 'alt' : 'alt',
    confirm: (hasThreeOptions ? thirdOption : secondOption || firstOption)?.id || 'confirm',
  };

  confirmTitleEl.textContent = title || t('editor.confirmAction');
  confirmMessageEl.textContent = message || '';
  confirmCancelEl.textContent = firstOption?.label || t('editor.cancel');
  confirmCancelEl.classList.toggle('danger', Boolean(firstOption?.danger));

  if (hasThreeOptions && secondOption) {
    confirmAltEl.textContent = secondOption.label || t('editor.option');
    confirmAltEl.classList.remove('hidden');
    confirmAltEl.classList.toggle('danger', Boolean(secondOption.danger));
  } else {
    confirmAltEl.textContent = '';
    confirmAltEl.classList.add('hidden');
    confirmAltEl.classList.remove('danger');
  }

  const confirmButtonOption = hasThreeOptions
    ? thirdOption || firstOption
    : secondOption || firstOption;
  confirmConfirmEl.textContent = confirmButtonOption?.label || t('editor.confirm');
  confirmConfirmEl.classList.toggle('danger', Boolean(confirmButtonOption?.danger));
  confirmOverlayEl.classList.remove('hidden');
  if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur();
  }

  return new Promise((resolve) => {
    confirmResolver = resolve;
  });
}

async function handleAttemptClose() {
  if (!isDirty) {
    window.markdownApi.confirmClose();
    return;
  }

  const choice = await showChoiceDialog({
    title: t('editor.unsavedTitle'),
    message: t('editor.unsavedMessage', { fileName }),
    options: [
      { id: 'cancel', label: t('editor.cancel'), danger: false },
      { id: 'discard', label: t('editor.discard'), danger: false },
      { id: 'save', label: t('editor.save'), danger: false },
    ],
  });

  if (choice === 'cancel') {
    return;
  }

  if (choice === 'discard') {
    window.markdownApi.confirmClose();
    return;
  }

  if (choice === 'save') {
    const saved = await saveDocument();
    if (saved) {
      window.markdownApi.confirmClose();
    }
  }
}

function initializeConfirmDialog() {
  confirmCancelEl?.addEventListener('mousedown', (event) => {
    event.preventDefault();
  });
  confirmAltEl?.addEventListener('mousedown', (event) => {
    event.preventDefault();
  });
  confirmConfirmEl?.addEventListener('mousedown', (event) => {
    event.preventDefault();
  });
  confirmCancelEl?.addEventListener('click', () => {
    closeConfirmDialog(confirmChoiceIds.cancel);
  });
  confirmAltEl?.addEventListener('click', () => {
    closeConfirmDialog(confirmChoiceIds.alt);
  });
  confirmConfirmEl?.addEventListener('click', () => {
    closeConfirmDialog(confirmChoiceIds.confirm);
  });

  document.addEventListener(
    'keydown',
    (event) => {
      if (!isConfirmDialogActive()) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    },
    true
  );
}

function initializeToolbar() {
  toolbarEl.addEventListener('click', (event) => {
    const button = event.target.closest('[data-action]');
    if (!button) {
      return;
    }
    event.preventDefault();
    applyFormatting(button.getAttribute('data-action'));
  });
}

async function initializeEditor() {
  try {
    const context = await window.markdownApi.getContext();
    fileName = context.fileName || 'document.md';
    titleEl.textContent = fileName;
    document.title = `${fileName} - Glaux`;

    const file = await window.markdownApi.readFile();
    savedContent = file.content ?? '';
    textareaEl.value = savedContent;
    updateDirtyState();
    renderPreview();

    loadingEl.classList.add('hidden');
    toolbarEl.classList.remove('hidden');
    mainEl.classList.remove('hidden');
    setPreviewVisible(true);
    textareaEl.focus();
  } catch (err) {
    loadingEl.textContent = err.message || String(err);
    setStatus(t('editor.failedToLoad'), 'error');
  }
}

saveBtn.addEventListener('click', () => {
  void saveDocument();
});

previewToggleBtn.addEventListener('click', () => {
  setPreviewVisible(!previewVisible);
});

textareaEl.addEventListener('input', () => {
  updateDirtyState();
  if (previewVisible) {
    renderPreview();
  }
});

window.addEventListener('markdown-attempt-close', () => {
  void handleAttemptClose();
});

initializeConfirmDialog();
initializeToolbar();
void initializeEditor();
