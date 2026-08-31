// Generic confirm / choice / progress / name modal dialogs shared across panels.
// Exposed as window.Glaux.Dialogs. Owns its own DOM element lookups since the
// markup for these overlays lives once in index.html.
(function () {
  window.Glaux = window.Glaux || {};

  function t(key, vars) {
    return window.Glaux.i18n.t(key, vars);
  }

  const confirmOverlayEl = document.getElementById('confirm-overlay');
  const confirmTitleEl = document.getElementById('confirm-title');
  const confirmMessageEl = document.getElementById('confirm-message');
  const confirmCancelEl = document.getElementById('confirm-cancel');
  const confirmAltEl = document.getElementById('confirm-alt');
  const confirmConfirmEl = document.getElementById('confirm-confirm');

  const nameOverlayEl = document.getElementById('prompt-name-overlay');
  const nameTitleEl = document.getElementById('prompt-name-title');
  const nameMessageEl = document.getElementById('prompt-name-message');
  const nameInputEl = document.getElementById('prompt-name-input');
  const nameErrorEl = document.getElementById('prompt-name-error');
  const nameCancelEl = document.getElementById('prompt-name-cancel');
  const nameConfirmEl = document.getElementById('prompt-name-confirm');

  const progressOverlayEl = document.getElementById('app-progress-overlay');
  const progressTitleEl = document.getElementById('app-progress-title');
  const progressMessageEl = document.getElementById('app-progress-message');
  const progressBarEl = document.getElementById('app-progress-bar');

  /** @type {((value: string) => void) | null} */
  let confirmResolver = null;
  let confirmChoiceIds = { cancel: 'cancel', alt: 'alt', confirm: 'confirm' };

  /** @type {((value: string | null) => void) | null} */
  let nameResolver = null;
  /** @type {((value: string, existingNames: string[]) => string | null) | null} */
  let nameValidateFn = null;
  /** @type {string[]} */
  let nameExistingNames = [];
  /** @type {string} */
  let nameInitialValue = '';

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

  function isNameDialogActive() {
    return Boolean(nameOverlayEl && !nameOverlayEl.classList.contains('hidden'));
  }

  function isProgressModalOpen() {
    return Boolean(progressOverlayEl && !progressOverlayEl.classList.contains('hidden'));
  }

  function setNameError(message) {
    if (!nameErrorEl) {
      return;
    }
    const text = typeof message === 'string' ? message.trim() : '';
    if (!text) {
      nameErrorEl.textContent = '';
      nameErrorEl.classList.add('hidden');
      return;
    }
    nameErrorEl.textContent = text;
    nameErrorEl.classList.remove('hidden');
  }

  function closeNameDialog(result) {
    if (!nameOverlayEl) {
      return;
    }
    nameOverlayEl.classList.add('hidden');
    setNameError('');
    nameValidateFn = null;
    nameExistingNames = [];
    nameInitialValue = '';
    const resolver = nameResolver;
    nameResolver = null;
    if (resolver) {
      requestAnimationFrame(() => {
        resolver(result);
      });
    }
  }

  function tryConfirmNameDialog() {
    if (!nameInputEl) {
      closeNameDialog(null);
      return;
    }
    const raw = nameInputEl.value;
    if (typeof nameValidateFn === 'function') {
      const error = nameValidateFn(raw, nameExistingNames);
      if (error) {
        setNameError(error);
        nameInputEl.focus();
        nameInputEl.select();
        return;
      }
    }
    const trimmed = typeof raw === 'string' ? raw.trim() : '';
    if (trimmed === nameInitialValue) {
      closeNameDialog(null);
      return;
    }
    closeNameDialog(trimmed);
  }

  /**
   * @param {{
   *   title?: string,
   *   message?: string,
   *   initialValue?: string,
   *   confirmLabel?: string,
   *   existingNames?: string[],
   *   validate?: (value: string, existingNames: string[]) => string | null,
   * }} opts
   * @returns {Promise<string | null>} resolved name, or null if cancelled / unchanged
   */
  function showNameDialog({
    title,
    message,
    initialValue = '',
    confirmLabel = t('dialogs.save'),
    existingNames = [],
    validate = null,
  }) {
    if (
      !nameOverlayEl ||
      !nameTitleEl ||
      !nameMessageEl ||
      !nameInputEl ||
      !nameCancelEl ||
      !nameConfirmEl
    ) {
      return Promise.resolve(null);
    }

    nameTitleEl.textContent = title || t('dialogs.rename');
    nameMessageEl.textContent = message || '';
    nameMessageEl.classList.toggle('hidden', !message);
    nameConfirmEl.textContent = confirmLabel || t('dialogs.save');
    nameConfirmEl.classList.remove('danger');
    nameExistingNames = Array.isArray(existingNames) ? existingNames.slice() : [];
    nameValidateFn = typeof validate === 'function' ? validate : null;
    nameInitialValue = typeof initialValue === 'string' ? initialValue : '';
    nameInputEl.value = nameInitialValue;
    setNameError('');
    nameOverlayEl.classList.remove('hidden');
    requestAnimationFrame(() => {
      nameInputEl.focus();
      nameInputEl.select();
    });

    return new Promise((resolve) => {
      nameResolver = resolve;
    });
  }

  /**
   * @param {{ title?: string, message?: string, options: Array<{ id: string, label: string, danger?: boolean }> }} opts
   * @returns {Promise<string>}
   */
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
            { id: 'cancel', label: t('dialogs.cancel'), danger: false },
            { id: 'confirm', label: t('dialogs.confirm'), danger: true },
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

    confirmTitleEl.textContent = title || t('dialogs.confirmAction');
    confirmMessageEl.textContent = message || '';
    confirmCancelEl.textContent = firstOption?.label || t('dialogs.cancel');
    confirmCancelEl.classList.toggle('danger', Boolean(firstOption?.danger));

    if (hasThreeOptions && secondOption) {
      confirmAltEl.textContent = secondOption.label || t('dialogs.option');
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
    confirmConfirmEl.textContent = confirmButtonOption?.label || t('dialogs.confirm');
    confirmConfirmEl.classList.toggle('danger', Boolean(confirmButtonOption?.danger));
    confirmOverlayEl.classList.remove('hidden');
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }

    return new Promise((resolve) => {
      confirmResolver = resolve;
    });
  }

  /**
   * @param {{ title?: string, message?: string, confirmLabel?: string }} opts
   * @returns {Promise<boolean>}
   */
  function showConfirmDialog({ title, message, confirmLabel = t('dialogs.confirm') }) {
    return showChoiceDialog({
      title,
      message,
      options: [
        { id: 'cancel', label: t('dialogs.cancel'), danger: false },
        { id: 'confirm', label: confirmLabel, danger: true },
      ],
    }).then((choice) => choice === 'confirm');
  }

  function showProgressModal({ title, message }) {
    if (!progressOverlayEl || !progressTitleEl || !progressMessageEl || !progressBarEl) {
      return;
    }
    progressTitleEl.textContent = title || t('dialogs.working');
    progressMessageEl.textContent = message || '';
    progressBarEl.style.width = '0%';
    progressOverlayEl.classList.remove('hidden');
  }

  function updateProgressModal({ title, message, completed, total }) {
    if (!progressOverlayEl || progressOverlayEl.classList.contains('hidden')) {
      return;
    }
    if (typeof title === 'string' && progressTitleEl) {
      progressTitleEl.textContent = title;
    }
    if (typeof message === 'string' && progressMessageEl) {
      progressMessageEl.textContent = message;
    }
    if (progressBarEl && typeof completed === 'number' && typeof total === 'number' && total > 0) {
      const percent = Math.max(0, Math.min(100, Math.round((completed / total) * 100)));
      progressBarEl.style.width = `${percent}%`;
    }
  }

  function hideProgressModal() {
    if (!progressOverlayEl) {
      return;
    }
    progressOverlayEl.classList.add('hidden');
  }

  async function waitForUiPaint() {
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  function initializeConfirmDialogButtons() {
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
  }
  initializeConfirmDialogButtons();

  function initializeNameDialogButtons() {
    nameCancelEl?.addEventListener('click', () => {
      closeNameDialog(null);
    });
    nameConfirmEl?.addEventListener('click', () => {
      tryConfirmNameDialog();
    });
    nameInputEl?.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        tryConfirmNameDialog();
      } else if (event.key === 'Escape') {
        event.preventDefault();
        closeNameDialog(null);
      } else {
        setNameError('');
      }
    });
    nameInputEl?.addEventListener('input', () => {
      setNameError('');
    });
  }
  initializeNameDialogButtons();

  window.Glaux.Dialogs = {
    elements: {
      confirmOverlayEl,
      confirmTitleEl,
      confirmMessageEl,
      confirmCancelEl,
      confirmAltEl,
      confirmConfirmEl,
      nameOverlayEl,
      nameTitleEl,
      nameMessageEl,
      nameInputEl,
      nameErrorEl,
      nameCancelEl,
      nameConfirmEl,
      progressOverlayEl,
      progressTitleEl,
      progressMessageEl,
      progressBarEl,
    },
    closeConfirmDialog,
    closeNameDialog,
    isConfirmDialogActive,
    isNameDialogActive,
    isProgressModalOpen,
    showChoiceDialog,
    showConfirmDialog,
    showNameDialog,
    showProgressModal,
    updateProgressModal,
    hideProgressModal,
    waitForUiPaint,
  };
})();
