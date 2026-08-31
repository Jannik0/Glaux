// Sessions side panel: list, create, rename, delete, and load chat sessions.

function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

const sessionsPanelEl = document.getElementById('sessions-panel');
const sessionsListEl = document.getElementById('sessions-list');
const sessionsNewEl = document.getElementById('sessions-new');
const sessionsSearchEl = document.getElementById('sessions-search');
const sessionsStatusEl = document.getElementById('sessions-status');

/** @type {string | null} */
let activeSessionName = null;
/** @type {Array<{ name: string, displayName: string }>} */
let sessionsList = [];
/** @type {string | null} */
let sessionsEditingName = null;
/** Prevents duplicate rename commits (e.g. Enter then blur). */
let sessionsRenameInFlight = false;

function setSessionsStatus(message, isError = false) {
  window.Glaux.Status.setTimedPanelStatus(sessionsStatusEl, 'sessions', message, isError, 'error');
}

function syncSessionsPanelDisabled() {
  const dis = engineBusy || engineLoading || Boolean(activeStream);
  if (sessionsNewEl) {
    sessionsNewEl.disabled = dis;
  }
  sessionsListEl?.querySelectorAll('.sessions-row').forEach((row) => {
    row.classList.toggle('is-disabled', dis);
  });
  sessionsListEl?.querySelectorAll('.sessions-trash').forEach((btn) => {
    if (btn instanceof HTMLButtonElement) {
      btn.disabled = dis;
    }
  });
  sessionsListEl?.querySelectorAll('.sessions-row-main').forEach((el) => {
    if (!(el instanceof HTMLElement)) {
      return;
    }
    if (dis) {
      el.setAttribute('aria-disabled', 'true');
      el.tabIndex = -1;
    } else {
      el.removeAttribute('aria-disabled');
      el.tabIndex = 0;
    }
  });
}

async function refreshSessionsList() {
  if (!(window.api && window.api.sessions && typeof window.api.sessions.list === 'function')) {
    return;
  }
  try {
    if (typeof window.api.sessions.getActive === 'function') {
      activeSessionName = await window.api.sessions.getActive();
    }
    sessionsList = await window.api.sessions.list();
    renderSessionsList();
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    setSessionsStatus(msg, true);
  }
}

function renderSessionsList() {
  if (!sessionsListEl) {
    return;
  }
  sessionsListEl.innerHTML = '';
  if (!sessionsList.length) {
    const empty = document.createElement('div');
    empty.className = 'sessions-empty';
    empty.textContent = t('panels.sessions.empty');
    sessionsListEl.appendChild(empty);
    return;
  }

  const query = (sessionsSearchEl?.value || '').trim().toLowerCase();
  const visibleSessions = query
    ? sessionsList.filter(
        (session) =>
          session.displayName.toLowerCase().includes(query) ||
          session.name.toLowerCase().includes(query)
      )
    : sessionsList;

  if (!visibleSessions.length) {
    const empty = document.createElement('div');
    empty.className = 'sessions-empty';
    empty.textContent = t('panels.sessions.noSearchMatches');
    sessionsListEl.appendChild(empty);
    return;
  }

  for (const session of visibleSessions) {
    const row = document.createElement('div');
    row.className = 'sessions-row';
    row.setAttribute('data-session-name', session.name);
    if (session.name === activeSessionName) {
      row.classList.add('is-active');
    }

    const main = document.createElement('div');
    main.className = 'sessions-row-main';
    main.setAttribute('role', 'button');
    main.tabIndex = 0;
    main.setAttribute('aria-label', t('panels.sessions.loadAria', { name: session.displayName }));

    if (sessionsEditingName === session.name) {
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'sessions-rename-input';
      input.value = session.displayName;
      input.setAttribute('data-session-name', session.name);
      input.addEventListener('click', (e) => e.stopPropagation());
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          void commitSessionRename(session.name, input.value);
        } else if (e.key === 'Escape') {
          e.preventDefault();
          sessionsEditingName = null;
          renderSessionsList();
        }
      });
      input.addEventListener('blur', () => {
        if (sessionsEditingName !== session.name) {
          return;
        }
        void commitSessionRename(session.name, input.value);
      });
      main.appendChild(input);
      requestAnimationFrame(() => {
        input.focus();
        input.select();
      });
    } else {
      const label = document.createElement('span');
      label.className = 'sessions-row-label';
      label.textContent = session.displayName;
      label.addEventListener('dblclick', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (engineBusy || engineLoading || activeStream) {
          return;
        }
        sessionsEditingName = session.name;
        renderSessionsList();
      });
      main.appendChild(label);
    }

    row.appendChild(main);

    const trashBtn = document.createElement('button');
    trashBtn.type = 'button';
    trashBtn.className = 'sessions-trash';
    trashBtn.setAttribute('data-session-name', session.name);
    trashBtn.setAttribute('aria-label', t('panels.sessions.deleteAria', { name: session.displayName }));
    trashBtn.title = t('panels.sessions.delete');
    trashBtn.innerHTML = MODEL_TRASH_ICON_SVG;
    trashBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      void trashSession(session.name);
    });
    row.appendChild(trashBtn);

    sessionsListEl.appendChild(row);
  }
  syncSessionsPanelDisabled();
}

async function loadSession(sessionName) {
  if (activeStream || engineBusy || engineLoading) {
    return;
  }
  if (sessionName === activeSessionName) {
    return;
  }
  if (!(window.api && window.api.sessions && typeof window.api.sessions.load === 'function')) {
    return;
  }
  try {
    const { messages, sessionName: loadedName } = await window.api.sessions.load(sessionName);
    activeSessionName = loadedName;
    clearChatMessages();
    clearChatAttachments();
    rebuildChatFromContext(messages);
    scrollMessagesToBottomAfterRender();
    void updateContextUsageIndicator({ refresh: true });
    sessionsEditingName = null;
    renderSessionsList();
    setSessionsStatus('', false);
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    setSessionsStatus(msg, true);
  }
}

async function startNewSession() {
  if (activeStream || engineBusy || engineLoading) {
    return;
  }
  if (!(window.api && window.api.sessions && typeof window.api.sessions.new === 'function')) {
    return;
  }
  try {
    await window.api.sessions.new();
    activeSessionName = null;
    clearChatMessages();
    if (inputEl) {
      inputEl.value = '';
      updateMessageInputHighlight();
    }
    clearChatAttachments();
    void updateContextUsageIndicator({ refresh: true });
    sessionsEditingName = null;
    renderSessionsList();
    setSessionsStatus('', false);
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    setSessionsStatus(msg, true);
  }
}

async function trashSession(sessionName) {
  if (engineBusy || engineLoading || activeStream) {
    return;
  }
  const session = sessionsList.find((s) => s.name === sessionName);
  const label = session ? session.displayName : sessionName;
  const confirmed = await window.Glaux.Dialogs.showConfirmDialog({
    title: t('panels.sessions.deleteConfirmTitle'),
    message: t('panels.sessions.deleteConfirmMessage', { name: label }),
    confirmLabel: t('panels.sessions.delete'),
  });
  if (!confirmed) {
    return;
  }
  try {
    const { wasActive, sessions } = await window.api.sessions.trash(sessionName);
    sessionsList = sessions;
    if (wasActive) {
      activeSessionName = null;
      clearChatMessages();
      clearChatAttachments();
      void updateContextUsageIndicator({ refresh: true });
    }
    sessionsEditingName = null;
    renderSessionsList();
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    setSessionsStatus(msg, true);
  }
}

/**
 * @param {string} oldName
 * @param {string} nextName
 */
async function commitSessionRename(oldName, nextName) {
  if (sessionsEditingName !== oldName || sessionsRenameInFlight) {
    return;
  }
  const session = sessionsList.find((s) => s.name === oldName);
  const currentDisplay = session ? session.displayName : oldName.replace(/\.json$/i, '');
  let safeName;
  try {
    safeName = window.Glaux.TreePanel.validateEntryName(nextName);
  } catch (err) {
    setSessionsStatus(err.message || String(err), true);
    sessionsEditingName = null;
    renderSessionsList();
    return;
  }
  if (safeName === currentDisplay) {
    sessionsEditingName = null;
    renderSessionsList();
    return;
  }
  sessionsRenameInFlight = true;
  try {
    const { sessionName: newFileName, sessions } = await window.api.sessions.rename(oldName, safeName);
    sessionsList = sessions;
    if (activeSessionName === oldName) {
      activeSessionName = newFileName;
    }
    sessionsEditingName = null;
    renderSessionsList();
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    setSessionsStatus(msg, true);
    sessionsEditingName = null;
    renderSessionsList();
  } finally {
    sessionsRenameInFlight = false;
  }
}

function initializeSessionsPanel() {
  sessionsPanelEl?.addEventListener('mousedown', () => {
    activePanel = 'sessions';
  });

  sessionsNewEl?.addEventListener('click', () => {
    void startNewSession();
  });

  sessionsSearchEl?.addEventListener('input', () => {
    renderSessionsList();
  });

  sessionsListEl?.addEventListener('click', (e) => {
    if (sessionsEditingName) {
      return;
    }
    const trash = e.target && /** @type {HTMLElement} */ (e.target).closest('.sessions-trash');
    if (trash) {
      return;
    }
    const main = e.target && /** @type {HTMLElement} */ (e.target).closest('.sessions-row-main');
    if (!main || engineBusy || engineLoading || activeStream) {
      return;
    }
    const row = main.closest('.sessions-row');
    const name = row && row.getAttribute('data-session-name');
    if (name) {
      void loadSession(name);
    }
  });

  void refreshSessionsList();
}
