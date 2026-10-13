'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function createClassList(classes) {
  return {
    add(...names) {
      for (const name of names) {
        if (name) {
          classes.add(name);
        }
      }
    },
    remove(...names) {
      for (const name of names) {
        classes.delete(name);
      }
    },
    contains(name) {
      return classes.has(name);
    },
    toggle(name, force) {
      const on = force === undefined ? !classes.has(name) : Boolean(force);
      if (on) {
        classes.add(name);
      } else {
        classes.delete(name);
      }
      return on;
    },
  };
}

function classSet(el) {
  return new Set(String(el.className || '').split(/\s+/).filter(Boolean));
}

function matches(el, sel) {
  if (!el || !sel) {
    return false;
  }
  const selector = sel.trim();
  if (!selector || selector.includes(' ')) {
    return false;
  }
  if (selector.startsWith('.')) {
    const body = selector.slice(1);
    const not = body.match(/^([\w-]+):not\(\.([\w-]+)\)$/);
    const classes = classSet(el);
    if (not) {
      return classes.has(not[1]) && !classes.has(not[2]);
    }
    return body.split('.').every((name) => classes.has(name));
  }
  return String(el.tagName || '').toLowerCase() === selector.toLowerCase();
}

function queryAll(root, sel) {
  const selector = String(sel || '').trim();
  if (!selector) {
    return [];
  }
  if (selector.includes(',')) {
    const out = [];
    for (const part of selector.split(',')) {
      out.push(...queryAll(root, part));
    }
    return out;
  }
  const sibling = selector.match(/^(.*?)\s*\+\s*(\S+)$/);
  if (sibling) {
    const prevs = queryAll(root, sibling[1]);
    const out = [];
    for (const prev of prevs) {
      const kids = prev.parent ? prev.parent.children : [];
      const index = kids.indexOf(prev);
      const next = kids[index + 1];
      if (next && matches(next, sibling[2])) {
        out.push(next);
      }
    }
    return out;
  }
  const parts = selector.split(/\s+/);
  if (parts.length > 1) {
    let current = [root];
    for (const part of parts) {
      const next = [];
      for (const node of current) {
        next.push(...queryAll(node, part));
      }
      current = next;
    }
    return current;
  }
  const out = [];
  const walk = (el) => {
    for (const child of el.children || []) {
      if (matches(child, selector)) {
        out.push(child);
      }
      walk(child);
    }
  };
  walk(root);
  return out;
}

function installElement(el) {
  el.children = [];
  el.parent = null;
  el.style = {};
  el.attributes = {};
  el.listeners = {};
  el.hidden = false;
  el.disabled = false;
  el.value = '';
  el.scrollTop = 0;
  el.scrollLeft = 0;
  el.tabIndex = 0;
  const classes = new Set();
  Object.defineProperty(el, 'className', {
    get() {
      return [...classes].join(' ');
    },
    set(value) {
      classes.clear();
      for (const name of String(value || '').split(/\s+/)) {
        if (name) {
          classes.add(name);
        }
      }
    },
  });
  el.classList = createClassList(classes);
  let text = '';
  let html = '';
  Object.defineProperty(el, 'textContent', {
    get() {
      return text;
    },
    set(value) {
      text = value == null ? '' : String(value);
      el.children = [];
    },
  });
  Object.defineProperty(el, 'innerHTML', {
    get() {
      return html;
    },
    set(value) {
      html = value == null ? '' : String(value);
      if (!html) {
        el.children = [];
      }
    },
  });
  el.appendChild = (child) => {
    if (child && typeof child.remove === 'function' && child.parent) {
      child.remove();
    }
    child.parent = el;
    el.children.push(child);
    return child;
  };
  el.append = (...nodes) => {
    for (const node of nodes) {
      el.appendChild(node);
    }
  };
  el.remove = () => {
    if (!el.parent) {
      return;
    }
    const index = el.parent.children.indexOf(el);
    if (index >= 0) {
      el.parent.children.splice(index, 1);
    }
    el.parent = null;
  };
  el.querySelectorAll = (sel) => queryAll(el, sel);
  el.querySelector = (sel) => queryAll(el, sel)[0] || null;
  el.closest = (sel) => {
    let node = el;
    while (node) {
      if (matches(node, sel)) {
        return node;
      }
      node = node.parent || null;
    }
    return null;
  };
  el.addEventListener = (type, fn) => {
    if (!el.listeners[type]) {
      el.listeners[type] = [];
    }
    el.listeners[type].push(fn);
  };
  el.removeEventListener = () => {};
  el.setAttribute = (name, value) => {
    el.attributes[name] = String(value);
  };
  el.getAttribute = (name) =>
    Object.prototype.hasOwnProperty.call(el.attributes, name) ? el.attributes[name] : null;
  el.removeAttribute = (name) => {
    delete el.attributes[name];
  };
  el.click = () => {
    let node = el;
    const event = {
      preventDefault() {},
      stopPropagation() {
        event.stopped = true;
      },
      target: el,
    };
    while (node) {
      event.currentTarget = node;
      for (const fn of node.listeners.click || []) {
        fn(event);
        if (event.stopped) {
          return;
        }
      }
      node = node.parent || null;
    }
  };
  el.focus = () => {};
  el.select = () => {};
  return el;
}

function createMemoryStorage() {
  const store = new Map();
  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(String(key), String(value));
    },
    removeItem(key) {
      store.delete(String(key));
    },
    clear() {
      store.clear();
    },
  };
}

function bootRenderer() {
  const HTMLElement = function HTMLElement() {};
  const HTMLSelectElement = function HTMLSelectElement() {};
  const HTMLButtonElement = function HTMLButtonElement() {};
  const HTMLInputElement = function HTMLInputElement() {};
  const HTMLTextAreaElement = function HTMLTextAreaElement() {};
  HTMLSelectElement.prototype = Object.create(HTMLElement.prototype);
  HTMLButtonElement.prototype = Object.create(HTMLElement.prototype);
  HTMLInputElement.prototype = Object.create(HTMLElement.prototype);
  HTMLTextAreaElement.prototype = Object.create(HTMLElement.prototype);

  const byTag = {
    select: HTMLSelectElement,
    button: HTMLButtonElement,
    input: HTMLInputElement,
    textarea: HTMLTextAreaElement,
  };
  const ids = new Map();

  function createElement(tag) {
    const Ctor = byTag[tag] || HTMLElement;
    const el = installElement(Object.create(Ctor.prototype));
    el.tagName = String(tag || 'div').toUpperCase();
    return el;
  }

  function elementFor(id) {
    if (ids.has(id)) {
      return ids.get(id);
    }
    let tag = 'div';
    if (id === 'workspace-select' || id === 'language-select' || id === 'theme-select') {
      tag = 'select';
    } else if (
      id === 'send-button' ||
      id === 'sessions-new' ||
      id === 'workspace-create' ||
      id === 'workspace-rename' ||
      id === 'workspace-trash'
    ) {
      tag = 'button';
    } else if (id === 'message-input' || id === 'sessions-search') {
      tag = 'textarea';
    } else if (id === 'reasoning-enabled' || id === 'resubmit-enabled') {
      tag = 'input';
    }
    const el = createElement(tag);
    el.id = id;
    if (id === 'send-button') {
      const icon = createElement('span');
      icon.classList.add('icon');
      icon.textContent = '>';
      const label = createElement('span');
      label.textContent = 'Send';
      el.append(icon, label);
    }
    if (id === 'messages') {
      Object.defineProperty(el, 'scrollHeight', { get: () => 400 });
    }
    ids.set(id, el);
    return el;
  }

  const document = {
    getElementById: (id) => elementFor(id),
    createElement,
    addEventListener() {},
    body: createElement('body'),
  };

  const sandbox = {
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    queueMicrotask,
    requestAnimationFrame(cb) {
      cb();
      return 1;
    },
    cancelAnimationFrame() {},
    HTMLElement,
    HTMLSelectElement,
    HTMLButtonElement,
    HTMLInputElement,
    HTMLTextAreaElement,
    document,
    sessionStorage: createMemoryStorage(),
    MODEL_TRASH_ICON_SVG: '<svg></svg>',
  };
  sandbox.globalThis = sandbox;
  sandbox.window = sandbox;
  sandbox.window.Glaux = {
    i18n: {
      t: (key) => key,
      language: 'en',
      available: [{ code: 'en', nativeName: 'English' }],
    },
  };
  const api = {
    _listeners: [],
    cancels: [],
    _sessions: [],
    _active: null,
    _loads: 0,
    _newError: null,
    onEngineStreamEvent(callback) {
      api._listeners.push(callback);
      return () => {};
    },
    cancelGeneration() {
      api.cancels.push('engine:cancel');
      return Promise.resolve({ ok: true });
    },
    cancelStream(id) {
      api.cancels.push(String(id));
    },
    getContextUsage: async () => ({ supported: false }),
    sessions: {
      list: async () => api._sessions,
      getActive: async () => api._active,
      load: async (name) => {
        api._loads += 1;
        return {
          sessionName: name,
          messages: [{ role: 'user', content: [{ type: 'text', text: 'from file' }] }],
        };
      },
      new: async () => {
        if (api._newError) {
          throw api._newError;
        }
      },
    },
  };
  sandbox.window.api = api;

  const context = vm.createContext(sandbox);
  vm.runInContext('function syncModelPanelDisabled() {}', context);
  const rendererRoot = path.join(__dirname, '../src/renderer');
  const files = [
    'shared/status.js',
    'shared/thinkingParse.js',
    'shared/mediaKinds.js',
    'state.js',
    'panels/workspaces.js',
    'panels/sessions.js',
    'panels/chat.js',
  ];
  for (const file of files) {
    const source = fs.readFileSync(path.join(rendererRoot, file), 'utf8');
    vm.runInContext(source, context, { filename: file });
  }

  function run(code) {
    return vm.runInContext(code, context);
  }

  return { context, api, run, document, ids };
}

describe('chat restore renderer', { concurrency: 1 }, () => {
  let env;
  const originalError = console.error;

  before(() => {
    env = bootRenderer();
    env.run('initializeSessionsPanel()');
  });

  after(() => {
    console.error = originalError;
  });

  function reset() {
    env.api._listeners = [];
    env.api.cancels = [];
    env.api._loads = 0;
    env.api._newError = null;
    env.api.getChatRestoreState = async () => ({ sessionName: null, messages: [], stream: null });
    console.error = () => {};
    env.run(`
      clearChatMessages();
      pendingChatAttachments = [];
      renderChatAttachments();
      inputEl.value = '';
      messagesEl.scrollTop = 0;
      chatRestorePending = false;
      chatRestoreFailed = false;
      activeStream = null;
      setBusy(false);
      sessionStorage.removeItem('glaux.chatComposer');
    `);
    console.error = originalError;
  }

  it('restores the unsent draft, attachments, and scroll when nothing is streaming', async () => {
    reset();
    env.run(`
      inputEl.value = 'half a thought';
      pendingChatAttachments = [{ source: 'resources', relativePath: 'pics/a.png', name: 'a.png' }];
      messagesEl.scrollTop = 80;
      saveChatComposerForReload();
      inputEl.value = '';
      pendingChatAttachments = [];
      renderChatAttachments();
      messagesEl.scrollTop = 0;
    `);
    const saved = JSON.parse(env.run(`sessionStorage.getItem('glaux.chatComposer')`));
    assert.equal(saved.draft, 'half a thought');
    assert.equal(saved.attachments[0].relativePath, 'pics/a.png');
    assert.equal(saved.scrollTop, 80);
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'earlier' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'reply' }] },
      ],
      stream: null,
    });
    await env.run('restoreChatAfterReload()');
    assert.equal(env.run('inputEl.value'), 'half a thought');
    assert.equal(env.run('pendingChatAttachments.length'), 1);
    assert.equal(env.run(`chatAttachmentsListEl.querySelector('.chat-attachment-name').textContent`), 'a.png');
    assert.equal(env.run('messagesEl.scrollTop'), 80);
    assert.notEqual(env.run('messagesEl.scrollTop'), env.run('messagesEl.scrollHeight'));
  });

  it('scrolls to the bottom while a live stream is restored and still keeps the draft', async () => {
    reset();
    env.run(`
      inputEl.value = 'still typing';
      messagesEl.scrollTop = 80;
      saveChatComposerForReload();
      inputEl.value = '';
      messagesEl.scrollTop = 0;
    `);
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      stream: {
        requestId: 'live',
        text: 'Hello',
        seq: 4,
        userPending: false,
        settling: false,
        startsInThinking: false,
      },
    });
    await env.run('restoreChatAfterReload()');
    assert.equal(env.run('inputEl.value'), 'still typing');
    assert.equal(env.run('messagesEl.scrollTop'), 400);
    assert.equal(env.run(`messagesEl.querySelector('.answer-content').textContent`), 'Hello');
  });

  it('ignores a status poll that is shorter or older than the text already shown', async () => {
    reset();
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      stream: {
        requestId: 'live',
        text: 'Hello',
        seq: 4,
        userPending: false,
        settling: false,
        startsInThinking: false,
      },
    });
    await env.run('restoreChatAfterReload()');
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      stream: {
        requestId: 'live',
        text: 'Hello there from an older read',
        seq: 1,
        userPending: false,
        settling: false,
      },
    });
    await env.run('reconcileRestoredGeneration()');
    assert.equal(env.run(`messagesEl.querySelector('.answer-content').textContent`), 'Hello');
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      stream: {
        requestId: 'live',
        text: 'Hel',
        seq: 6,
        userPending: false,
        settling: false,
      },
    });
    await env.run('reconcileRestoredGeneration()');
    assert.equal(env.run(`messagesEl.querySelector('.answer-content').textContent`), 'Hello');
  });

  it('paints an image when the restored run finishes', async () => {
    reset();
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'a red fox' }] }],
      stream: {
        requestId: 'img',
        text: '',
        seq: 1,
        userPending: false,
        settling: false,
        startsInThinking: false,
      },
    });
    await env.run('restoreChatAfterReload()');
    const listener = env.api._listeners[0];
    assert.equal(typeof listener, 'function');
    listener({
      requestId: 'img',
      type: 'done',
      seq: 2,
      response: '',
      images: [{ source: 'sessions', relativePath: 'chat-1.png', name: 'chat-1.png' }],
    });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(
      env.run(`messagesEl.querySelector('.message-attachment-name').textContent`),
      'chat-1.png'
    );
  });

  it('holds the busy lock for a settling stop without a second assistant bubble', async () => {
    reset();
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'hi' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'Hello\\n\\n[STOP]' }] },
      ],
      stream: {
        requestId: 'stop',
        text: 'Hello',
        seq: 3,
        userPending: false,
        settling: true,
        startsInThinking: false,
      },
    });
    await env.run('restoreChatAfterReload()');
    assert.equal(env.run(`messagesEl.querySelectorAll('.message.assistant').length`), 1);
    assert.equal(env.run('engineBusy'), true);
    assert.equal(env.run('Boolean(activeStream && activeStream.settling)'), true);
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'hi' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'Hello\\n\\n[STOP]' }] },
      ],
      stream: null,
    });
    await env.run('reconcileRestoredGeneration()');
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(env.run('engineBusy'), false);
    assert.equal(env.run('activeStream'), null);
    assert.equal(env.run(`messagesEl.querySelectorAll('.message.assistant').length`), 1);
  });

  it('does not render the session list until the list fetch has completed', async () => {
    reset();
    env.run(`
      sessionsListLoaded = false;
      sessionsList = [];
      globalThis.__renders = 0;
      globalThis.__renderSessionsList = renderSessionsList;
      renderSessionsList = function () {
        globalThis.__renders += 1;
        return globalThis.__renderSessionsList();
      };
    `);
    env.api.getChatRestoreState = async () => ({
      sessionName: 'chat.json',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      stream: null,
    });
    await env.run('restoreChatAfterReload()');
    assert.equal(env.run('globalThis.__renders'), 0);
    assert.equal(env.run(`messagesEl.querySelector('.message.user .message-body').textContent`), 'hi');
    assert.equal(env.run('messagesEl.scrollTop'), 400);
    env.run('renderSessionsList = globalThis.__renderSessionsList;');
  });

  it('leaves composer chips alone when the open session row is clicked', async () => {
    reset();
    env.api._sessions = [{ name: 'chat.json', displayName: 'chat' }];
    env.api._active = 'chat.json';
    await env.run('refreshSessionsList()');
    env.run(`
      pendingChatAttachments = [{ source: 'resources', relativePath: 'pics/a.png', name: 'a.png' }];
      renderChatAttachments();
      chatRestoreFailed = false;
      clearChatMessages();
    `);
    const loadsBefore = env.api._loads;
    env.ids.get('sessions-list').querySelector('.sessions-row-main').click();
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(env.api._loads, loadsBefore);
    assert.equal(env.run('pendingChatAttachments.length'), 1);
  });

  it('reloads a blank transcript after a failed restore without clearing composer chips', async () => {
    reset();
    env.api._sessions = [{ name: 'chat.json', displayName: 'chat' }];
    env.api._active = 'chat.json';
    await env.run('refreshSessionsList()');
    env.run(`
      pendingChatAttachments = [{ source: 'outputs', relativePath: 'note.txt', name: 'note.txt' }];
      renderChatAttachments();
      clearChatMessages();
      chatRestoreFailed = true;
      activeSessionName = 'chat.json';
    `);
    env.ids.get('sessions-list').querySelector('.sessions-row-main').click();
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(env.api._loads, 1);
    assert.equal(env.run('pendingChatAttachments.length'), 1);
    assert.equal(env.run(`messagesEl.querySelector('.message.user .message-body').textContent`), 'from file');
    assert.equal(env.run('chatRestoreFailed'), false);
  });

  it('treats a pending restore as busy and keeps the transcript when session reset fails', async () => {
    reset();
    env.run(`
      appendMessage('user', 'keep me');
      globalThis.__epoch = chatViewEpoch;
      chatRestorePending = true;
      syncWorkspaceControlsDisabled();
    `);
    assert.equal(env.ids.get('workspace-select').disabled, true);
    assert.equal(env.ids.get('workspace-create').disabled, true);
    env.run('chatRestorePending = false; syncWorkspaceControlsDisabled();');
    assert.equal(env.ids.get('workspace-select').disabled, false);
    env.api._newError = new Error('generation in flight');
    console.error = () => {};
    await env.run('refreshPanelsForWorkspaceChange()');
    console.error = originalError;
    assert.equal(env.run(`messagesEl.querySelector('.message.user .message-body').textContent`), 'keep me');
    assert.equal(env.run('chatViewEpoch'), env.run('globalThis.__epoch'));
    assert.equal(env.run('inputEl.value'), '');
  });

  it('calls engine:cancel from the send button while restore is still pending', () => {
    reset();
    env.run('chatRestorePending = true; updateSendButton();');
    env.ids.get('send-button').click();
    assert.deepEqual(env.api.cancels, ['engine:cancel']);
    assert.equal(env.run(`sendButtonEl.querySelector('.icon + span').textContent`), 'chat.stop');
    env.run('chatRestorePending = false; activeStream = null; setBusy(false);');
  });
});
