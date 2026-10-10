const path = require('path');
const fs = require('fs/promises');
const { ipcMain } = require('electron');
const engineManager = require('../../../engines/engineManager');
const state = require('../state');
const { ok, fail } = require('../ipc/result');
const {
  assertValidEntryName,
  assertValidSessionFilename,
  resolveSessionsPath,
  getSessionsRoot,
  getResourcesRoot,
  ensureOutputsDirectory,
  getOutputsRoot,
  moveEntryToRecycleBin,
  pathExists,
  ensureSessionsDirectory,
} = require('../paths');
const { outputsFs } = require('./outputs');
const { t } = require('../../i18n');
const {
  copyAssistantImagesToOutputs,
  deleteSessionSidecarImages,
} = require('../sessionImages');

function formatSessionTimestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  );
}

function validateSessionMessages(messages) {
  if (!Array.isArray(messages)) {
    throw new Error(t('errors.sessions.mustBeJsonArray'));
  }
  for (let i = 0; i < messages.length; i += 1) {
    const msg = messages[i];
    if (!msg || typeof msg !== 'object') {
      throw new Error(t('errors.sessions.messageMustBeObject', { index: i }));
    }
    const role = msg.role;
    if (role !== 'user' && role !== 'assistant') {
      throw new Error(t('errors.sessions.messageInvalidRole', { index: i }));
    }
  }
  return messages;
}

async function listSessionFiles() {
  await ensureSessionsDirectory();
  const sessionsRoot = getSessionsRoot();
  const entries = await fs.readdir(sessionsRoot, { withFileTypes: true });
  const files = entries.filter(
    (e) => e.isFile() && e.name.toLowerCase().endsWith('.json')
  );
  const withStats = await Promise.all(
    files.map(async (e) => {
      const abs = path.join(sessionsRoot, e.name);
      const stats = await fs.stat(abs);
      return {
        name: e.name,
        displayName: e.name.replace(/\.json$/i, ''),
        mtimeMs: stats.mtimeMs,
      };
    })
  );
  withStats.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return withStats.map(({ name, displayName }) => ({ name, displayName }));
}

async function writeSessionFile(filename, messages) {
  await ensureSessionsDirectory();
  const abs = resolveSessionsPath(filename);
  validateSessionMessages(messages);
  await fs.writeFile(abs, `${JSON.stringify(messages, null, 2)}\n`, 'utf8');
}

async function readSessionFile(filename) {
  await ensureSessionsDirectory();
  const abs = resolveSessionsPath(filename);
  const raw = await fs.readFile(abs, 'utf8');
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(t('errors.sessions.notValidJson'));
  }
  return validateSessionMessages(parsed);
}

async function renameSessionFile(oldName, newName) {
  await ensureSessionsDirectory();
  const sourcePath = resolveSessionsPath(oldName);
  const destName = assertValidSessionFilename(newName);
  const destPath = resolveSessionsPath(destName);
  if (destPath === sourcePath) {
    return destName;
  }
  if (await pathExists(destPath)) {
    throw new Error(t('errors.sessions.alreadyExists', { name: destName }));
  }
  const sourceStats = await fs.stat(sourcePath).catch(() => null);
  if (!sourceStats || !sourceStats.isFile()) {
    throw new Error(t('errors.sessions.doesNotExist'));
  }
  await fs.rename(sourcePath, destPath);
  return destName;
}

/**
 * Name reserved for a session that does not have a JSON file yet.
 * Assigned to state.activeSessionFilename only after persist succeeds.
 * @type {string | null}
 */
let pendingSessionFilename = null;

function warnSession(message, err) {
  console.warn(message, err);
}

/**
 * Move the session JSON to the recycle bin, then delete its sidecar images.
 * A sidecar delete failure is logged and does not fail the trash. Outputs
 * copies are never removed. An unreadable JSON is still trashed.
 *
 * @param {string} filename
 * @param {{
 *   resolveSessionsPath?: (filename: string) => string,
 *   readSessionFile?: (filename: string) => Promise<Array<object>>,
 *   moveEntryToRecycleBin?: (abs: string) => Promise<void>,
 *   deleteSessionSidecarImages?: (messages: Array<object>, sessionsRoot: string) => Promise<unknown>,
 *   sessionsRoot?: string,
 *   warn?: (message: string, err: unknown) => void,
 * }} [deps]
 */
async function trashSessionFile(filename, deps = {}) {
  const resolve = deps.resolveSessionsPath || resolveSessionsPath;
  const read = deps.readSessionFile || readSessionFile;
  const move = deps.moveEntryToRecycleBin || moveEntryToRecycleBin;
  const sessionsRoot = deps.sessionsRoot || getSessionsRoot();
  const removeSidecars =
    deps.deleteSessionSidecarImages ||
    ((messages, root) => deleteSessionSidecarImages(messages, root));
  const warn = deps.warn || warnSession;
  const abs = resolve(filename);
  let messages = null;
  try {
    messages = await read(filename);
  } catch (err) {
    warn('Unreadable session JSON; moving it to the recycle bin without deleting sidecars:', err);
  }
  await move(abs);
  if (!messages) {
    return;
  }
  try {
    await removeSidecars(messages, sessionsRoot);
  } catch (err) {
    warn('Failed to delete session sidecar images:', err);
  }
}

function extractSessionMessageText(msg) {
  if (!msg || typeof msg !== 'object') {
    return '';
  }
  const content = msg.content;
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    return '';
  }
  return content
    .filter((part) => part && typeof part === 'object' && part.type === 'text')
    .map((part) => String(part.text || ''))
    .join('\n')
    .trim();
}

function messageToExportMarkdown(msg) {
  if (!msg || typeof msg !== 'object') {
    return '';
  }
  if (msg.role === 'assistant') {
    return extractSessionMessageText(msg);
  }
  const lines = [];
  const text = extractSessionMessageText(msg);
  if (text) {
    lines.push(text);
  }
  if (Array.isArray(msg.content)) {
    for (const part of msg.content) {
      if (!part || typeof part !== 'object') {
        continue;
      }
      const type = part.type;
      if (type !== 'image' && type !== 'audio' && type !== 'video' && type !== 'file') {
        continue;
      }
      let label = '';
      if (typeof part.relativePath === 'string' && part.relativePath) {
        label = part.relativePath;
      } else if (typeof part.path === 'string' && part.path) {
        label = part.path;
      }
      if (label) {
        lines.push(`- Attachment: ${label}`);
      }
    }
  }
  return lines.join('\n\n');
}

function getSessionExportBaseName() {
  if (state.activeSessionFilename) {
    return state.activeSessionFilename.replace(/\.json$/i, '');
  }
  return 'untitled';
}

function reserveActiveSessionFilename() {
  if (state.activeSessionFilename) {
    return state.activeSessionFilename;
  }
  if (!pendingSessionFilename) {
    pendingSessionFilename = `${formatSessionTimestamp()}.json`;
  }
  return pendingSessionFilename;
}

function clearPendingSessionFilename() {
  pendingSessionFilename = null;
}

/**
 * @returns {string | null}
 */
function getPendingSessionFilename() {
  return pendingSessionFilename;
}

/**
 * @param {{ contextClear?: () => Promise<void> }} [deps]
 */
async function startNewSession(deps = {}) {
  const clear = deps.contextClear || (() => engineManager.contextClear());
  await clear();
  state.activeSessionFilename = null;
  clearPendingSessionFilename();
}

/**
 * @param {unknown} rawName
 * @param {{
 *   readSessionFile?: (filename: string) => Promise<Array<object>>,
 *   contextReplace?: (messages: Array<object>) => Promise<void>,
 *   assertValidSessionFilename?: (name: string) => string,
 * }} [deps]
 */
async function loadSession(rawName, deps = {}) {
  const read = deps.readSessionFile || readSessionFile;
  const replace = deps.contextReplace || ((messages) => engineManager.contextReplace(messages));
  const assertName = deps.assertValidSessionFilename || assertValidSessionFilename;
  if (typeof rawName !== 'string' || !rawName.trim()) {
    throw new Error(t('errors.sessions.nameRequired'));
  }
  const sessionName = assertName(rawName.trim());
  const messages = await read(sessionName);
  await replace(messages);
  state.activeSessionFilename = sessionName;
  clearPendingSessionFilename();
  return { sessionName, messages };
}

/**
 * @param {unknown} rawName
 * @param {{
 *   trashSessionFile?: (filename: string) => Promise<void>,
 *   contextClear?: () => Promise<void>,
 *   assertValidSessionFilename?: (name: string) => string,
 *   getActiveSessionFilename?: () => string | null,
 * }} [deps]
 */
async function trashNamedSession(rawName, deps = {}) {
  const assertName = deps.assertValidSessionFilename || assertValidSessionFilename;
  const trash = deps.trashSessionFile || ((filename) => trashSessionFile(filename));
  const clearContext = deps.contextClear || (() => engineManager.contextClear());
  const activeName = deps.getActiveSessionFilename || (() => state.activeSessionFilename);
  if (typeof rawName !== 'string' || !rawName.trim()) {
    throw new Error(t('errors.sessions.nameRequired'));
  }
  const sessionName = assertName(rawName.trim());
  const wasActive = activeName() === sessionName;
  await trash(sessionName);
  if (wasActive) {
    state.activeSessionFilename = null;
    clearPendingSessionFilename();
    try {
      await clearContext();
    } catch {
      /* worker may not be running */
    }
  }
  return { sessionName, wasActive };
}

/**
 * @param {{
 *   contextSnapshot?: () => Promise<Array<object>>,
 *   writeSessionFile?: (filename: string, messages: Array<object>) => Promise<void>,
 * }} [deps]
 * @returns {Promise<string>}
 */
async function persistActiveSession(deps = {}) {
  const snapshot = deps.contextSnapshot || (() => engineManager.contextSnapshot());
  const write = deps.writeSessionFile || writeSessionFile;
  const messages = await snapshot();
  const filename = reserveActiveSessionFilename();
  await write(filename, messages);
  state.activeSessionFilename = filename;
  pendingSessionFilename = null;
  return filename;
}

/**
 * Drop one context message, then the session file, and only then its sidecars.
 * A sidecar delete failure is logged and does not fail the operation.
 *
 * @param {number} messageIndex
 * @param {{
 *   contextSnapshot?: () => Promise<Array<object>>,
 *   contextReplace?: (messages: Array<object>) => Promise<void>,
 *   writeSessionFile?: (filename: string, messages: Array<object>) => Promise<void>,
 *   persistActiveSession?: () => Promise<unknown>,
 *   deleteSessionSidecarImages?: (messages: Array<object>, sessionsRoot: string) => Promise<unknown>,
 *   sessionsRoot?: string,
 *   getActiveSessionFilename?: () => string | null,
 *   warn?: (message: string, err: unknown) => void,
 * }} [deps]
 * @returns {Promise<Array<object>>}
 */
async function deleteSessionMessage(messageIndex, deps = {}) {
  if (!Number.isInteger(messageIndex) || messageIndex < 0) {
    throw new Error(t('errors.sessions.messageIndexRequired'));
  }
  const snapshot = deps.contextSnapshot || (() => engineManager.contextSnapshot());
  const replace = deps.contextReplace || ((messages) => engineManager.contextReplace(messages));
  const write = deps.writeSessionFile || writeSessionFile;
  const persist = deps.persistActiveSession || (() => persistActiveSession());
  const sessionsRoot = deps.sessionsRoot || getSessionsRoot();
  const removeSidecars =
    deps.deleteSessionSidecarImages ||
    ((messages, root) => deleteSessionSidecarImages(messages, root));
  const activeName = deps.getActiveSessionFilename || (() => state.activeSessionFilename);
  const warn = deps.warn || warnSession;

  const messages = await snapshot();
  if (messageIndex >= messages.length) {
    throw new Error(t('errors.sessions.messageDoesNotExist'));
  }
  const removed = messages[messageIndex];
  const next = messages.slice();
  next.splice(messageIndex, 1);
  await replace(next);
  const active = activeName();
  if (active) {
    await write(active, next);
  } else if (next.length > 0) {
    await persist();
  }
  try {
    await removeSidecars([removed], sessionsRoot);
  } catch (err) {
    warn('Failed to delete session sidecar images:', err);
  }
  return next;
}

function registerSessionsIpc() {
  ipcMain.handle('sessions:list', async () => {
    try {
      const sessions = await listSessionFiles();
      return ok({ sessions });
    } catch (err) {
      return fail(err, 'E_SESSIONS');
    }
  });

  ipcMain.handle('sessions:getActive', async () => {
    try {
      return ok({ sessionName: state.activeSessionFilename });
    } catch (err) {
      return fail(err, 'E_SESSIONS');
    }
  });

  ipcMain.handle('sessions:new', async () => {
    try {
      await startNewSession();
      return ok({});
    } catch (err) {
      return fail(err, 'E_SESSIONS');
    }
  });

  ipcMain.handle('sessions:load', async (_event, payload) => {
    try {
      const result = await loadSession(payload && payload.sessionName);
      return ok(result);
    } catch (err) {
      return fail(err, 'E_SESSIONS');
    }
  });

  ipcMain.handle('sessions:rename', async (_event, payload) => {
    try {
      const oldName = payload && payload.oldName;
      const newName = payload && payload.newName;
      if (typeof oldName !== 'string' || !oldName.trim()) {
        throw new Error(t('errors.sessions.currentNameRequired'));
      }
      if (typeof newName !== 'string' || !newName.trim()) {
        throw new Error(t('errors.sessions.newNameRequired'));
      }
      const safeOld = assertValidSessionFilename(oldName.trim());
      const safeNew = await renameSessionFile(safeOld, newName.trim());
      if (state.activeSessionFilename === safeOld) {
        state.activeSessionFilename = safeNew;
      }
      const sessions = await listSessionFiles();
      return ok({ sessionName: safeNew, sessions });
    } catch (err) {
      return fail(err, 'E_SESSIONS');
    }
  });

  ipcMain.handle('sessions:trash', async (_event, payload) => {
    try {
      const rawName = payload && payload.sessionName;
      if (typeof rawName !== 'string' || !rawName.trim()) {
        throw new Error(t('errors.sessions.nameRequired'));
      }
      const { sessionName, wasActive } = await trashNamedSession(rawName);
      const sessions = await listSessionFiles();
      return ok({ sessionName, wasActive, sessions });
    } catch (err) {
      return fail(err, 'E_SESSIONS');
    }
  });

  ipcMain.handle('sessions:deleteMessage', async (_event, payload) => {
    try {
      const messageIndex = payload && payload.messageIndex;
      if (!Number.isInteger(messageIndex) || messageIndex < 0) {
        throw new Error(t('errors.sessions.messageIndexRequired'));
      }
      const messages = await deleteSessionMessage(messageIndex);
      return ok({ messages });
    } catch (err) {
      return fail(err, 'E_SESSIONS');
    }
  });

  ipcMain.handle('chat:exportMessage', async (_event, payload) => {
    try {
      const messageIndex = payload && payload.messageIndex;
      if (!Number.isInteger(messageIndex) || messageIndex < 0) {
        throw new Error(t('errors.sessions.messageIndexRequired'));
      }
      const messages = await engineManager.contextSnapshot();
      if (messageIndex >= messages.length) {
        throw new Error(t('errors.sessions.messageDoesNotExist'));
      }
      const msg = messages[messageIndex];
      const sessionBase = getSessionExportBaseName();
      const messageNumber = messageIndex + 1;
      await ensureOutputsDirectory();
      const imageNames = await copyAssistantImagesToOutputs(msg, {
        sessionsRoot: getSessionsRoot(),
        resourcesRoot: getResourcesRoot(),
        outputsRoot: getOutputsRoot(),
      });
      const markdown = messageToExportMarkdown(msg);
      const assistantImagesOnly =
        msg.role === 'assistant' &&
        imageNames.length > 0 &&
        !String(markdown || '').trim();
      const written = [];
      if (!assistantImagesOnly) {
        const mdName = assertValidEntryName(`${sessionBase}-${messageNumber}.md`);
        await fs.writeFile(path.join(getOutputsRoot(), mdName), markdown, 'utf8');
        written.push(mdName);
      }
      written.push(...imageNames);
      const fileName = written.join(', ');
      const tree = await outputsFs.getTree();
      return ok({ tree, fileName });
    } catch (err) {
      return fail(err, 'E_RUNTIME');
    }
  });
}

module.exports = {
  registerSessionsIpc,
  persistActiveSession,
  reserveActiveSessionFilename,
  clearPendingSessionFilename,
  getPendingSessionFilename,
  startNewSession,
  loadSession,
  trashNamedSession,
  deleteSessionMessage,
  trashSessionFile,
};
