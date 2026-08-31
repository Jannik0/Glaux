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
  ensureOutputsDirectory,
  getOutputsRoot,
  moveEntryToRecycleBin,
  pathExists,
  ensureSessionsDirectory,
} = require('../paths');
const { outputsFs } = require('./outputs');
const { t } = require('../../i18n');

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

async function trashSessionFile(filename) {
  const abs = resolveSessionsPath(filename);
  await moveEntryToRecycleBin(abs);
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
      if (type !== 'image' && type !== 'audio' && type !== 'video') {
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

async function persistActiveSession() {
  const messages = await engineManager.contextSnapshot();
  if (!state.activeSessionFilename) {
    state.activeSessionFilename = `${formatSessionTimestamp()}.json`;
  }
  await writeSessionFile(state.activeSessionFilename, messages);
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
      await engineManager.contextClear();
      state.activeSessionFilename = null;
      return ok({});
    } catch (err) {
      return fail(err, 'E_SESSIONS');
    }
  });

  ipcMain.handle('sessions:load', async (_event, payload) => {
    try {
      const rawName = payload && payload.sessionName;
      if (typeof rawName !== 'string' || !rawName.trim()) {
        throw new Error(t('errors.sessions.nameRequired'));
      }
      const sessionName = assertValidSessionFilename(rawName.trim());
      const messages = await readSessionFile(sessionName);
      await engineManager.contextReplace(messages);
      state.activeSessionFilename = sessionName;
      return ok({ sessionName, messages });
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
      const sessionName = assertValidSessionFilename(rawName.trim());
      const wasActive = state.activeSessionFilename === sessionName;
      await trashSessionFile(sessionName);
      if (wasActive) {
        state.activeSessionFilename = null;
        try {
          await engineManager.contextClear();
        } catch {
          /* worker may not be running */
        }
      }
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
      const messages = await engineManager.contextSnapshot();
      if (messageIndex >= messages.length) {
        throw new Error(t('errors.sessions.messageDoesNotExist'));
      }
      messages.splice(messageIndex, 1);
      await engineManager.contextReplace(messages);
      if (state.activeSessionFilename) {
        await writeSessionFile(state.activeSessionFilename, messages);
      } else if (messages.length > 0) {
        await persistActiveSession();
      }
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
      const fileName = assertValidEntryName(`${sessionBase}-${messageNumber}.md`);
      const markdown = messageToExportMarkdown(msg);
      await ensureOutputsDirectory();
      const destinationPath = path.join(getOutputsRoot(), fileName);
      await fs.writeFile(destinationPath, markdown, 'utf8');
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
};
