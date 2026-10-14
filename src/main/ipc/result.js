const { t } = require('../../i18n');
const state = require('../state');

function classifyErrorCode(message, fallback = 'E_RUNTIME') {
  const text = String(message || '').toLowerCase();
  if (text.includes('cancel')) return 'E_CANCELED';
  if (text.includes('busy')) return 'E_BUSY';
  if (text.includes('not initialized') || text.includes('not available')) return 'E_NOT_READY';
  if (text.includes('timed out') || text.includes('timeout')) return 'E_TIMEOUT';
  if (text.includes('model') && text.includes('missing')) return 'E_MODEL_INVALID';
  if (text.includes('tokenizer') || text.includes('tensor') || text.includes('logits')) return 'E_INFERENCE';
  return fallback;
}

function toStructuredError(err, fallbackCode = 'E_RUNTIME') {
  const message = err && err.message ? err.message : String(err || t('errors.unknownError'));
  const existingCode = err && typeof err.code === 'string' && err.code ? err.code : '';
  const code = existingCode || classifyErrorCode(message, fallbackCode);
  return {
    code,
    message,
    timestamp: Date.now(),
  };
}

function ok(data = {}) {
  return { ok: true, ...data };
}

function fail(err, fallbackCode = 'E_RUNTIME') {
  const errorInfo = toStructuredError(err, fallbackCode);
  return { ok: false, error: errorInfo.message, errorInfo };
}

/**
 * The page that should receive stream tokens. A language switch reloads the
 * main window; the sender captured when the run started still points at the
 * frame that was discarded, so later tokens and the terminal event never
 * reach the restored page.
 *
 * @param {{ isDestroyed?: () => boolean, send?: Function } | null | undefined} sender
 */
function streamEventTarget(sender) {
  try {
    const win = state.getMainWindow();
    if (
      win &&
      !win.isDestroyed() &&
      win.webContents &&
      !win.webContents.isDestroyed()
    ) {
      return win.webContents;
    }
  } catch {
    /* no window in tests */
  }
  return sender;
}

function emitStreamEvent(sender, payload) {
  const target = streamEventTarget(sender);
  if (!target) {
    return false;
  }
  try {
    // send throws once the frame is disposed. That must not reject the
    // generation still running here.
    if (typeof target.isDestroyed === 'function' && target.isDestroyed()) {
      return false;
    }
    target.send('engine:streamEvent', payload);
    return true;
  } catch {
    return false;
  }
}

module.exports = {
  classifyErrorCode,
  toStructuredError,
  ok,
  fail,
  emitStreamEvent,
};
