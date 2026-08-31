const { t } = require('../../i18n');

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

function emitStreamEvent(sender, payload) {
  if (!sender || sender.isDestroyed()) {
    return false;
  }
  sender.send('engine:streamEvent', payload);
  return true;
}

module.exports = {
  classifyErrorCode,
  toStructuredError,
  ok,
  fail,
  emitStreamEvent,
};
