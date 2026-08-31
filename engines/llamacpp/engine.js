'use strict';

/**
 * llama.cpp engine bridge: spawns bundled llama-server and speaks its
 * OpenAI-compatible HTTP API. GPU backends (CUDA/Vulkan/Metal) are used
 * automatically when present; GLAUX_FORCE_CPU=1 forces CPU.
 *
 * This is the facade: process lifecycle + raw HTTP live in ./server;
 * message conversion + media URLs live in ./chat.
 * ASR GGUFs are routed to engines/transcribecpp by engineManager.
 */

const path = require('path');
const { resolveLocalGgufPaths } = require('../common/modelFormat');
const { stripThinkingFromMessages } = require('../common/stripThinking');
const { isForceCpu, withVendorLibPath, withSharedCudaLibPath } = require('../common/gpuRuntime');
const { withFfmpegEnv } = require('../common/ffmpeg');
const server = require('./server');
const chat = require('./chat');

/**
 * Explicit llama-server --ctx-size. Unset or 0 omits -c so llama.cpp --fit can
 * shrink from n_ctx_train (floor 4096) instead of locking the full trained
 * window. Passing -c 0 disables that shrink and OOMs long-context models on
 * small GPUs (Vulkan iGPUs in particular).
 * @param {unknown} [raw]
 * @returns {number | null}
 */
function parseLlamaCtxSize(raw = process.env.GLAUX_LLAMA_CTX) {
  if (raw == null || String(raw).trim() === '') {
    return null;
  }
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) {
    return null;
  }
  return Math.floor(n);
}

/** Fallback gauge denominator before /props reports the real window. */
const FALLBACK_CTX = 4096;

let closed = false;

/** @type {string | null} */
let modelsCacheDir = null;
/** @type {string | null} */
let resourcesRoot = null;
/** @type {string | null} */
let outputsRoot = null;

/** @type {string | null} */
let activeModelId = null;
/** @type {AbortController | null} */
let activeChatAbort = null;
/** Detected from llama-server /props chat_template after load. */
let supportsThinking = false;
/** Cached chat template string from /props. */
let chatTemplateSource = '';

/**
 * Ephemeral working buffer for the current inference/usage call.
 * Canonical history lives in engines/contextManager.js.
 * @type {Array<object>}
 */
let CONTEXT = [];

let nCtx = FALLBACK_CTX;
let cachedUsage = { used: 0, total: FALLBACK_CTX, valid: false };

/**
 * @param {object | null | undefined} usage
 */
function applyUsageFromServer(usage) {
  if (!usage || typeof usage !== 'object') {
    return false;
  }
  const prompt = Number(usage.prompt_tokens);
  const completion = Number(usage.completion_tokens);
  const total = Number(usage.total_tokens);
  let used = 0;
  if (Number.isFinite(total) && total >= 0) {
    used = total;
  } else if (Number.isFinite(prompt) && Number.isFinite(completion)) {
    used = prompt + completion;
  } else if (Number.isFinite(prompt)) {
    used = prompt;
  } else {
    return false;
  }
  cachedUsage = { used, total: nCtx, valid: true };
  return true;
}

/**
 * Measure tokens for a message list via chat template + tokenize.
 * @param {Array<object>} sourceMessages
 * @param {boolean} resubmit
 * @returns {Promise<number>}
 */
async function measureContextTokens(sourceMessages, resubmit) {
  let source = stripThinkingFromMessages(
    Array.isArray(sourceMessages) ? sourceMessages : []
  );
  if (resubmit === false && source.length > 2) {
    source = source.slice(-2);
  }
  const openMessages = source
    .map(chat.glauxMessageToOpenAITextOnly)
    .filter((m) => m.content.length > 0);
  if (openMessages.length === 0) {
    return 0;
  }

  const tmplRes = await server.httpRequest('POST', '/apply-template', {
    body: { messages: openMessages },
  });
  if (tmplRes.status < 200 || tmplRes.status >= 300) {
    throw new Error(tmplRes.body || `apply-template failed (${tmplRes.status})`);
  }
  let prompt = '';
  try {
    const parsed = JSON.parse(tmplRes.body);
    prompt = typeof parsed.prompt === 'string' ? parsed.prompt : '';
  } catch {
    throw new Error('Invalid apply-template response.');
  }
  if (!prompt) {
    return 0;
  }

  // Specials are already embedded by the chat template.
  const tokRes = await server.httpRequest('POST', '/tokenize', {
    body: { content: prompt, add_special: false, parse_special: true },
  });
  if (tokRes.status < 200 || tokRes.status >= 300) {
    throw new Error(tokRes.body || `tokenize failed (${tokRes.status})`);
  }
  const data = JSON.parse(tokRes.body);
  if (Array.isArray(data.tokens)) {
    return data.tokens.length;
  }
  if (typeof data.count === 'number') {
    return data.count;
  }
  return 0;
}

async function refreshNctxFromProps() {
  try {
    const props = await server.httpRequest('GET', '/props');
    if (props.status !== 200) {
      return;
    }
    const data = JSON.parse(props.body);
    const ctx =
      data.default_generation_settings?.n_ctx ||
      data.default_generation_settings?.params?.n_ctx ||
      data.n_ctx ||
      data.context_size;
    if (typeof ctx === 'number' && ctx > 0) {
      nCtx = ctx;
      cachedUsage.total = ctx;
    }
  } catch {
    /* keep current nCtx */
  }
}

/**
 * @param {{ modelsCacheDir?: string, resourcesRoot?: string, outputsRoot?: string }} [options]
 */
function configure(options = {}) {
  if (typeof options.modelsCacheDir === 'string' && options.modelsCacheDir.trim()) {
    modelsCacheDir = path.resolve(options.modelsCacheDir);
  }
  if (typeof options.resourcesRoot === 'string' && options.resourcesRoot.trim()) {
    resourcesRoot = path.resolve(options.resourcesRoot);
  }
  if (typeof options.outputsRoot === 'string' && options.outputsRoot.trim()) {
    outputsRoot = path.resolve(options.outputsRoot);
  }
  return Promise.resolve();
}

async function stopServer() {
  activeModelId = null;
  supportsThinking = false;
  chatTemplateSource = '';
  chat.setMediaPathRoot(null);
  if (activeChatAbort) {
    try {
      activeChatAbort.abort();
    } catch {
      /* ignore */
    }
    activeChatAbort = null;
  }
  return server.stopServer();
}

/**
 * Build llama-server argv. Omitting --device / -ngl lets llama.cpp auto-pick
 * GPU (default -ngl auto, --fit on). Omitting -c lets --fit shrink context
 * from n_ctx_train down to 4096 when VRAM is tight. GLAUX_FORCE_CPU pins CPU.
 * GLAUX_LLAMA_CTX / ctxSize > 0 pins an explicit window (--fit will not shrink it).
 *
 * @param {{
 *   modelPath: string,
 *   port: number,
 *   mediaPathRoot: string,
 *   mmprojPath?: string | null,
 *   ctxSize?: number | null,
 *   forceCpu?: boolean,
 * }} opts
 * @returns {string[]}
 */
function buildLlamaServerArgs(opts) {
  const ctxSize =
    opts.ctxSize != null && Number.isFinite(opts.ctxSize) && opts.ctxSize > 0
      ? Math.floor(opts.ctxSize)
      : null;
  const forceCpu = opts.forceCpu != null ? Boolean(opts.forceCpu) : isForceCpu();
  const args = [
    '-m',
    opts.modelPath,
    '--host',
    '127.0.0.1',
    '--port',
    String(opts.port),
    '--jinja',
    // Extract thoughts into reasoning_content so we can re-wrap them for the Glaux UI.
    '--reasoning-format',
    'deepseek',
    '--media-path',
    opts.mediaPathRoot,
  ];
  if (ctxSize != null) {
    args.push('-c', String(ctxSize));
  }
  if (forceCpu) {
    args.push('--device', 'none', '-ngl', '0', '--no-mmproj-offload');
  }
  if (opts.mmprojPath) {
    args.push('--mmproj', opts.mmprojPath);
  }
  return args;
}

/**
 * Turn raw llama-server stderr into a short, actionable load error.
 * @param {unknown} raw
 * @returns {string}
 */
function formatLlamaServerLoadError(raw) {
  const text = String(raw || '');
  const archMatch = text.match(/unknown model architecture:\s*'([^']+)'/i);
  if (archMatch) {
    const arch = archMatch[1];
    if (/^parakeet$/i.test(arch)) {
      return (
        `This GGUF uses architecture '${arch}', which llama-server does not support. ` +
        `Parakeet / Nemotron ASR GGUFs are handled by Glaux's transcribe.cpp engine when the ` +
        `model README declares pipeline_tag: automatic-speech-recognition. ` +
        `Alternatively use the Hugging Face safetensors build of the model.`
      );
    }
    return (
      `This GGUF uses architecture '${arch}', which the bundled llama-server does not support. ` +
      `Try a different quant/repo, or use a Hugging Face (safetensors) build of the model if available.`
    );
  }
  if (/failed to load model/i.test(text) || /exited before becoming ready/i.test(text)) {
    const trimmed = text.replace(/\s+/g, ' ').trim();
    return trimmed.length > 400 ? `${trimmed.slice(0, 400)}…` : trimmed || 'Failed to start llama-server.';
  }
  return text || 'Failed to start llama-server.';
}

/**
 * @param {string} modelId
 * @param {{ onProgress?: Function }} [options]
 */
async function chatbotCreate(modelId, options = {}) {
  if (closed) {
    throw new Error('Engine closed');
  }
  if (!modelsCacheDir) {
    throw new Error('Models cache directory is not configured.');
  }
  const onProgress =
    typeof options === 'function' ? options : options && options.onProgress;

  await stopServer();

  const modelRoot = path.join(modelsCacheDir, ...modelId.split('/'));
  const { modelPath, mmprojPath } = await resolveLocalGgufPaths(modelRoot);

  const port = await server.getFreePort();
  const bin = server.pickLlamaServer();
  const mediaRoots = [modelsCacheDir, resourcesRoot, outputsRoot].filter(Boolean);
  // Single media-path: userData root that contains Models/ and Workspaces/.
  // llama-server requires file:// URLs to be relative to this directory.
  const mediaPathRoot =
    mediaRoots.length > 0 ? path.dirname(mediaRoots[0]) : path.dirname(modelRoot);
  chat.setMediaPathRoot(mediaPathRoot);
  supportsThinking = false;
  chatTemplateSource = '';

  const binDir = path.dirname(bin);
  const args = buildLlamaServerArgs({
    modelPath,
    port,
    mediaPathRoot,
    mmprojPath,
    ctxSize: parseLlamaCtxSize(),
  });

  let env = withFfmpegEnv(withSharedCudaLibPath(withVendorLibPath({ ...process.env }, binDir)));

  if (onProgress) {
    onProgress({ status: 'progress', loaded: 0, total: 100, percent: 0 });
  }

  activeModelId = modelId;
  nCtx = FALLBACK_CTX;
  cachedUsage = { used: 0, total: nCtx, valid: true };

  try {
    await server.startServer({ bin, args, cwd: path.dirname(bin), env, port, onProgress });
  } catch (err) {
    await stopServer();
    activeModelId = null;
    throw new Error(formatLlamaServerLoadError(err && err.message ? err.message : err));
  }

  // Try to read context size + chat template (thinking support) from props.
  try {
    const props = await server.httpRequest('GET', '/props');
    if (props.status === 200) {
      const data = JSON.parse(props.body);
      const ctx =
        data.default_generation_settings?.n_ctx ||
        data.default_generation_settings?.params?.n_ctx ||
        data.n_ctx ||
        data.context_size;
      if (typeof ctx === 'number' && ctx > 0) {
        nCtx = ctx;
        cachedUsage.total = ctx;
      }
      if (typeof data.chat_template === 'string') {
        chatTemplateSource = data.chat_template;
        supportsThinking = chat.detectThinkingSupport(chatTemplateSource);
      }
    }
  } catch {
    /* keep defaults */
  }
}

async function chatbotDestroy() {
  await stopServer();
}

function contextClear() {
  CONTEXT = [];
  cachedUsage = { used: 0, total: nCtx, valid: true };
  return Promise.resolve();
}

function contextSnapshot() {
  return Promise.resolve(JSON.parse(JSON.stringify(CONTEXT)));
}

/**
 * Set the ephemeral working buffer (tests / legacy). Canonical history is in contextManager.
 * @param {unknown} newMessages
 */
async function contextReplace(newMessages) {
  CONTEXT = Array.isArray(newMessages) ? JSON.parse(JSON.stringify(newMessages)) : [];
  cachedUsage.valid = false;
  if (server.getPort() && activeModelId) {
    try {
      const used = await measureContextTokens(CONTEXT, true);
      cachedUsage = { used, total: nCtx, valid: true };
    } catch {
      cachedUsage = { used: 0, total: nCtx, valid: false };
    }
  }
}

/**
 * @param {boolean} [resubmit=true]
 * @param {boolean} [refresh=false]
 * @param {Array<object>} [sourceMessages] Canonical history from contextManager
 */
async function contextUsage(resubmit = true, refresh = false, sourceMessages) {
  if (!server.getPort() || !activeModelId) {
    return { used: 0, total: null, supported: false };
  }
  const source = Array.isArray(sourceMessages) ? sourceMessages : CONTEXT;
  if (!refresh && cachedUsage.valid) {
    return {
      used: cachedUsage.used,
      total: cachedUsage.total || nCtx,
      stale: false,
      supported: true,
    };
  }
  try {
    await refreshNctxFromProps();
    const used = await measureContextTokens(source, resubmit !== false);
    cachedUsage = { used, total: nCtx, valid: true };
    return { used, total: nCtx, stale: false, supported: true };
  } catch (err) {
    if (process.env.GLAUX_LLAMA_DEBUG === '1') {
      process.stderr.write(`[llamacpp] contextUsage failed: ${err && err.message ? err.message : err}\n`);
    }
    return {
      used: cachedUsage.used || 0,
      total: nCtx,
      stale: true,
      supported: true,
    };
  }
}

function chatbotSupportsThinking() {
  return Promise.resolve(Boolean(supportsThinking));
}

function chatbotHasChatTemplate() {
  return Promise.resolve(Boolean(activeModelId));
}

function chatGenerationStartsInThinking(_modelId, thinking) {
  // Reasoning-capable GGUF templates typically open a think block when thinking is enabled.
  return Promise.resolve(Boolean(thinking && supportsThinking));
}

/**
 * @param {string} modelId
 * @param {boolean} thinking
 * @param {string} message
 * @param {{ onToken?: Function, imagePaths?: string[], audioPaths?: string[], videoPaths?: string[], resubmit?: boolean, messages?: Array<object> }} [options]
 */
async function runChat(modelId, thinking, message, options = {}) {
  if (!activeModelId || activeModelId !== modelId) {
    throw new Error('Model is not loaded.');
  }
  const opts = typeof options === 'function' ? { onToken: options } : options || {};
  const resubmit = opts.resubmit !== false;
  const imagePaths = opts.imagePaths || [];
  const audioPaths = opts.audioPaths || [];
  const videoPaths = opts.videoPaths || [];

  // Prefer canonical history from contextManager; fall back to building a single-turn buffer.
  if (Array.isArray(opts.messages)) {
    CONTEXT = JSON.parse(JSON.stringify(opts.messages));
  } else {
    const userContent = chat.buildUserContent(message, imagePaths, audioPaths, videoPaths);
    CONTEXT = [{ role: 'user', content: userContent }];
  }

  const abort = new AbortController();
  activeChatAbort = abort;

  try {
    let reply = '';

    const openAiMessages = chat.buildInferenceOpenAIMessages(CONTEXT, resubmit);
    const body = {
      messages: openAiMessages,
      stream: true,
      temperature: 0.7,
      stream_options: { include_usage: true },
    };
    if (supportsThinking) {
      // Always pass the toggle explicitly so templates that default to thinking can be turned off.
      body.chat_template_kwargs = { enable_thinking: Boolean(thinking) };
      if (!thinking) {
        // Force an immediate end to any residual thinking budget when the user disables it.
        body.reasoning_budget = 0;
      }
    }
    const result = await server.streamChatCompletions(body, {
      onToken: opts.onToken,
      signal: abort.signal,
    });
    reply = result.text;
    if (!applyUsageFromServer(result.usage)) {
      cachedUsage.valid = false;
    }

    // Re-measure so the gauge includes the assistant reply (contextManager appends after return).
    const withAssistant = [
      ...CONTEXT,
      { role: 'assistant', content: [{ type: 'text', text: reply }] },
    ];
    try {
      const used = await measureContextTokens(withAssistant, true);
      cachedUsage = { used, total: nCtx, valid: true };
    } catch {
      cachedUsage.valid = false;
    }
    return reply;
  } catch (err) {
    if (abort.signal.aborted || /abort/i.test(String(err && err.message))) {
      return '';
    }
    throw err;
  } finally {
    if (activeChatAbort === abort) {
      activeChatAbort = null;
    }
  }
}

function chatStop() {
  if (activeChatAbort) {
    try {
      activeChatAbort.abort();
    } catch {
      /* ignore */
    }
  }
  const conversationId = server.getConversationId();
  if (conversationId && server.getPort()) {
    server.httpRequest('DELETE', `/v1/stream/${encodeURIComponent(conversationId)}`).catch(() => {});
  }
  return Promise.resolve();
}

function close(options = {}) {
  const final = options.final !== false;
  if (final) {
    closed = true;
  }
  return stopServer();
}

/** Downloads are handled by the huggingface engine; stubs for API parity. */
function downloadModel() {
  return Promise.reject(new Error('Use the Hub downloader via engineManager.downloadModel.'));
}

function cancelDownloadModel() {
  return Promise.resolve();
}

module.exports = {
  configure,
  close,
  downloadModel,
  cancelDownloadModel,
  chatbotCreate,
  chatbotDestroy,
  contextClear,
  contextSnapshot,
  contextUsage,
  contextReplace,
  chatbotSupportsThinking,
  chatbotHasChatTemplate,
  chatGenerationStartsInThinking,
  runChat,
  chatStop,
  parseLlamaCtxSize,
  buildLlamaServerArgs,
};
