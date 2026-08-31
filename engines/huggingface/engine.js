'use strict';

const { spawn, execFile } = require('child_process');
const path = require('path');
const readline = require('readline');
const { isPackagedApp, pickPython } = require('../common/runtimePaths');
const { withForceCpuTorchEnv, withSharedCudaLibPath } = require('../common/gpuRuntime');
const { withFfmpegEnv } = require('../common/ffmpeg');

/** Directory that contains engine.py (+ model-downloader.py). */
function getEngineDir() {
  if (isPackagedApp()) {
    return path.join(process.resourcesPath, 'engines', 'huggingface');
  }
  return __dirname;
}

function getEnginePy() {
  return path.join(getEngineDir(), 'engine.py');
}

/** Loads engine.py once and serves JSON-RPC on stdin/stdout (newline-delimited).
 *  Requests run in a thread pool so e.g. chat_stop can be handled while run_chat() generates. */
const IPC_BOOTSTRAP = `import sys,json,importlib.util,threading,os
from concurrent.futures import ThreadPoolExecutor
p=sys.argv[1]
s=importlib.util.spec_from_file_location("engine",p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
_cache=os.environ.get("GLAUX_MODELS_CACHE_DIR")
if _cache:
 m.models_cache_dir(_cache)
_write_lock=threading.Lock()
_out=sys.stdout.buffer
def _emit(obj):
 with _write_lock:
  _out.write(json.dumps(obj,ensure_ascii=False).encode("utf-8")+b"\\n")
  _out.flush()
def _json_result(value):
 from pathlib import Path
 if isinstance(value, Path):
  return str(value)
 return value
def _call(name,args):
 f=getattr(m,name)
 return f(**args)
def _handle(q):
 i=q["id"];n=q["method"];a=q.get("args")or{}
 if n=="run_chat":
  try:
   full=[]
   for t in m.chat_stream(a["model_id"],a.get("thinking",False),a["message"],a.get("image_paths"),a.get("audio_paths"),a.get("video_paths"),resubmit=a.get("resubmit",True),messages=a.get("messages")):
    full.append(t)
    _emit({"id":i,"stream":True,"text":t})
   _emit({"id":i,"ok":True,"result":"".join(full)})
  except Exception as e:
   _emit({"id":i,"ok":False,"error":str(e),"errorType":type(e).__name__})
  return
 if n=="download_model":
  try:
   def _sink(ev):
    _emit({"id":i,"download":True,"event":ev})
   m.set_download_progress_callback(_sink)
   try:
    kw={"model_id":a["model_id"]}
    if a.get("allow_patterns") is not None:
     kw["allow_patterns"]=a.get("allow_patterns")
    if a.get("gguf_variant") is not None:
     kw["gguf_variant"]=a.get("gguf_variant")
    r=m.download_model(**kw)
   finally:
    m.set_download_progress_callback(None)
   _emit({"id":i,"ok":True,"result":r})
  except Exception as e:
   m.set_download_progress_callback(None)
   _emit({"id":i,"ok":False,"error":str(e),"errorType":type(e).__name__})
  return
 if n=="download_model_cancel":
  try:
   m.request_download_cancel(a["model_id"])
   _emit({"id":i,"ok":True,"result":None})
  except Exception as e:
   _emit({"id":i,"ok":False,"error":str(e),"errorType":type(e).__name__})
  return
 if n=="chatbot_create":
  try:
   def _sink(ev):
    _emit({"id":i,"load":True,"event":ev})
   m.set_load_progress_callback(_sink)
   try:
    r=m.chatbot_create(a["model_id"])
   finally:
    m.set_load_progress_callback(None)
   _emit({"id":i,"ok":True,"result":_json_result(r)})
  except Exception as e:
   m.set_load_progress_callback(None)
   _emit({"id":i,"ok":False,"error":str(e),"errorType":type(e).__name__})
  return
 try:
  r={"id":i,"ok":True,"result":_json_result(_call(n,a))}
 except Exception as e:
  r={"id":i,"ok":False,"error":str(e),"errorType":type(e).__name__}
 _emit(r)
_executor=ThreadPoolExecutor(max_workers=8)
for raw_line in sys.stdin.buffer:
 line=raw_line.decode("utf-8").strip()
 if not line:continue
 q=json.loads(line)
 _executor.submit(_handle,q)
`;

let child = null;
let rl = null;
let nextId = 0;
const pending = new Map();
let writeChain = Promise.resolve();
/** @type {string | null} */
let modelsCacheDir = null;
/** @type {number | null} */
let activeDownloadRpcId = null;
/** When true, no new Python worker is started (app is shutting down). */
let closed = false;
/** @type {Promise<void> | null} */
let terminatePromise = null;

function isChildAlive(proc) {
  return (
    proc != null &&
    proc.exitCode === null &&
    proc.signalCode === null &&
    !proc.killed
  );
}

/**
 * Kill the worker and any child processes (e.g. torch). Resolves when the OS process is gone.
 * @param {import('child_process').ChildProcess} proc
 * @returns {Promise<void>}
 */
function terminateProcess(proc) {
  const pid = proc.pid;
  if (!pid) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };

    proc.once('exit', finish);

    try {
      proc.stdin.end();
    } catch {
      /* ignore */
    }

    if (process.platform === 'win32') {
      execFile(
        'taskkill',
        ['/PID', String(pid), '/T', '/F'],
        { windowsHide: true },
        () => {
          if (!settled) {
            try {
              proc.kill();
            } catch {
              /* ignore */
            }
          }
        }
      );
    } else {
      // Spawned with a new process group (detached); kill the whole tree.
      try {
        process.kill(-pid, 'SIGTERM');
      } catch {
        try {
          proc.kill('SIGTERM');
        } catch {
          /* ignore */
        }
      }
    }

    setTimeout(() => {
      if (settled) return;
      try {
        if (process.platform === 'win32') {
          proc.kill('SIGKILL');
        } else {
          try {
            process.kill(-pid, 'SIGKILL');
          } catch {
            proc.kill('SIGKILL');
          }
        }
      } catch {
        /* ignore */
      }
      finish();
    }, 5000);
  });
}

function rejectAll(reason) {
  for (const [, { reject }] of pending) {
    reject(reason);
  }
  pending.clear();
}

function attachReaders(proc) {
  proc.stdout.setEncoding('utf8');
  rl = readline.createInterface({ input: proc.stdout });
  rl.on('line', (line) => {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    const id = msg.id;
    const p = pending.get(id);
    if (!p) return;

    if (msg.stream === true) {
      if (p.onToken) p.onToken(msg.text ?? '');
      return;
    }

    if (msg.download === true) {
      if (p.onProgress && msg.event) p.onProgress(msg.event);
      return;
    }

    if (msg.load === true) {
      if (p.onProgress && msg.event) p.onProgress(msg.event);
      return;
    }

    if (activeDownloadRpcId === id) {
      activeDownloadRpcId = null;
    }
    pending.delete(id);
    if (msg.ok) {
      p.resolve(msg.result);
    } else {
      const err = new Error(msg.error || 'Python RPC error');
      if (msg.errorType) err.pythonErrorType = msg.errorType;
      if (
        msg.errorType === 'DownloadCancelledError' ||
        /cancelled/i.test(String(msg.error || ''))
      ) {
        err.code = 'DOWNLOAD_CANCELLED';
      }
      p.reject(err);
    }
  });

  proc.stderr.setEncoding('utf8');
  proc.stderr.on('data', (chunk) => {
    process.stderr.write(chunk);
  });

  proc.on('error', (err) => {
    rejectAll(err);
  });

  proc.on('exit', (code, signal) => {
    child = null;
    if (rl) {
      rl.close();
      rl = null;
    }
    const reason = new Error(
      signal ? `Python process killed (${signal})` : `Python process exited with code ${code}`
    );
    rejectAll(reason);
  });
}

/**
 * @param {{ modelsCacheDir?: string }} [options]
 */
function configure(options = {}) {
  if (typeof options.modelsCacheDir === 'string' && options.modelsCacheDir.trim()) {
    modelsCacheDir = path.resolve(options.modelsCacheDir);
    if (child && !child.killed) {
      return rpcVoid('models_cache_dir', { path: modelsCacheDir });
    }
  }
  return Promise.resolve();
}

function ensureChild() {
  if (closed) {
    throw new Error('Engine closed');
  }
  if (terminatePromise) {
    throw new Error('Python worker is shutting down');
  }
  if (isChildAlive(child)) return;
  if (child) {
    child = null;
    if (rl) {
      rl.close();
      rl = null;
    }
  }

  const env = withFfmpegEnv(
    withSharedCudaLibPath(
      withForceCpuTorchEnv({
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONUTF8: '1',
        PYTHONIOENCODING: 'utf-8:utf-8',
      })
    )
  );
  if (modelsCacheDir) {
    env.GLAUX_MODELS_CACHE_DIR = modelsCacheDir;
  }
  // huggingface_hub freezes these into module-level constants the moment it's imported
  // (which `transformers` does before model-downloader.py ever runs), so they must be set
  // in the process environment before Python starts rather than from within engine.py.
  // hf_xet is not shipped; keep Xet disabled so huggingface_hub uses HTTP with timeout+resume.
  if (env.HF_HUB_DISABLE_XET === undefined) {
    env.HF_HUB_DISABLE_XET = '1';
  }
  if (env.HF_HUB_DOWNLOAD_TIMEOUT === undefined) {
    env.HF_HUB_DOWNLOAD_TIMEOUT = '30';
  }

  const enginePy = getEnginePy();
  const proc = spawn(pickPython(), ['-X', 'utf8', '-u', '-c', IPC_BOOTSTRAP, enginePy], {
    cwd: path.dirname(enginePy),
    env,
    windowsHide: true,
    // New process group on Unix so terminateProcess can kill torch children via -pid.
    detached: process.platform !== 'win32',
  });

  child = proc;
  attachReaders(proc);
}

function writeToStdin(chunk) {
  return new Promise((resolve, reject) => {
    if (!isChildAlive(child)) {
      reject(new Error('Python process is not running'));
      return;
    }
    child.stdin.write(chunk, 'utf8', (err) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

/**
 * @param {string} method Python function name on engine.py
 * @param {Record<string, unknown>} args Keyword arguments (snake_case to match Python)
 * @returns {Promise<unknown>}
 */
function rpc(method, args = {}) {
  ensureChild();
  const id = ++nextId;
  const payload = JSON.stringify({ id, method, args }) + '\n';

  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    writeChain = writeChain.then(() => writeToStdin(payload)).catch((err) => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(err);
      }
    });
  });
}

function rpcVoid(method, args = {}) {
  return rpc(method, args).then(() => undefined);
}

/**
 * Chat with streaming: Python emits newline-delimited `{ stream: true, text }` chunks, then `{ ok, result }`.
 *
 * @param {string} modelId
 * @param {boolean} thinking
 * @param {string} message
 * @param {{ onToken?: (chunk: string) => void, imagePaths?: string[], audioPaths?: string[], videoPaths?: string[], resubmit?: boolean } | ((chunk: string) => void)} [options]
 * @returns {Promise<string>}
 */
function runChat(modelId, thinking, message, options) {
  const opts = typeof options === 'function' ? { onToken: options } : options || {};
  const onToken = opts.onToken;
  ensureChild();
  const id = ++nextId;
  const args = { model_id: modelId, thinking, message };
  if (opts.imagePaths?.length) args.image_paths = opts.imagePaths;
  if (opts.audioPaths?.length) args.audio_paths = opts.audioPaths;
  if (opts.videoPaths?.length) args.video_paths = opts.videoPaths;
  if (opts.resubmit === false) args.resubmit = false;
  if (Array.isArray(opts.messages)) args.messages = opts.messages;
  const payload =
    JSON.stringify({
      id,
      method: 'run_chat',
      args,
    }) + '\n';

  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onToken });
    writeChain = writeChain.then(() => writeToStdin(payload)).catch((err) => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(err);
      }
    });
  });
}

/**
 * Terminates the long-lived Python worker (state in engine.py is lost).
 * Safe to call multiple times; returns a promise that resolves when the OS process exits.
 * @param {{ final?: boolean }} [options] If `final` is true (default), the worker cannot be started again until the app restarts.
 * @returns {Promise<void>}
 */
function close(options = {}) {
  const final = options.final !== false;
  if (final) {
    closed = true;
  }
  writeChain = Promise.resolve();
  rejectAll(new Error('Engine closed'));

  if (!child) {
    return terminatePromise || Promise.resolve();
  }
  if (terminatePromise) {
    return terminatePromise;
  }

  const proc = child;
  child = null;
  if (rl) {
    rl.close();
    rl = null;
  }

  terminatePromise = terminateProcess(proc).finally(() => {
    terminatePromise = null;
  });
  return terminatePromise;
}

module.exports = {
  configure,
  rpc,
  close,

  /**
   * @param {string} modelId
   * @param {{ onProgress?: (event: { status: string, file?: string, loaded?: number, total?: number }) => void, allowPatterns?: string[], ggufVariant?: string }} [options]
   */
  downloadModel: (modelId, options) => {
    const onProgress =
      typeof options === 'function' ? options : options && options.onProgress;
    const allowPatterns =
      options && typeof options === 'object' ? options.allowPatterns : undefined;
    const ggufVariant =
      options && typeof options === 'object' ? options.ggufVariant : undefined;
    ensureChild();
    const id = ++nextId;
    const args = { model_id: modelId };
    if (Array.isArray(allowPatterns) && allowPatterns.length) {
      args.allow_patterns = allowPatterns;
    }
    if (typeof ggufVariant === 'string' && ggufVariant.trim()) {
      args.gguf_variant = ggufVariant.trim();
    }
    const payload = JSON.stringify({ id, method: 'download_model', args }) + '\n';

    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject, onProgress });
      activeDownloadRpcId = id;
      writeChain = writeChain.then(() => writeToStdin(payload)).catch((err) => {
        if (pending.has(id)) {
          pending.delete(id);
          activeDownloadRpcId = null;
          reject(err);
        }
      });
    });
  },

  /**
   * @param {string} modelId
   * @returns {Promise<Array<{ path: string, size: number }>>}
   */
  listModelFiles: (modelId) => rpc('list_model_files', { model_id: modelId }),

  /**
   * Ask Python to cancel an in-flight download. Does not resolve the download RPC;
   * wait for that promise (or reset the worker) before deleting partial files.
   */
  cancelDownloadModel: (modelId) =>
    rpc('download_model_cancel', { model_id: modelId }).catch(() => {}),

  /**
   * @param {string} modelId
   * @param {{ onProgress?: (event: { status: string, loaded?: number, total?: number, percent?: number }) => void }} [options]
   */
  chatbotCreate: (modelId, options) => {
    const onProgress =
      typeof options === 'function' ? options : options && options.onProgress;
    if (!onProgress) {
      return rpcVoid('chatbot_create', { model_id: modelId });
    }
    ensureChild();
    const id = ++nextId;
    const payload =
      JSON.stringify({ id, method: 'chatbot_create', args: { model_id: modelId } }) + '\n';
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject, onProgress });
      writeChain = writeChain.then(() => writeToStdin(payload)).catch((err) => {
        if (pending.has(id)) {
          pending.delete(id);
          reject(err);
        }
      });
    });
  },

  chatbotDestroy: () => rpcVoid('chatbot_destroy', {}),

  contextClear: () => rpcVoid('context_clear', {}),

  contextSnapshot: () => rpc('context_snapshot', {}),

  /**
   * @param {boolean} [resubmit=true]
   * @param {boolean} [refresh=false]
   * @param {Array<object>} [messages] Canonical history from contextManager
   * @returns {Promise<{ used: number, total: number | null, stale?: boolean }>}
   */
  contextUsage: (resubmit = true, refresh = false, messages) => {
    const args = { resubmit: resubmit !== false, refresh: refresh === true };
    if (Array.isArray(messages)) {
      args.messages = messages;
    }
    return rpc('context_usage', args);
  },

  contextReplace: (messages) => rpcVoid('context_replace', { messages }),

  /**
   * Whether the **currently loaded** chat pipeline's tokenizer chat template supports thinking
   * supports a configured thinking template kwarg (e.g. `enable_thinking`).
   * @returns {Promise<boolean>}
   */
  chatbotSupportsThinking: () => rpc('chatbot_supports_thinking', {}),

  /** @returns {Promise<boolean>} */
  chatbotHasChatTemplate: () => rpc('chatbot_has_chat_template', {}),

  /**
   * @param {string} modelId
   * @param {boolean} thinking
   * @param {string} message
   * @param {{ imagePaths?: string[], audioPaths?: string[], videoPaths?: string[], resubmit?: boolean, messages?: Array<object> }} [media]
   */
  chatGenerationStartsInThinking: (
    modelId,
    thinking,
    message,
    media = {}
  ) => {
    const args = { model_id: modelId, thinking, message };
    if (media.imagePaths?.length) args.image_paths = media.imagePaths;
    if (media.audioPaths?.length) args.audio_paths = media.audioPaths;
    if (media.videoPaths?.length) args.video_paths = media.videoPaths;
    if (media.resubmit === false) args.resubmit = false;
    if (Array.isArray(media.messages)) args.messages = media.messages;
    return rpc('chat_generation_starts_in_thinking', args);
  },

  runChat,

  /** Requests the current `runChat()` generation to stop at the next token (via StoppingCriteria). */
  chatStop: () => (isChildAlive(child) ? rpcVoid('chat_stop', {}) : Promise.resolve()),
};
