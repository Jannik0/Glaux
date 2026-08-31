'use strict';

/**
 * llama-server process lifecycle (spawn / health-check / terminate) plus the
 * raw HTTP client used to talk to it, including the streaming chat endpoint.
 */

const { spawn, execFile } = require('child_process');
const http = require('http');
const net = require('net');
const path = require('path');
const { findVendorBinary, getVendorRoot } = require('../common/runtimePaths');
const { pickFfmpeg } = require('../common/ffmpeg');

const HEALTH_TIMEOUT_MS = 300000;
const HEALTH_POLL_MS = 400;

/** @type {import('child_process').ChildProcess | null} */
let serverProc = null;
/** @type {Promise<void> | null} */
let terminatePromise = null;
/** @type {number | null} */
let serverPort = null;
/** @type {string | null} */
let conversationId = null;

function isProcAlive(proc) {
  return (
    proc != null && proc.exitCode === null && proc.signalCode === null && !proc.killed
  );
}

/**
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
    if (process.platform === 'win32') {
      execFile('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true }, () => {
        if (!settled) {
          try {
            proc.kill();
          } catch {
            /* ignore */
          }
        }
      });
    } else {
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

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = addr && typeof addr === 'object' ? addr.port : 0;
      srv.close((err) => {
        if (err) reject(err);
        else resolve(port);
      });
    });
    srv.on('error', reject);
  });
}

/**
 * @returns {string}
 */
function pickLlamaServer() {
  if (process.env.GLAUX_LLAMA_SERVER) {
    return process.env.GLAUX_LLAMA_SERVER;
  }
  const bin = findVendorBinary('llamacpp', 'llama-server.exe', 'llama-server');
  if (bin) {
    return bin;
  }
  const name = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server';
  const candidate = path.join(getVendorRoot('llamacpp'), name);
  throw new Error(
    `llama-server not found at ${candidate}. Run "npm run build:llamacpp" or set GLAUX_LLAMA_SERVER.`
  );
}

/**
 * Prefer bundled ffmpeg from vendor/ffmpeg so video decode works offline.
 * That directory is prepended to PATH so sibling ffprobe (required by
 * llama.cpp mtmd video probe) is found as well.
 * @returns {string | undefined}
 */
function pickFfmpegPath() {
  return pickFfmpeg() || undefined;
}

/**
 * @param {string} method
 * @param {string} urlPath
 * @param {object} [options]
 * @returns {Promise<{ status: number, headers: http.IncomingHttpHeaders, body: string }>}
 */
function httpRequest(method, urlPath, options = {}) {
  if (!serverPort) {
    return Promise.reject(new Error('llama-server is not running.'));
  }
  const body = options.body != null ? options.body : null;
  const headers = { ...(options.headers || {}) };
  let payload = null;
  if (body != null && typeof body === 'object' && !Buffer.isBuffer(body)) {
    payload = JSON.stringify(body);
    headers['Content-Type'] = headers['Content-Type'] || 'application/json';
    headers['Content-Length'] = Buffer.byteLength(payload);
  } else if (typeof body === 'string' || Buffer.isBuffer(body)) {
    payload = body;
  }
  const signal = options.signal;

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: '127.0.0.1',
        port: serverPort,
        path: urlPath,
        method,
        headers,
      },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          resolve({
            status: res.statusCode || 0,
            headers: res.headers,
            body: Buffer.concat(chunks).toString('utf8'),
          });
        });
      }
    );
    req.on('error', reject);
    if (signal) {
      if (signal.aborted) {
        req.destroy(new Error('Aborted'));
        return;
      }
      signal.addEventListener(
        'abort',
        () => {
          req.destroy(new Error('Aborted'));
        },
        { once: true }
      );
    }
    if (payload != null) {
      req.write(payload);
    }
    req.end();
  });
}

async function waitForHealth(onProgress, getRecentStderr) {
  const started = Date.now();
  let tick = 0;
  while (Date.now() - started < HEALTH_TIMEOUT_MS) {
    if (!isProcAlive(serverProc)) {
      const detail =
        typeof getRecentStderr === 'function' ? String(getRecentStderr() || '').trim() : '';
      throw new Error(
        detail
          ? `llama-server exited before becoming ready.\n${detail}`
          : 'llama-server exited before becoming ready.'
      );
    }
    try {
      const res = await httpRequest('GET', '/health');
      if (res.status === 200) {
        if (onProgress) {
          onProgress({ status: 'progress', loaded: 100, total: 100, percent: 100 });
        }
        return;
      }
    } catch {
      /* not ready */
    }
    tick += 1;
    if (onProgress) {
      const percent = Math.min(95, 5 + tick * 2);
      onProgress({ status: 'progress', loaded: percent, total: 100, percent });
    }
    await new Promise((r) => setTimeout(r, HEALTH_POLL_MS));
  }
  const detail =
    typeof getRecentStderr === 'function' ? String(getRecentStderr() || '').trim() : '';
  throw new Error(
    detail
      ? `Timed out waiting for llama-server to become ready.\n${detail}`
      : 'Timed out waiting for llama-server to become ready.'
  );
}

/**
 * Spawn llama-server with the given args and wait until `/health` responds.
 * Terminates the process and rethrows if the health check fails.
 * @param {{ bin: string, args: string[], cwd: string, env: NodeJS.ProcessEnv, port: number, onProgress?: Function }} opts
 * @returns {Promise<void>}
 */
async function startServer({ bin, args, cwd, env, port, onProgress }) {
  const proc = spawn(bin, args, {
    cwd,
    env,
    windowsHide: true,
    detached: process.platform !== 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  /** @type {string[]} */
  const stderrChunks = [];
  const STDERR_KEEP = 8;
  proc.stderr.on('data', (chunk) => {
    const text = chunk.toString('utf8');
    stderrChunks.push(text);
    if (stderrChunks.length > STDERR_KEEP) {
      stderrChunks.shift();
    }
    if (process.env.GLAUX_LLAMA_DEBUG === '1') {
      process.stderr.write(`[llama-server] ${text}`);
    }
  });
  proc.on('exit', () => {
    if (serverProc === proc) {
      serverProc = null;
      serverPort = null;
    }
  });

  serverProc = proc;
  serverPort = port;
  conversationId = `glaux-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  const recentStderr = () => stderrChunks.join('').trim().slice(-2000);

  try {
    await waitForHealth(onProgress, recentStderr);
  } catch (err) {
    await stopServer();
    throw err;
  }
}

async function stopServer() {
  conversationId = null;
  serverPort = null;
  if (!serverProc) {
    return terminatePromise || Promise.resolve();
  }
  if (terminatePromise) {
    return terminatePromise;
  }
  const proc = serverProc;
  serverProc = null;
  terminatePromise = terminateProcess(proc).finally(() => {
    terminatePromise = null;
  });
  return terminatePromise;
}

function isRunning() {
  return Boolean(serverProc) && Boolean(serverPort);
}

function getPort() {
  return serverPort;
}

function getConversationId() {
  return conversationId;
}

/**
 * Stream SSE chat completions.
 * Reasoning deltas are wrapped in `<think>…</think>` so the Glaux renderer splits
 * them into the thoughts panel (same markup HF models stream).
 * @param {object} body
 * @param {{ onToken?: (t: string) => void, signal?: AbortSignal }} opts
 * @returns {Promise<{ text: string, usage: object | null }>}
 */
function streamChatCompletions(body, opts = {}) {
  if (!serverPort) {
    return Promise.reject(new Error('llama-server is not running.'));
  }
  const payload = JSON.stringify(body);
  const headers = {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload),
  };
  if (conversationId) {
    headers['X-Conversation-Id'] = conversationId;
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    let full = '';
    let inReasoning = false;
    /** @type {object | null} */
    let usage = null;

    const emit = (text) => {
      if (!text) {
        return;
      }
      full += text;
      if (opts.onToken) {
        opts.onToken(text);
      }
    };
    const closeReasoning = () => {
      if (inReasoning) {
        emit('</think>');
        inReasoning = false;
      }
    };
    const settleOk = () => {
      if (settled) {
        return;
      }
      settled = true;
      closeReasoning();
      resolve({ text: full, usage });
    };
    const settleErr = (err) => {
      if (settled) {
        return;
      }
      if (opts.signal && opts.signal.aborted) {
        settleOk();
        return;
      }
      settled = true;
      reject(err);
    };

    const req = http.request(
      {
        host: '127.0.0.1',
        port: serverPort,
        path: '/v1/chat/completions',
        method: 'POST',
        headers,
      },
      (res) => {
        if (res.statusCode && res.statusCode >= 400) {
          const chunks = [];
          res.on('data', (c) => chunks.push(c));
          res.on('end', () => {
            settleErr(
              new Error(
                Buffer.concat(chunks).toString('utf8') || `llama-server HTTP ${res.statusCode}`
              )
            );
          });
          return;
        }
        let buffer = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          buffer += chunk;
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) {
              continue;
            }
            const data = trimmed.slice(5).trim();
            if (!data || data === '[DONE]') {
              continue;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.usage && typeof parsed.usage === 'object') {
                usage = parsed.usage;
              }
              const deltaObj =
                parsed.choices && parsed.choices[0] && parsed.choices[0].delta
                  ? parsed.choices[0].delta
                  : null;
              if (!deltaObj) {
                continue;
              }
              const reasoning =
                (typeof deltaObj.reasoning_content === 'string' && deltaObj.reasoning_content) ||
                (typeof deltaObj.reasoning === 'string' && deltaObj.reasoning) ||
                '';
              if (reasoning) {
                if (!inReasoning) {
                  emit('<think>');
                  inReasoning = true;
                }
                emit(reasoning);
              }
              const content = typeof deltaObj.content === 'string' ? deltaObj.content : '';
              if (content) {
                closeReasoning();
                emit(content);
              }
            } catch {
              /* ignore partial JSON */
            }
          }
        });
        res.on('end', settleOk);
        res.on('error', settleErr);
      }
    );
    req.on('error', settleErr);
    if (opts.signal) {
      if (opts.signal.aborted) {
        req.destroy();
        settleOk();
        return;
      }
      opts.signal.addEventListener(
        'abort',
        () => {
          req.destroy();
          if (conversationId && serverPort) {
            httpRequest('DELETE', `/v1/stream/${encodeURIComponent(conversationId)}`).catch(
              () => {}
            );
          }
          settleOk();
        },
        { once: true }
      );
    }
    req.write(payload);
    req.end();
  });
}

module.exports = {
  getFreePort,
  pickLlamaServer,
  pickFfmpegPath,
  httpRequest,
  waitForHealth,
  startServer,
  stopServer,
  isRunning,
  getPort,
  getConversationId,
  streamChatCompletions,
};
