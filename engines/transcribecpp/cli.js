'use strict';

/**
 * Resolve and spawn the bundled transcribe-cli binary.
 */

const { spawn } = require('child_process');
const path = require('path');
const { findVendorBinary } = require('../common/runtimePaths');
const { withVendorLibPath, withSharedCudaLibPath } = require('../common/gpuRuntime');

/**
 * @returns {string}
 */
function pickTranscribeCli() {
  if (process.env.GLAUX_TRANSCRIBE_CLI) {
    return process.env.GLAUX_TRANSCRIBE_CLI;
  }
  const found = findVendorBinary('transcribe', 'transcribe-cli.exe', 'transcribe-cli');
  if (found) {
    return found;
  }
  throw new Error(
    'transcribe-cli not found. Run: npm run build:transcribe\n' +
      '(Requires Git, CMake, and a C++ toolchain.)'
  );
}

/**
 * Spawn transcribe-cli and collect stdout/stderr. Emits line-oriented stdout
 * via onStdoutLine for progressive parsing.
 *
 * @param {string[]} args
 * @param {{ signal?: AbortSignal, onStdoutLine?: (line: string) => void, cwd?: string, env?: NodeJS.ProcessEnv }} [opts]
 * @returns {Promise<{ code: number | null, stdout: string, stderr: string }>}
 */
function runTranscribeCli(args, opts = {}) {
  const bin = pickTranscribeCli();
  const binDir = path.dirname(bin);
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      windowsHide: true,
      cwd: opts.cwd || binDir,
      env: withSharedCudaLibPath(withVendorLibPath(opts.env || process.env, binDir)),
    });

    let stdout = '';
    let stderr = '';
    let lineBuf = '';
    let settled = false;

    const onAbort = () => {
      try {
        child.kill();
      } catch {
        /* ignore */
      }
    };

    if (opts.signal) {
      if (opts.signal.aborted) {
        onAbort();
      } else {
        opts.signal.addEventListener('abort', onAbort, { once: true });
      }
    }

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString('utf8');
      stdout += text;
      if (typeof opts.onStdoutLine === 'function') {
        lineBuf += text;
        let idx;
        while ((idx = lineBuf.indexOf('\n')) >= 0) {
          const line = lineBuf.slice(0, idx).replace(/\r$/, '');
          lineBuf = lineBuf.slice(idx + 1);
          if (line.length) {
            opts.onStdoutLine(line);
          }
        }
      }
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString('utf8');
    });

    child.on('error', (err) => {
      if (settled) {
        return;
      }
      settled = true;
      if (opts.signal) {
        opts.signal.removeEventListener('abort', onAbort);
      }
      reject(err);
    });

    child.on('close', (code) => {
      if (settled) {
        return;
      }
      settled = true;
      if (opts.signal) {
        opts.signal.removeEventListener('abort', onAbort);
      }
      if (lineBuf.trim() && typeof opts.onStdoutLine === 'function') {
        opts.onStdoutLine(lineBuf.replace(/\r$/, ''));
        lineBuf = '';
      }
      resolve({ code, stdout, stderr });
    });
  });
}

module.exports = {
  pickTranscribeCli,
  runTranscribeCli,
};
