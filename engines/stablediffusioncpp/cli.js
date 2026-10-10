'use strict';

/**
 * Resolve and spawn the bundled sd-cli binary (stable-diffusion.cpp).
 */

const { spawn } = require('child_process');
const path = require('path');
const { findVendorBinary } = require('../common/runtimePaths');
const { withVendorLibPath, withSharedCudaLibPath, withUnsupportedCudaHidden } = require('../common/gpuRuntime');

/** Keep the tail of sd-cli logs so a long run cannot grow without a bound. */
const SD_CLI_OUTPUT_TAIL_BYTES = 256 * 1024;
/** After SIGTERM, SIGKILL a child that is still running. */
const SD_CLI_KILL_GRACE_MS = 1000;

/**
 * @param {{ parts: Buffer[], bytes: number }} state
 * @param {Buffer | string} chunk
 * @param {number} cap
 */
function pushOutputTail(state, chunk, cap) {
  const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
  state.parts.push(buf);
  state.bytes += buf.length;
  while (state.bytes > cap && state.parts.length > 1) {
    state.bytes -= state.parts.shift().length;
  }
  if (state.parts.length === 1 && state.parts[0].length > cap) {
    state.parts[0] = state.parts[0].subarray(state.parts[0].length - cap);
    state.bytes = state.parts[0].length;
  }
}

/**
 * Drop a leading partial UTF-8 sequence left by a byte-cap cut.
 * @param {Buffer} buf
 * @returns {Buffer}
 */
function trimUtf8Start(buf) {
  let i = 0;
  while (i < buf.length && (buf[i] & 0xc0) === 0x80) {
    i += 1;
  }
  return i === 0 ? buf : buf.subarray(i);
}

/**
 * @param {{ parts: Buffer[], bytes: number }} state
 * @returns {string}
 */
function outputTailString(state) {
  if (!state.parts.length || state.bytes <= 0) {
    return '';
  }
  return trimUtf8Start(Buffer.concat(state.parts, state.bytes)).toString('utf8');
}

/**
 * @returns {string}
 */
function pickSdCli() {
  if (process.env.GLAUX_SD_CLI) {
    return process.env.GLAUX_SD_CLI;
  }
  const found = findVendorBinary('stablediffusion', 'sd-cli.exe', 'sd-cli');
  if (found) {
    return found;
  }
  throw new Error(
    'sd-cli not found. Run: npm run build:stablediffusion\n' +
      '(Requires Git, CMake, and a C++ toolchain.)'
  );
}

/**
 * Spawn sd-cli and collect stdout/stderr.
 *
 * @param {string[]} args
 * @param {{ signal?: AbortSignal, cwd?: string, env?: NodeJS.ProcessEnv, bin?: string, killGraceMs?: number, outputTailBytes?: number }} [opts]
 * @returns {Promise<{ code: number | null, stdout: string, stderr: string }>}
 */
function runSdCli(args, opts = {}) {
  const bin = opts.bin || pickSdCli();
  const binDir = path.dirname(bin);
  const graceMs = Number.isFinite(opts.killGraceMs) ? opts.killGraceMs : SD_CLI_KILL_GRACE_MS;
  const tailBytes = Number.isFinite(opts.outputTailBytes) ? opts.outputTailBytes : SD_CLI_OUTPUT_TAIL_BYTES;
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      windowsHide: true,
      cwd: opts.cwd || binDir,
      env: withUnsupportedCudaHidden(
        withSharedCudaLibPath(withVendorLibPath(opts.env || process.env, binDir))
      ),
    });

    const stdoutTail = { parts: [], bytes: 0 };
    const stderrTail = { parts: [], bytes: 0 };
    let settled = false;
    /** @type {NodeJS.Timeout | null} */
    let killTimer = null;

    const clearKillTimer = () => {
      if (killTimer) {
        clearTimeout(killTimer);
        killTimer = null;
      }
    };

    const onAbort = () => {
      try {
        child.kill();
      } catch {
        /* ignore */
      }
      if (killTimer || settled) {
        return;
      }
      killTimer = setTimeout(() => {
        if (settled) {
          return;
        }
        try {
          child.kill('SIGKILL');
        } catch {
          /* already exited */
        }
      }, graceMs);
      if (typeof killTimer.unref === 'function') {
        killTimer.unref();
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
      pushOutputTail(stdoutTail, chunk, tailBytes);
    });

    child.stderr.on('data', (chunk) => {
      pushOutputTail(stderrTail, chunk, tailBytes);
    });

    child.on('error', (err) => {
      if (settled) {
        return;
      }
      settled = true;
      clearKillTimer();
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
      clearKillTimer();
      if (opts.signal) {
        opts.signal.removeEventListener('abort', onAbort);
      }
      resolve({
        code,
        stdout: outputTailString(stdoutTail),
        stderr: outputTailString(stderrTail),
      });
    });
  });
}

module.exports = {
  pickSdCli,
  runSdCli,
  pushOutputTail,
  outputTailString,
  SD_CLI_KILL_GRACE_MS,
  SD_CLI_OUTPUT_TAIL_BYTES,
};
