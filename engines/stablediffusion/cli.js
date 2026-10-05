'use strict';

/**
 * Resolve and spawn the bundled sd-cli binary (stable-diffusion.cpp).
 */

const { spawn } = require('child_process');
const path = require('path');
const { findVendorBinary } = require('../common/runtimePaths');
const { withVendorLibPath, withSharedCudaLibPath, withUnsupportedCudaHidden } = require('../common/gpuRuntime');

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
 * @param {{ signal?: AbortSignal, cwd?: string, env?: NodeJS.ProcessEnv }} [opts]
 * @returns {Promise<{ code: number | null, stdout: string, stderr: string }>}
 */
function runSdCli(args, opts = {}) {
  const bin = pickSdCli();
  const binDir = path.dirname(bin);
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      windowsHide: true,
      cwd: opts.cwd || binDir,
      env: withUnsupportedCudaHidden(
        withSharedCudaLibPath(withVendorLibPath(opts.env || process.env, binDir))
      ),
    });

    let stdout = '';
    let stderr = '';
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
      stdout += chunk.toString('utf8');
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
      resolve({ code, stdout, stderr });
    });
  });
}

module.exports = {
  pickSdCli,
  runSdCli,
};
