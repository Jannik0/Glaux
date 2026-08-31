'use strict';

/**
 * Shared runtime path/binary resolution helpers used by engine backends
 * (huggingface → Python; llamacpp → llama-server; transcribecpp → transcribe-cli;
 *  ffmpeg → ffmpeg + ffprobe).
 * Packaged Electron apps ship bundled runtimes under `process.resourcesPath`;
 * dev checkouts use `vendor/<name>` instead.
 */

const fs = require('fs');
const path = require('path');

function isPackagedApp() {
  try {
    return require('electron').app.isPackaged;
  } catch {
    return false;
  }
}

/**
 * Candidate interpreter paths under a bundled runtime root (vendor/ or resources/python).
 * @param {string} rootDir
 * @returns {string[]}
 */
function bundledPythonCandidates(rootDir) {
  if (process.platform === 'win32') {
    return [path.join(rootDir, 'python.exe')];
  }
  return [
    path.join(rootDir, 'bin', 'python3'),
    path.join(rootDir, 'bin', 'python'),
    path.join(rootDir, 'python3'),
  ];
}

/**
 * @param {string} rootDir
 * @returns {string | null}
 */
function findBundledPython(rootDir) {
  for (const candidate of bundledPythonCandidates(rootDir)) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

/**
 * Prefer PYTHON override, then bundled runtime (packaged or vendor/), then PATH.
 * @returns {string}
 */
function pickPython() {
  if (process.env.PYTHON) {
    return process.env.PYTHON;
  }

  if (isPackagedApp()) {
    const packaged = findBundledPython(path.join(process.resourcesPath, 'python'));
    if (packaged) {
      return packaged;
    }
  }

  const vendorPython = findBundledPython(path.join(__dirname, '..', '..', 'vendor', 'python'));
  if (vendorPython) {
    return vendorPython;
  }

  return process.platform === 'win32' ? 'python' : 'python3';
}

/**
 * Root directory for a bundled vendor tool: `resourcesPath/<name>` when packaged,
 * `vendor/<name>` in dev checkouts.
 * @param {string} name
 * @returns {string}
 */
function getVendorRoot(name) {
  if (isPackagedApp()) {
    return path.join(process.resourcesPath, name);
  }
  return path.join(__dirname, '..', '..', 'vendor', name);
}

/**
 * Absolute path to a named binary inside a vendor root, or null if it doesn't exist.
 * @param {string} vendorName Subfolder under the vendor root (e.g. "llamacpp").
 * @param {string} winName Binary file name on Windows.
 * @param {string} posixName Binary file name on macOS/Linux.
 * @returns {string | null}
 */
function findVendorBinary(vendorName, winName, posixName) {
  const root = getVendorRoot(vendorName);
  const name = process.platform === 'win32' ? winName : posixName;
  const candidate = path.join(root, name);
  return fs.existsSync(candidate) ? candidate : null;
}

module.exports = {
  isPackagedApp,
  bundledPythonCandidates,
  findBundledPython,
  pickPython,
  getVendorRoot,
  findVendorBinary,
};
