#!/usr/bin/env node
'use strict';

/**
 * Run electron-builder. On Linux, stage temp files under dist/.tmp instead of
 * /tmp: that path is often a small tmpfs, and Glaux's unpacked CUDA Torch tree
 * is multi-GB (ENOSPC while writing the deb / rpm / tar.gz). dist/.tmp is
 * removed after electron-builder exits.
 *
 * Override with GLAUX_PACKAGING_TMP=/path if needed (that path is not deleted).
 */

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');

function applyLinuxPackagingTmp() {
  if (process.platform !== 'linux') {
    return null;
  }
  const overridden = Boolean(process.env.GLAUX_PACKAGING_TMP);
  const tmpDir = overridden
    ? path.resolve(process.env.GLAUX_PACKAGING_TMP)
    : path.join(ROOT, 'dist', '.tmp');
  fs.mkdirSync(tmpDir, { recursive: true });
  process.env.TMPDIR = tmpDir;
  process.env.TEMP = tmpDir;
  process.env.TMP = tmpDir;
  console.log(`Packaging temp dir: ${tmpDir}`);
  return { tmpDir, cleanup: !overridden };
}

const packagingTmp = applyLinuxPackagingTmp();

const cli = require.resolve('electron-builder/cli.js');
const result = spawnSync(process.execPath, [cli, ...process.argv.slice(2)], {
  stdio: 'inherit',
  env: process.env,
});

if (packagingTmp && packagingTmp.cleanup) {
  try {
    fs.rmSync(packagingTmp.tmpDir, { recursive: true, force: true });
  } catch (err) {
    console.warn(`Could not remove packaging temp dir ${packagingTmp.tmpDir}: ${err.message}`);
  }
}

process.exit(result.status === null ? 1 : result.status);
