'use strict';

/**
 * Clone a git repo into dest at a pinned revision when dest is missing.
 * Existing checkouts are left untouched so local work is not overwritten.
 * An empty directory is not a checkout: git leaves one for an unpopulated
 * submodule gitlink, and that placeholder must still be filled in.
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const { which } = require('./gpuBackends');

function runGit(args, options = {}) {
  console.log(`> git ${args.join(' ')}`);
  // Windows joins into one string: Node warns (DEP0190) when shell:true is
  // paired with an args array, and that array is only concatenated anyway.
  const shell = process.platform === 'win32';
  const result = spawnSync(shell ? ['git', ...args].join(' ') : 'git', shell ? [] : args, {
    stdio: 'inherit',
    shell,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed (${result.status}): git ${args.join(' ')}`);
  }
}

/**
 * @param {{ dest: string, url: string, rev: string, name: string }} opts
 */
function isEmptyDirectory(dest) {
  try {
    return fs.statSync(dest).isDirectory() && fs.readdirSync(dest).length === 0;
  } catch {
    return false;
  }
}

/**
 * @param {{ dest: string, url: string, rev: string, name: string, git?: typeof runGit }} opts
 */
function ensureGitDep({ dest, url, rev, name, git }) {
  if (fs.existsSync(dest) && !isEmptyDirectory(dest)) {
    console.log(`Using existing ${name} at ${dest}`);
    return;
  }
  const run = typeof git === 'function' ? git : runGit;
  if (run === runGit && !which('git')) {
    throw new Error(`git not found on PATH. Install Git to clone ${name} into ${dest}.`);
  }

  console.log(`Cloning ${name} (${rev}) into ${dest}`);
  try {
    run(['init', dest]);
    run(['-C', dest, 'remote', 'add', 'origin', url]);
    run(['-C', dest, 'fetch', '--depth', '1', 'origin', rev]);
    run(['-C', dest, '-c', 'advice.detachedHead=false', 'checkout', '--force', 'FETCH_HEAD']);
  } catch (err) {
    fs.rmSync(dest, { recursive: true, force: true });
    throw err;
  }
}

module.exports = { ensureGitDep, isEmptyDirectory };
