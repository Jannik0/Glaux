'use strict';

/**
 * Clone a git repo into dest at a pinned revision when dest is missing.
 * Existing checkouts are left untouched so local work is not overwritten.
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const { which } = require('./gpuBackends');

function runGit(args, options = {}) {
  console.log(`> git ${args.join(' ')}`);
  const result = spawnSync('git', args, {
    stdio: 'inherit',
    shell: process.platform === 'win32',
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
function ensureGitDep({ dest, url, rev, name }) {
  if (fs.existsSync(dest)) {
    console.log(`Using existing ${name} at ${dest}`);
    return;
  }
  if (!which('git')) {
    throw new Error(`git not found on PATH. Install Git to clone ${name} into ${dest}.`);
  }

  console.log(`Cloning ${name} (${rev}) into ${dest}`);
  try {
    runGit(['init', dest]);
    runGit(['-C', dest, 'remote', 'add', 'origin', url]);
    runGit(['-C', dest, 'fetch', '--depth', '1', 'origin', rev]);
    runGit(['-C', dest, '-c', 'advice.detachedHead=false', 'checkout', '--force', 'FETCH_HEAD']);
  } catch (err) {
    fs.rmSync(dest, { recursive: true, force: true });
    throw err;
  }
}

module.exports = { ensureGitDep };
