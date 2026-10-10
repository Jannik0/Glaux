'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { ensureGitDep, isEmptyDirectory } = require('../scripts/ensureGitDep');

describe('ensureGitDep', () => {
  it('treats an empty directory as missing and leaves a populated checkout alone', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-gitdep-'));
    const empty = path.join(root, 'empty');
    const filled = path.join(root, 'filled');
    fs.mkdirSync(empty);
    fs.mkdirSync(filled);
    fs.writeFileSync(path.join(filled, 'CMakeLists.txt'), 'project(x)\n');
    try {
      assert.equal(isEmptyDirectory(empty), true);
      assert.equal(isEmptyDirectory(filled), false);
      assert.equal(isEmptyDirectory(path.join(root, 'missing')), false);

      const skipped = [];
      ensureGitDep({
        dest: filled,
        url: 'https://example.invalid/repo.git',
        rev: 'abc',
        name: 'filled',
        git: (args) => {
          skipped.push(args);
        },
      });
      assert.deepEqual(skipped, []);

      const commands = [];
      ensureGitDep({
        dest: empty,
        url: 'https://example.invalid/libwebp.git',
        rev: '0c9546f',
        name: 'libwebp',
        git: (args) => {
          commands.push(args.join(' '));
        },
      });
      assert.deepEqual(commands, [
        `init ${empty}`,
        `-C ${empty} remote add origin https://example.invalid/libwebp.git`,
        `-C ${empty} fetch --depth 1 origin 0c9546f`,
        `-C ${empty} -c advice.detachedHead=false checkout --force FETCH_HEAD`,
      ]);
      assert.equal(fs.existsSync(empty), true);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
