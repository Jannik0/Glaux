'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const {
  assertValidEntryName,
  toPosixRelative,
  resolveWorkspacePath,
  isSubPath,
} = require('../src/main/pathSandbox');

describe('pathSandbox', () => {
  it('accepts and trims valid entry names', () => {
    assert.equal(assertValidEntryName('  notes.md  '), 'notes.md');
  });

  it('rejects empty, dots, and path separators', () => {
    assert.throws(() => assertValidEntryName(''), /empty/i);
    assert.throws(() => assertValidEntryName('..'), /not valid/i);
    assert.throws(() => assertValidEntryName('a/b'), /separators/i);
    assert.throws(() => assertValidEntryName('a\\b'), /separators/i);
  });

  it('normalizes relative paths to posix', () => {
    assert.equal(toPosixRelative(path.join('a', 'b', 'c')), 'a/b/c');
    assert.equal(toPosixRelative(''), '');
  });

  it('resolves paths inside a workspace root', () => {
    const root = path.resolve('/tmp/glaux-resources');
    const resolved = resolveWorkspacePath(root, 'folder/file.md', 'escape');
    assert.equal(resolved, path.resolve(root, 'folder/file.md'));
  });

  it('rejects absolute and escaping relative paths', () => {
    const root = path.resolve('/tmp/glaux-resources');
    assert.throws(
      () => resolveWorkspacePath(root, path.resolve('/etc/passwd'), 'escape'),
      /Absolute paths/
    );
    assert.throws(
      () => resolveWorkspacePath(root, '../outside.txt', 'Path escapes sandbox.'),
      /Path escapes sandbox/
    );
  });

  it('detects subpaths', () => {
    const parent = path.resolve('/tmp/root');
    assert.equal(isSubPath(parent, parent), true);
    assert.equal(isSubPath(parent, path.join(parent, 'a')), true);
    assert.equal(isSubPath(parent, path.resolve('/tmp/other')), false);
  });
});
