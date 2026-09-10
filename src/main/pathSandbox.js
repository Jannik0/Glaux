'use strict';

/**
 * Pure path-sandbox helpers (no Electron). Used by paths.js / workspaceFs and unit tests.
 */

const path = require('path');
const { t } = require('../i18n');
const { isValidOsFolderName } = require('../renderer/shared/osFolderName');

/**
 * @param {unknown} name
 * @returns {string}
 */
function assertValidEntryName(name) {
  if (typeof name !== 'string') {
    throw new Error(t('errors.pathSandbox.nameMustBeString'));
  }

  const trimmedName = name.trim();
  if (!trimmedName) {
    throw new Error(t('errors.pathSandbox.nameCannotBeEmpty'));
  }

  if (trimmedName === '.' || trimmedName === '..') {
    throw new Error(t('errors.pathSandbox.nameNotValid'));
  }

  if (trimmedName.includes('/') || trimmedName.includes('\\')) {
    throw new Error(t('errors.pathSandbox.nameCannotContainSeparators'));
  }

  return trimmedName;
}

/**
 * Like assertValidEntryName, plus Windows/macOS/Linux folder-name rules.
 * Use for *new* names only so existing on-disk entries can still be referenced.
 * @param {unknown} name
 * @returns {string}
 */
function assertValidOsFolderName(name) {
  const trimmedName = assertValidEntryName(name);
  if (!isValidOsFolderName(trimmedName)) {
    throw new Error(t('errors.pathSandbox.nameNotValid'));
  }
  return trimmedName;
}

/**
 * @param {string} relativePath
 * @returns {string}
 */
function toPosixRelative(relativePath) {
  if (!relativePath) {
    return '';
  }

  return relativePath.split(path.sep).join('/');
}

/**
 * @param {string} root
 * @param {string} [relativePath]
 * @param {string} escapeMessage
 * @returns {string}
 */
function resolveWorkspacePath(root, relativePath = '', escapeMessage) {
  if (typeof relativePath !== 'string') {
    throw new Error(t('errors.pathSandbox.pathMustBeString'));
  }

  if (path.isAbsolute(relativePath)) {
    throw new Error(t('errors.pathSandbox.absolutePathsNotAllowed'));
  }

  const resolvedPath = path.resolve(root, relativePath);
  const relativeToRoot = path.relative(root, resolvedPath);

  if (
    relativeToRoot === '..' ||
    relativeToRoot.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativeToRoot)
  ) {
    throw new Error(escapeMessage);
  }

  return resolvedPath;
}

/**
 * @param {string} parentPath
 * @param {string} candidatePath
 * @returns {boolean}
 */
function isSubPath(parentPath, candidatePath) {
  const relative = path.relative(parentPath, candidatePath);
  return (
    relative === '' ||
    (!relative.startsWith(`..${path.sep}`) &&
      relative !== '..' &&
      !path.isAbsolute(relative))
  );
}

module.exports = {
  assertValidEntryName,
  assertValidOsFolderName,
  isValidOsFolderName,
  toPosixRelative,
  resolveWorkspacePath,
  isSubPath,
};
