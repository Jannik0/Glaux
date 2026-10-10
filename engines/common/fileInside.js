'use strict';

/**
 * Path confinement shared by the engine and the session image sandbox.
 * Engines must not import src/main; session code imports this helper.
 */

const path = require('path');

/**
 * @param {string} parentPath
 * @param {string} candidatePath
 * @returns {boolean}
 */
function isSubPath(parentPath, candidatePath) {
  const relative = path.relative(parentPath, candidatePath);
  return (
    relative === '' ||
    (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
  );
}

/**
 * True when candidate is a path strictly inside root (not the root itself).
 * @param {string} root
 * @param {string} candidate
 * @returns {boolean}
 */
function isFileInside(root, candidate) {
  if (!root || !candidate) {
    return false;
  }
  const parent = path.resolve(root);
  const file = path.resolve(candidate);
  if (file === parent) {
    return false;
  }
  return isSubPath(parent, file);
}

module.exports = {
  isFileInside,
  isSubPath,
};
