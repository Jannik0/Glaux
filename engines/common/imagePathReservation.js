'use strict';

/**
 * In-memory claims for session sidecar paths. allocateSessionImagePath records
 * a name before the file exists; the run releases it when the file is written,
 * discarded, or the run ends. fs.access alone cannot see a claim that has not
 * been created on disk yet.
 */

const path = require('path');

/** @type {Set<string>} */
const reservedSessionImagePaths = new Set();

/**
 * @param {string} absolutePath
 * @returns {string}
 */
function reservationKey(absolutePath) {
  return path.resolve(absolutePath);
}

/**
 * @param {string} absolutePath
 * @returns {boolean} false when that path is already reserved
 */
function reserveSessionImagePath(absolutePath) {
  const key = reservationKey(absolutePath);
  if (reservedSessionImagePaths.has(key)) {
    return false;
  }
  reservedSessionImagePaths.add(key);
  return true;
}

/**
 * @param {string | undefined | null} absolutePath
 */
function releaseSessionImagePath(absolutePath) {
  if (typeof absolutePath !== 'string' || !absolutePath) {
    return;
  }
  reservedSessionImagePaths.delete(reservationKey(absolutePath));
}

module.exports = {
  releaseSessionImagePath,
  reserveSessionImagePath,
};
