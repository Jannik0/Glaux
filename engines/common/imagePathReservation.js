'use strict';

/**
 * In-memory claims for session sidecar paths. allocateSessionImagePath records
 * a name before the file exists; the run releases it when the file is written,
 * discarded, or the run ends. fs.access alone cannot see a claim that has not
 * been created on disk yet.
 *
 * Each claim has an owner token. A later release from the previous owner does
 * not free a name that a newer run has claimed.
 */

const path = require('path');

/** @type {Map<string, number>} */
const reservedSessionImagePaths = new Map();
let nextReservationId = 1;

/**
 * @param {string} absolutePath
 * @returns {string}
 */
function reservationKey(absolutePath) {
  return path.resolve(absolutePath);
}

/**
 * @param {string} absolutePath
 * @returns {number | null} owner token, or null when that path is already reserved
 */
function reserveSessionImagePath(absolutePath) {
  if (typeof absolutePath !== 'string' || !absolutePath) {
    return null;
  }
  const key = reservationKey(absolutePath);
  if (reservedSessionImagePaths.has(key)) {
    return null;
  }
  const owner = nextReservationId;
  nextReservationId += 1;
  reservedSessionImagePaths.set(key, owner);
  return owner;
}

/**
 * Drop this owner's claim. A second release by the same owner is a no-op, and
 * a release by an owner who no longer holds the path leaves the current claim.
 *
 * @param {string | undefined | null} absolutePath
 * @param {number | undefined | null} owner
 */
function releaseSessionImagePath(absolutePath, owner) {
  if (typeof absolutePath !== 'string' || !absolutePath) {
    return;
  }
  if (owner == null) {
    console.warn('releaseSessionImagePath called without an owner token:', absolutePath);
    return;
  }
  const key = reservationKey(absolutePath);
  if (reservedSessionImagePaths.get(key) !== owner) {
    return;
  }
  reservedSessionImagePaths.delete(key);
}

module.exports = {
  releaseSessionImagePath,
  reserveSessionImagePath,
};
