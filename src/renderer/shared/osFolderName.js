// Shared folder-name rules for Windows, macOS, and Linux.
// Loaded in the renderer (script tag) and required from the main process.
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
  if (root && typeof root === 'object') {
    root.Glaux = root.Glaux || {};
    root.Glaux.OsFolderName = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // Windows forbids <>:"/\|?* and C0 controls; Linux/macOS forbid NUL and `/`;
  // macOS/HFS also treats `:` as a separator. Intersection of all three.
  const INVALID_CHARS = /[<>:"/\\|?*\u0000-\u001f]/;
  const WINDOWS_RESERVED = /^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(\..*)?$/i;
  const MAX_NAME_UNITS = 255;
  const MAX_UTF8_BYTES = 255;

  /**
   * @param {string} name
   * @returns {number}
   */
  function utf8ByteLength(name) {
    return new TextEncoder().encode(name).length;
  }

  /**
   * True if `name` is a legal single-segment folder name on Windows, macOS, and Linux.
   * Pass an already-trimmed name (leading/trailing ASCII spaces are not valid either).
   * @param {unknown} name
   * @returns {boolean}
   */
  function isValidOsFolderName(name) {
    if (typeof name !== 'string' || !name) {
      return false;
    }
    if (name === '.' || name === '..') {
      return false;
    }
    if (name.length > MAX_NAME_UNITS || utf8ByteLength(name) > MAX_UTF8_BYTES) {
      return false;
    }
    if (INVALID_CHARS.test(name)) {
      return false;
    }
    // Win32 strips trailing dots and spaces, leaving a folder that cannot be opened or deleted normally.
    if (name.endsWith('.') || name.endsWith(' ')) {
      return false;
    }
    if (WINDOWS_RESERVED.test(name)) {
      return false;
    }
    return true;
  }

  return { isValidOsFolderName };
});
