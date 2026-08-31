(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
  if (root && typeof root === 'object') {
    root.Glaux = root.Glaux || {};
    root.Glaux._translate = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /**
   * @param {unknown} catalog
   * @param {string} key
   * @returns {string | undefined}
   */
  function lookup(catalog, key) {
    if (!catalog || typeof catalog !== 'object' || typeof key !== 'string' || !key) {
      return undefined;
    }
    const parts = key.split('.');
    let node = catalog;
    for (const part of parts) {
      if (!node || typeof node !== 'object' || !(part in node)) {
        return undefined;
      }
      node = node[part];
    }
    return typeof node === 'string' ? node : undefined;
  }

  /**
   * @param {string} template
   * @param {Record<string, unknown> | null | undefined} vars
   * @returns {string}
   */
  function interpolate(template, vars) {
    if (!vars || typeof vars !== 'object') {
      return template;
    }
    return template.replace(/\{(\w+)\}/g, (match, name) => {
      if (Object.prototype.hasOwnProperty.call(vars, name)) {
        const value = vars[name];
        return value == null ? '' : String(value);
      }
      return match;
    });
  }

  /**
   * @param {unknown} catalog
   * @param {string} key
   * @param {number | undefined} count
   * @returns {string | undefined}
   */
  function resolveKey(catalog, key, count) {
    if (typeof count === 'number' && count === 1) {
      const one = lookup(catalog, `${key}_one`);
      if (one !== undefined) {
        return one;
      }
    }
    return lookup(catalog, key);
  }

  /**
   * @param {unknown} catalog
   * @param {string} key
   * @param {Record<string, unknown> | null | undefined} vars
   * @param {unknown} fallbackCatalog
   * @returns {string}
   */
  function t(catalog, key, vars, fallbackCatalog) {
    const count = vars && typeof vars.count === 'number' ? vars.count : undefined;
    const fromCatalog = resolveKey(catalog, key, count);
    if (fromCatalog !== undefined) {
      return interpolate(fromCatalog, vars);
    }
    const fromFallback = resolveKey(fallbackCatalog, key, count);
    if (fromFallback !== undefined) {
      return interpolate(fromFallback, vars);
    }
    return key;
  }

  /**
   * @param {unknown} obj
   * @param {string} [prefix]
   * @param {string[]} [out]
   * @returns {string[]}
   */
  function flattenKeys(obj, prefix, out) {
    const keys = out || [];
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
      return keys;
    }
    for (const [k, v] of Object.entries(obj)) {
      const path = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        flattenKeys(v, path, keys);
      } else {
        keys.push(path);
      }
    }
    return keys;
  }

  return {
    lookup,
    interpolate,
    t,
    flattenKeys,
  };
});
