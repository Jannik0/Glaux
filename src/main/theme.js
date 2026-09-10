'use strict';

const SUPPORTED_THEMES = Object.freeze(['system', 'light', 'dark']);

const DEFAULT_THEME = 'system';

/**
 * @param {unknown} value
 * @returns {value is string}
 */
function isSupportedTheme(value) {
  return typeof value === 'string' && SUPPORTED_THEMES.includes(value);
}

/**
 * @param {unknown} persisted
 * @returns {'system' | 'light' | 'dark'}
 */
function resolveTheme(persisted) {
  return isSupportedTheme(persisted) ? persisted : DEFAULT_THEME;
}

module.exports = {
  SUPPORTED_THEMES,
  DEFAULT_THEME,
  isSupportedTheme,
  resolveTheme,
};
