'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const {
  resolveTheme,
  isSupportedTheme,
  DEFAULT_THEME,
  SUPPORTED_THEMES,
} = require('../src/main/theme');

describe('resolveTheme', () => {
  it('keeps a supported persisted theme', () => {
    assert.equal(resolveTheme('light'), 'light');
    assert.equal(resolveTheme('dark'), 'dark');
    assert.equal(resolveTheme('system'), 'system');
  });

  it('falls back to system when the persisted value is missing or invalid', () => {
    assert.equal(resolveTheme(null), DEFAULT_THEME);
    assert.equal(resolveTheme(undefined), 'system');
    assert.equal(resolveTheme(''), 'system');
    assert.equal(resolveTheme('auto'), 'system');
    assert.equal(resolveTheme('Light'), 'system');
  });
});

describe('isSupportedTheme', () => {
  it('accepts only system, light, and dark', () => {
    assert.deepEqual([...SUPPORTED_THEMES], ['system', 'light', 'dark']);
    assert.equal(isSupportedTheme('system'), true);
    assert.equal(isSupportedTheme('light'), true);
    assert.equal(isSupportedTheme('dark'), true);
    assert.equal(isSupportedTheme('auto'), false);
    assert.equal(isSupportedTheme(null), false);
  });
});
