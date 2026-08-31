'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { resolveLanguage, DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES } = require('../src/i18n/languages');
const { t, flattenKeys } = require('../src/i18n/translate');
const en = require('../src/i18n/locales/en.json');
const de = require('../src/i18n/locales/de.json');
const fr = require('../src/i18n/locales/fr.json');
const es = require('../src/i18n/locales/es.json');
const itCatalog = require('../src/i18n/locales/it.json');
const pt = require('../src/i18n/locales/pt.json');

describe('resolveLanguage', () => {
  it('uses a supported persisted language over the OS locale', () => {
    assert.equal(resolveLanguage('de', 'fr-FR'), 'de');
    assert.equal(resolveLanguage('pt', 'en-US'), 'pt');
  });

  it('maps OS locale primary tags to supported languages', () => {
    assert.equal(resolveLanguage(null, 'de-DE'), 'de');
    assert.equal(resolveLanguage(undefined, 'pt-BR'), 'pt');
    assert.equal(resolveLanguage('', 'fr_CA'), 'fr');
    assert.equal(resolveLanguage(null, 'es'), 'es');
    assert.equal(resolveLanguage(null, 'it-IT'), 'it');
  });

  it('falls back to English when the OS locale is unknown', () => {
    assert.equal(resolveLanguage(null, 'ja-JP'), DEFAULT_LANGUAGE);
    assert.equal(resolveLanguage(null, 'zh-CN'), 'en');
    assert.equal(resolveLanguage(null, ''), 'en');
    assert.equal(resolveLanguage(null, null), 'en');
  });

  it('treats an invalid persisted value as unset and uses the OS locale', () => {
    assert.equal(resolveLanguage('ja', 'de-DE'), 'de');
    assert.equal(resolveLanguage('not-a-locale', 'en-US'), 'en');
    assert.equal(resolveLanguage('zz', 'sv-SE'), 'en');
  });

  it('uses English when persisted is missing and the OS is English', () => {
    assert.equal(resolveLanguage(null, 'en-US'), 'en');
    assert.equal(resolveLanguage(undefined, 'en'), 'en');
  });
});

describe('chromiumLocaleTag', () => {
  const { chromiumLocaleTag } = require('../src/i18n/languages');

  it('maps Glaux languages to Chromium pack tags', () => {
    assert.equal(chromiumLocaleTag('de', 'en-US'), 'de');
    assert.equal(chromiumLocaleTag('fr', 'de-DE'), 'fr');
    assert.equal(chromiumLocaleTag('it', null), 'it');
  });

  it('picks regional English, Portuguese, and Spanish from the OS locale', () => {
    assert.equal(chromiumLocaleTag('en', 'en-GB'), 'en-GB');
    assert.equal(chromiumLocaleTag('en', 'en_GB'), 'en-GB');
    assert.equal(chromiumLocaleTag('en', 'de-DE'), 'en-US');
    assert.equal(chromiumLocaleTag('pt', 'pt-PT'), 'pt-PT');
    assert.equal(chromiumLocaleTag('pt', 'de-DE'), 'pt-BR');
    assert.equal(chromiumLocaleTag('es', 'es-MX'), 'es-419');
    assert.equal(chromiumLocaleTag('es', 'es-ES'), 'es');
  });
});

describe('isSupportedElectronLocaleFile', () => {
  const { isSupportedElectronLocaleFile } = require('../src/i18n/languages');

  it('keeps Chromium packs for Glaux languages and drops the rest', () => {
    assert.equal(isSupportedElectronLocaleFile('en-US'), true);
    assert.equal(isSupportedElectronLocaleFile('en_GB'), true);
    assert.equal(isSupportedElectronLocaleFile('English'), true);
    assert.equal(isSupportedElectronLocaleFile('de'), true);
    assert.equal(isSupportedElectronLocaleFile('pt-BR'), true);
    assert.equal(isSupportedElectronLocaleFile('es-419'), true);
    assert.equal(isSupportedElectronLocaleFile('ja'), false);
    assert.equal(isSupportedElectronLocaleFile('zh-CN'), false);
    assert.equal(isSupportedElectronLocaleFile('ru'), false);
  });
});

describe('t()', () => {
  it('interpolates named placeholders', () => {
    assert.equal(
      t(en, 'chat.contextUsed', { used: '1.2k', total: '8k' }),
      'Context used: 1.2k/8k',
    );
    assert.equal(
      t(en, 'panels.models.loadingNamed', { modelId: 'org/model' }),
      'Loading org/model…',
    );
  });

  it('falls back to English when a key is missing from the active catalog', () => {
    const partial = { chat: { send: 'Enviar' } };
    assert.equal(t(partial, 'chat.send'), 'Enviar');
    assert.equal(t(partial, 'chat.modelReady', undefined, en), 'Model ready');
  });

  it('returns the key when it is missing from both catalogs', () => {
    assert.equal(t({}, 'missing.key', undefined, {}), 'missing.key');
  });

  it('uses the _one variant when count is 1', () => {
    assert.equal(
      t(en, 'panels.resources.added', { count: 1, uploadedCount: 1 }),
      'Added 1 item.',
    );
    assert.equal(
      t(en, 'panels.resources.added', { count: 3, uploadedCount: 3 }),
      'Added 3 items.',
    );
  });
});

describe('locale key parity', () => {
  const enKeys = flattenKeys(en).sort();

  it('has a non-empty English catalog', () => {
    assert.ok(enKeys.length > 50);
  });

  for (const [code, catalog] of [
    ['de', de],
    ['fr', fr],
    ['es', es],
    ['it', itCatalog],
    ['pt', pt],
  ]) {
    it(`${code} has the same keys as en`, () => {
      const keys = flattenKeys(catalog).sort();
      assert.deepEqual(keys, enKeys);
    });
  }

  it('lists the supported language codes', () => {
    assert.deepEqual([...SUPPORTED_LANGUAGES], ['en', 'de', 'fr', 'es', 'it', 'pt']);
  });
});
