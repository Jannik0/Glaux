'use strict';

const {
  resolveLanguage,
  isSupportedLanguage,
  getAvailableLanguages,
  DEFAULT_LANGUAGE,
  SUPPORTED_LANGUAGES,
  chromiumLocaleTag,
  primaryLanguageTag,
} = require('./languages');
const { t: translate } = require('./translate');

const catalogs = {
  en: require('./locales/en.json'),
  de: require('./locales/de.json'),
  fr: require('./locales/fr.json'),
  es: require('./locales/es.json'),
  it: require('./locales/it.json'),
  pt: require('./locales/pt.json'),
};

let currentLanguage = DEFAULT_LANGUAGE;

/**
 * @param {unknown} code
 */
function setCurrentLanguage(code) {
  currentLanguage = isSupportedLanguage(code) ? code : DEFAULT_LANGUAGE;
}

function getCurrentLanguage() {
  return currentLanguage;
}

/**
 * @param {unknown} code
 * @returns {Record<string, unknown>}
 */
function getCatalog(code) {
  const lang = isSupportedLanguage(code) ? code : DEFAULT_LANGUAGE;
  return catalogs[lang] || catalogs.en;
}

/**
 * @param {string} key
 * @param {Record<string, unknown>} [vars]
 * @returns {string}
 */
function t(key, vars) {
  return translate(getCatalog(currentLanguage), key, vars, catalogs.en);
}

/**
 * @param {string} key
 * @param {Record<string, unknown>} [vars]
 * @param {string} [code]
 * @returns {Error}
 */
function appError(key, vars, code) {
  const err = new Error(t(key, vars));
  if (code) {
    err.code = code;
  }
  return err;
}

/**
 * @param {unknown} persistedLanguage
 * @param {unknown} osLocale
 */
function getI18nPayload(persistedLanguage, osLocale) {
  const persisted = isSupportedLanguage(persistedLanguage) ? persistedLanguage : null;
  const language = resolveLanguage(persisted, osLocale);
  return {
    language,
    persistedLanguage: persisted,
    available: getAvailableLanguages(),
    catalog: getCatalog(language),
    fallbackCatalog: catalogs.en,
  };
}

module.exports = {
  catalogs,
  setCurrentLanguage,
  getCurrentLanguage,
  getCatalog,
  t,
  appError,
  getI18nPayload,
  resolveLanguage,
  isSupportedLanguage,
  getAvailableLanguages,
  DEFAULT_LANGUAGE,
  SUPPORTED_LANGUAGES,
  chromiumLocaleTag,
  primaryLanguageTag,
};
