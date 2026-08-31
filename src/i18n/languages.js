'use strict';

const SUPPORTED_LANGUAGES = Object.freeze(['en', 'de', 'fr', 'es', 'it', 'pt']);

const LANGUAGE_NATIVE_NAMES = Object.freeze({
  en: 'English',
  de: 'Deutsch',
  fr: 'Français',
  es: 'Español',
  it: 'Italiano',
  pt: 'Português',
});

const DEFAULT_LANGUAGE = 'en';

/**
 * @param {unknown} code
 * @returns {code is string}
 */
function isSupportedLanguage(code) {
  return typeof code === 'string' && SUPPORTED_LANGUAGES.includes(code);
}

/**
 * @param {unknown} locale
 * @returns {string | null}
 */
function primaryLanguageTag(locale) {
  if (typeof locale !== 'string' || !locale.trim()) {
    return null;
  }
  const normalized = locale.trim().replace(/_/g, '-').toLowerCase();
  const primary = normalized.split('-')[0];
  return primary || null;
}

/**
 * @param {unknown} persisted
 * @param {unknown} osLocale
 * @returns {string}
 */
function resolveLanguage(persisted, osLocale) {
  if (isSupportedLanguage(persisted)) {
    return persisted;
  }
  if (typeof osLocale === 'string') {
    const normalized = osLocale.trim().replace(/_/g, '-').toLowerCase();
    if (isSupportedLanguage(normalized)) {
      return normalized;
    }
  }
  const primary = primaryLanguageTag(osLocale);
  if (primary && isSupportedLanguage(primary)) {
    return primary;
  }
  return DEFAULT_LANGUAGE;
}

function getAvailableLanguages() {
  return SUPPORTED_LANGUAGES.map((code) => ({
    code,
    nativeName: LANGUAGE_NATIVE_NAMES[code],
  }));
}

const LATIN_AMERICA_ES = new Set([
  'es-419',
  'es-mx',
  'es-ar',
  'es-co',
  'es-cl',
  'es-pe',
  'es-ve',
  'es-ec',
  'es-gt',
  'es-cu',
  'es-bo',
  'es-do',
  'es-hn',
  'es-py',
  'es-sv',
  'es-ni',
  'es-cr',
  'es-pa',
  'es-uy',
  'es-pr',
]);

/**
 * Chromium `--lang` / `.pak` tag for a Glaux language. Regional variants follow
 * the OS locale when it matches that language.
 * @param {unknown} glauxLanguage
 * @param {unknown} osLocale
 * @returns {string}
 */
function chromiumLocaleTag(glauxLanguage, osLocale) {
  const lang = isSupportedLanguage(glauxLanguage) ? glauxLanguage : DEFAULT_LANGUAGE;
  const os =
    typeof osLocale === 'string' ? osLocale.trim().replace(/_/g, '-').toLowerCase() : '';
  const osRegion = os.split('-').slice(0, 2).join('-');
  if (lang === 'en') {
    if (os.startsWith('en-gb') || os.startsWith('en-uk')) {
      return 'en-GB';
    }
    return 'en-US';
  }
  if (lang === 'pt') {
    if (os.startsWith('pt-pt')) {
      return 'pt-PT';
    }
    return 'pt-BR';
  }
  if (lang === 'es') {
    if (os.startsWith('es-419') || LATIN_AMERICA_ES.has(os) || LATIN_AMERICA_ES.has(osRegion)) {
      return 'es-419';
    }
    return 'es';
  }
  return lang;
}

/**
 * True if an Electron locale pack basename (e.g. `en-US`, `pt_BR`, `English`)
 * belongs to a Glaux UI language.
 * @param {unknown} basename
 * @returns {boolean}
 */
function isSupportedElectronLocaleFile(basename) {
  if (typeof basename !== 'string' || !basename.trim()) {
    return false;
  }
  if (basename.trim().toLowerCase() === 'english') {
    return isSupportedLanguage('en');
  }
  const primary = primaryLanguageTag(basename);
  return Boolean(primary && isSupportedLanguage(primary));
}

module.exports = {
  SUPPORTED_LANGUAGES,
  LANGUAGE_NATIVE_NAMES,
  DEFAULT_LANGUAGE,
  isSupportedLanguage,
  primaryLanguageTag,
  resolveLanguage,
  getAvailableLanguages,
  chromiumLocaleTag,
  isSupportedElectronLocaleFile,
};
