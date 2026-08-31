// Renderer i18n boot: load the catalog synchronously via preload, apply
// data-i18n* attributes, and expose window.Glaux.i18n.t for JS strings.
(function () {
  window.Glaux = window.Glaux || {};

  const translateApi =
    window.Glaux._translate || {
      t: (_catalog, key) => key,
    };

  const emptyPayload = {
    language: 'en',
    persistedLanguage: null,
    available: [],
    catalog: {},
    fallbackCatalog: {},
  };

  let payload = emptyPayload;
  try {
    if (window.api && typeof window.api.getI18n === 'function') {
      const loaded = window.api.getI18n();
      if (loaded && typeof loaded === 'object') {
        payload = loaded;
      }
    }
  } catch {
    payload = emptyPayload;
  }

  const catalog = payload.catalog && typeof payload.catalog === 'object' ? payload.catalog : {};
  const fallbackCatalog =
    payload.fallbackCatalog && typeof payload.fallbackCatalog === 'object'
      ? payload.fallbackCatalog
      : catalog;
  const language = typeof payload.language === 'string' && payload.language ? payload.language : 'en';
  const available = Array.isArray(payload.available) ? payload.available : [];

  function t(key, vars) {
    return translateApi.t(catalog, key, vars, fallbackCatalog);
  }

  function applyAttribute(el, attrName, key) {
    if (!key) {
      return;
    }
    el.setAttribute(attrName, t(key));
  }

  function applyDomTranslations(root) {
    const scope = root || document;
    const nodes = scope.querySelectorAll
      ? scope.querySelectorAll('[data-i18n], [data-i18n-title], [data-i18n-aria], [data-i18n-placeholder]')
      : [];
    for (const el of nodes) {
      if (!(el instanceof Element)) {
        continue;
      }
      const textKey = el.getAttribute('data-i18n');
      if (textKey) {
        el.textContent = t(textKey);
      }
      applyAttribute(el, 'title', el.getAttribute('data-i18n-title'));
      applyAttribute(el, 'aria-label', el.getAttribute('data-i18n-aria'));
      applyAttribute(el, 'placeholder', el.getAttribute('data-i18n-placeholder'));
    }
  }

  document.documentElement.lang = language;
  applyDomTranslations(document);
  document.documentElement.setAttribute('data-i18n-ready', '');

  window.Glaux.i18n = {
    t,
    language,
    persistedLanguage: payload.persistedLanguage || null,
    available,
    applyDomTranslations,
  };
})();
