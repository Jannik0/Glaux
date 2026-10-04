(() => {
  const THEME_KEY = "glaux-theme";
  const LANG_KEY = "glaux-lang";
  const SUPPORTED_LANGS = ["en", "de", "fr", "es", "it", "pt"];
  const root = document.documentElement;
  const nav = document.getElementById("site-nav");
  const navToggle = document.querySelector(".nav-toggle");
  const languageSelect = document.getElementById("language-select");
  const lightbox = document.getElementById("lightbox");
  const RELEASE_VERSION = "1.2.0";
  const RELEASE_BASE = `https://github.com/Jannik0/Glaux/releases/download/v${RELEASE_VERSION}`;
  const BUILDS = {
    windows: `${RELEASE_BASE}/Glaux-Setup-${RELEASE_VERSION}.exe`,
    deb: `${RELEASE_BASE}/Glaux_${RELEASE_VERSION}_amd64.deb`,
    rpm: `${RELEASE_BASE}/Glaux-${RELEASE_VERSION}.x86_64.rpm`,
  };
  const catalogs = Object.create(null);
  let catalog = null;

  const platform = navigator.userAgentData?.platform || navigator.platform || "";
  const ua = navigator.userAgent || "";
  const isWindows = /Win/i.test(platform) || /Windows/i.test(ua);
  const isMac = /Mac/i.test(platform) || /Mac OS/i.test(ua);
  const isLinux = /Linux/i.test(platform) || /Linux/i.test(ua);

  function osTheme() {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  function storedTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
    return osTheme();
  }

  function applyTheme(mode, persist) {
    root.setAttribute("data-theme", mode);
    if (persist) localStorage.setItem(THEME_KEY, mode);
    document.querySelectorAll("[data-theme-value]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.themeValue === mode);
    });
  }

  applyTheme(storedTheme(), false);

  const themeQuery = new URLSearchParams(window.location.search).get("theme");
  if (themeQuery === "light" || themeQuery === "dark") {
    applyTheme(themeQuery, true);
  }

  document.querySelectorAll("[data-theme-value]").forEach((button) => {
    button.addEventListener("click", () => applyTheme(button.dataset.themeValue, true));
  });

  function osLanguage() {
    const candidates = navigator.languages?.length
      ? navigator.languages
      : [navigator.language];
    for (const entry of candidates) {
      const tag = String(entry || "").replace(/_/g, "-").toLowerCase();
      if (SUPPORTED_LANGS.includes(tag)) return tag;
      const primary = tag.split("-")[0];
      if (SUPPORTED_LANGS.includes(primary)) return primary;
    }
    return "en";
  }

  function storedLanguage() {
    const stored = localStorage.getItem(LANG_KEY);
    if (SUPPORTED_LANGS.includes(stored)) return stored;
    return osLanguage();
  }

  function lookup(path) {
    if (!catalog || !path) return "";
    return path.split(".").reduce((value, key) => {
      if (value && typeof value === "object" && key in value) return value[key];
      return "";
    }, catalog);
  }

  function applyDownloadCopy() {
    const windows = document.getElementById("download-windows");
    const deb = document.getElementById("download-deb");
    const rpm = document.getElementById("download-rpm");
    if (windows) windows.href = BUILDS.windows;
    if (deb) deb.href = BUILDS.deb;
    if (rpm) rpm.href = BUILDS.rpm;

    [windows, deb, rpm].filter(Boolean).forEach((button) => {
      button.classList.remove("btn-primary");
      button.classList.add("btn-ghost");
    });

    const matched = isLinux ? deb : isMac ? null : windows;
    if (matched) {
      matched.classList.add("btn-primary");
      matched.classList.remove("btn-ghost");
    }

    if (!isLinux) return;
    const list = document.querySelector(".download-builds");
    const debItem = deb?.closest("li");
    const rpmItem = rpm?.closest("li");
    if (!list || !debItem || !rpmItem) return;
    list.prepend(rpmItem);
    list.prepend(debItem);
  }

  function applyI18n() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const value = lookup(el.dataset.i18n);
      if (value) el.textContent = value;
    });
    document.querySelectorAll("[data-i18n-html]").forEach((el) => {
      const value = lookup(el.dataset.i18nHtml);
      if (value) el.innerHTML = value;
    });
    document.querySelectorAll("[data-i18n-aria]").forEach((el) => {
      const value = lookup(el.dataset.i18nAria);
      if (value) el.setAttribute("aria-label", value);
    });
    document.querySelectorAll("[data-i18n-alt]").forEach((el) => {
      const value = lookup(el.dataset.i18nAlt);
      if (value) el.setAttribute("alt", value);
    });
    const title = lookup("meta.title");
    if (title) document.title = title;
    const description = lookup("meta.description");
    if (description) {
      const meta = document.querySelector('meta[name="description"]');
      const ogTitle = document.querySelector('meta[property="og:title"]');
      const ogDescription = document.querySelector('meta[property="og:description"]');
      if (meta) meta.setAttribute("content", description);
      if (ogTitle && title) ogTitle.setAttribute("content", title);
      if (ogDescription) ogDescription.setAttribute("content", description);
    }
    applyDownloadCopy();
  }

  async function loadCatalog(lang) {
    if (catalogs[lang]) return catalogs[lang];
    const response = await fetch(`i18n/${lang}.json`);
    if (!response.ok) throw new Error("catalog");
    catalogs[lang] = await response.json();
    return catalogs[lang];
  }

  async function setLanguage(lang, persist) {
    const next = SUPPORTED_LANGS.includes(lang) ? lang : "en";
    try {
      catalog = await loadCatalog(next);
    } catch {
      if (next !== "en") {
        try {
          catalog = await loadCatalog("en");
        } catch {
          return;
        }
      } else {
        return;
      }
    }
    root.lang = catalogs[next] ? next : "en";
    if (persist && catalogs[next]) localStorage.setItem(LANG_KEY, next);
    if (languageSelect) languageSelect.value = catalogs[next] ? next : "en";
    applyI18n();
  }

  const initialLang = storedLanguage();
  if (languageSelect) languageSelect.value = initialLang;
  setLanguage(initialLang, false);

  languageSelect?.addEventListener("pointerdown", () => {
    const { scrollX, scrollY } = window;
    const restore = () => {
      const html = document.documentElement;
      const previous = html.style.scrollBehavior;
      html.style.scrollBehavior = "auto";
      window.scrollTo(scrollX, scrollY);
      html.style.scrollBehavior = previous;
    };
    languageSelect.addEventListener("focus", restore, { once: true });
    requestAnimationFrame(restore);
  });

  languageSelect?.addEventListener("change", () => {
    setLanguage(languageSelect.value, true);
  });

  if (navToggle && nav) {
    navToggle.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        nav.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduceMotion && "IntersectionObserver" in window) {
    const reveal = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            reveal.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    document.querySelectorAll("[data-reveal]").forEach((el) => reveal.observe(el));
  } else {
    document.querySelectorAll("[data-reveal]").forEach((el) => el.classList.add("is-visible"));
  }

  const sectionIds = ["models", "features", "how-to", "download"];
  const sections = sectionIds
    .map((id) => document.getElementById(id))
    .filter(Boolean);
  const navLinks = [...document.querySelectorAll('.site-nav a[href^="#"]')].filter((link) =>
    sectionIds.includes(link.getAttribute("href").slice(1))
  );

  function setActiveNav(id) {
    navLinks.forEach((link) => {
      link.classList.toggle("is-active", link.getAttribute("href") === `#${id}`);
    });
  }

  if ("IntersectionObserver" in window) {
    const spy = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActiveNav(visible.target.id);
      },
      { rootMargin: "-35% 0px -50% 0px", threshold: [0.15, 0.35, 0.6] }
    );
    sections.forEach((section) => spy.observe(section));
  }

  const tabs = [...document.querySelectorAll(".use-tabs [role='tab']")];
  const panels = [...document.querySelectorAll(".use-panel")];

  function activateUse(id) {
    tabs.forEach((tab) => {
      const selected = tab.dataset.use === id;
      tab.setAttribute("aria-selected", String(selected));
    });
    panels.forEach((panel) => {
      const match = panel.id === `panel-${id}`;
      panel.classList.toggle("is-active", match);
      panel.hidden = !match;
    });
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activateUse(tab.dataset.use));
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        event.preventDefault();
        tabs[(index + 1) % tabs.length].focus();
      } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault();
        tabs[(index - 1 + tabs.length) % tabs.length].focus();
      }
    });
  });

  document.querySelectorAll(".step").forEach((step) => {
    const trigger = step.querySelector(".step-trigger");
    trigger.addEventListener("click", () => {
      const opening = !step.classList.contains("is-open");
      document.querySelectorAll(".step").forEach((other) => {
        other.classList.toggle("is-open", other === step && opening);
        other.querySelector(".step-trigger").setAttribute("aria-expanded", String(other === step && opening));
      });
    });
  });

  document.querySelector("[data-lightbox-open]")?.addEventListener("click", () => {
    if (typeof lightbox.showModal === "function") lightbox.showModal();
  });

  lightbox?.addEventListener("click", (event) => {
    if (event.target === lightbox) lightbox.close();
  });
})();
