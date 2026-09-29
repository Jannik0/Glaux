// Welcome tour: spotlight each panel from left to right. On first launch a
// welcome prompt offers the tour. The Tour button can replay it.

const SUGGESTED_MODEL_ID = 'unsloth/gemma-4-E2B-it-GGUF';

const TOUR_STEPS = [
  { id: 'models', selector: '#models-panel', offerDownload: true },
  { id: 'workspaces', selector: '#workspace-controls' },
  { id: 'sessions', selector: '#sessions-panel' },
  { id: 'resources', selector: '#resources-panel' },
  { id: 'chat', selector: '.chat-panel' },
  { id: 'outputs', selector: '#outputs-panel' },
];

function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

const welcomeOverlayEl = document.getElementById('welcome-overlay');
const welcomeBoxEl = document.getElementById('welcome-box');
const welcomeDismissEl = document.getElementById('welcome-dismiss');
const welcomeTourEl = document.getElementById('welcome-tour');
const tourButtonEl = document.getElementById('tour-button');
const tourOverlayEl = document.getElementById('tour-overlay');
const tourBlockerEl = document.getElementById('tour-blocker');
const tourHighlightEl = document.getElementById('tour-highlight');
const tourPopoverEl = document.getElementById('tour-popover');
const tourTitleEl = document.getElementById('tour-title');
const tourBodyEl = document.getElementById('tour-body');
const tourSkipEl = document.getElementById('tour-skip');
const tourBackEl = document.getElementById('tour-back');
const tourNextEl = document.getElementById('tour-next');
const tourDownloadEl = document.getElementById('tour-download');
const tourLaterEl = document.getElementById('tour-later');

let tourActive = false;
let tourStepIndex = 0;
let tourHandoff = false;
/** @type {{ models: boolean, sessions: boolean, resources: boolean, outputs: boolean } | null} */
let collapsedBeforeTour = null;

function isTourVisible() {
  return Boolean(tourOverlayEl && !tourOverlayEl.classList.contains('hidden'));
}

function isWelcomeVisible() {
  return Boolean(welcomeOverlayEl && !welcomeOverlayEl.classList.contains('hidden'));
}

function showWelcome() {
  if (!welcomeOverlayEl) {
    return;
  }
  welcomeOverlayEl.classList.remove('hidden');
  if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur();
  }
  welcomeTourEl?.focus();
}

function hideWelcome() {
  welcomeOverlayEl?.classList.add('hidden');
}

/**
 * @param {HTMLElement | null} el
 * @param {boolean} hidden
 */
function setTourHidden(el, hidden) {
  if (!el) {
    return;
  }
  el.classList.toggle('hidden', hidden);
}

function sidePanelsAnyCollapsed(snapshot) {
  return Boolean(
    snapshot &&
      (snapshot.models || snapshot.sessions || snapshot.resources || snapshot.outputs),
  );
}

function expandPanelsForTour() {
  const panels = window.Glaux.SidePanels;
  if (!panels) {
    collapsedBeforeTour = null;
    return;
  }
  const snapshot = panels.snapshotCollapsed();
  if (!sidePanelsAnyCollapsed(snapshot)) {
    collapsedBeforeTour = null;
    return;
  }
  collapsedBeforeTour = snapshot;
  panels.expandAll();
}

function restorePanelsAfterTour() {
  const snapshot = collapsedBeforeTour;
  collapsedBeforeTour = null;
  if (!snapshot || !window.Glaux.SidePanels) {
    return;
  }
  window.Glaux.SidePanels.restoreCollapsed(snapshot);
}

function positionTour() {
  if (!isTourVisible() || !tourHighlightEl || !tourPopoverEl) {
    return;
  }
  const step = TOUR_STEPS[tourStepIndex];
  const target = step ? document.querySelector(step.selector) : null;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const rect = target.getBoundingClientRect();
  tourHighlightEl.style.left = `${rect.left}px`;
  tourHighlightEl.style.top = `${rect.top}px`;
  tourHighlightEl.style.width = `${rect.width}px`;
  tourHighlightEl.style.height = `${rect.height}px`;

  const margin = 12;
  const gap = 16;
  const popRect = tourPopoverEl.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let left;
  let top;
  if (step.id === 'workspaces') {
    left = rect.left + (rect.width - popRect.width) / 2;
    top = rect.bottom + gap;
  } else if (step.id === 'chat') {
    left = rect.left + (rect.width - popRect.width) / 2;
    top = rect.top + 16;
  } else if (rect.left + rect.width / 2 < vw / 2) {
    left = rect.right + gap;
    top = rect.top;
  } else {
    left = rect.left - gap - popRect.width;
    top = rect.top;
  }
  left = Math.max(margin, Math.min(left, vw - popRect.width - margin));
  top = Math.max(margin, Math.min(top, vh - popRect.height - margin));
  tourPopoverEl.style.left = `${Math.round(left)}px`;
  tourPopoverEl.style.top = `${Math.round(top)}px`;
}

function focusTourPrimary() {
  const step = TOUR_STEPS[tourStepIndex];
  const primary = step && step.offerDownload ? tourDownloadEl : tourNextEl;
  if (primary instanceof HTMLElement) {
    primary.focus();
  }
}

/**
 * @param {number} index
 */
function showTourStep(index) {
  const step = TOUR_STEPS[index];
  if (!step || !tourOverlayEl || !tourTitleEl || !tourBodyEl) {
    return;
  }
  tourStepIndex = index;
  tourTitleEl.textContent = t(`tour.steps.${step.id}.title`);
  tourBodyEl.textContent = t(`tour.steps.${step.id}.body`);
  const isLast = index === TOUR_STEPS.length - 1;
  setTourHidden(tourBackEl, index === 0);
  setTourHidden(tourDownloadEl, !step.offerDownload);
  setTourHidden(tourLaterEl, !step.offerDownload);
  setTourHidden(tourNextEl, Boolean(step.offerDownload));
  if (tourNextEl) {
    tourNextEl.textContent = isLast ? t('tour.done') : t('tour.next');
  }
  tourOverlayEl.classList.remove('hidden');
  positionTour();
  focusTourPrimary();
}

function finishTour() {
  if (tourHandoff) {
    return;
  }
  tourActive = false;
  if (tourOverlayEl) {
    tourOverlayEl.classList.add('hidden');
  }
  restorePanelsAfterTour();
}

function startTour() {
  if (tourHandoff) {
    return;
  }
  if (!tourActive) {
    expandPanelsForTour();
    tourActive = true;
  }
  showTourStep(0);
}

function goTourBack() {
  if (tourStepIndex > 0) {
    showTourStep(tourStepIndex - 1);
  }
}

function goTourNext() {
  if (tourStepIndex >= TOUR_STEPS.length - 1) {
    finishTour();
    return;
  }
  showTourStep(tourStepIndex + 1);
}

async function chooseTourDownload() {
  if (tourHandoff || !tourActive) {
    return;
  }
  tourHandoff = true;
  if (tourOverlayEl) {
    tourOverlayEl.classList.add('hidden');
  }
  try {
    const begin = window.Glaux.Models && window.Glaux.Models.beginDownload;
    if (typeof begin === 'function') {
      await begin(SUGGESTED_MODEL_ID);
    }
  } finally {
    tourHandoff = false;
    if (tourActive) {
      showTourStep(1);
    }
  }
}

/**
 * @param {KeyboardEvent} event
 * @param {HTMLElement} popover
 */
function trapTourTab(event, popover) {
  const buttons = [...popover.querySelectorAll('button')].filter((btn) => {
    return btn instanceof HTMLButtonElement && !btn.classList.contains('hidden') && !btn.disabled;
  });
  if (!buttons.length) {
    return;
  }
  const first = buttons[0];
  const last = buttons[buttons.length - 1];
  const active = document.activeElement;
  if (event.shiftKey && (active === first || !popover.contains(active))) {
    event.preventDefault();
    last.focus();
    return;
  }
  if (!event.shiftKey && (active === last || !popover.contains(active))) {
    event.preventDefault();
    first.focus();
  }
}

function onTourKeyDown(event) {
  if (isWelcomeVisible() && welcomeBoxEl) {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopImmediatePropagation();
      hideWelcome();
      return;
    }
    if (event.key === 'Tab') {
      trapTourTab(event, welcomeBoxEl);
      return;
    }
    if (event.target instanceof Node && welcomeBoxEl.contains(event.target)) {
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    return;
  }
  if (!isTourVisible() || !tourPopoverEl) {
    return;
  }
  if (event.key === 'Escape') {
    event.preventDefault();
    event.stopImmediatePropagation();
    finishTour();
    return;
  }
  if (event.key === 'Tab') {
    trapTourTab(event, tourPopoverEl);
    return;
  }
  if (event.target instanceof Node && tourPopoverEl.contains(event.target)) {
    return;
  }
  event.preventDefault();
  event.stopImmediatePropagation();
}

async function startTourIfFirstLaunch() {
  if (!(window.api && typeof window.api.isFirstLaunch === 'function')) {
    return;
  }
  let first = false;
  try {
    first = await window.api.isFirstLaunch();
  } catch {
    return;
  }
  if (!first) {
    return;
  }
  if (typeof window.api.consumeFirstLaunch === 'function') {
    try {
      await window.api.consumeFirstLaunch();
    } catch {
      /* Still show the tour for this launch. */
    }
  }
  if (tourActive || tourHandoff || isWelcomeVisible()) {
    return;
  }
  showWelcome();
}

welcomeDismissEl?.addEventListener('click', () => {
  hideWelcome();
});
welcomeTourEl?.addEventListener('click', () => {
  hideWelcome();
  startTour();
});
welcomeOverlayEl?.addEventListener('mousedown', (event) => {
  if (event.target === welcomeOverlayEl) {
    event.preventDefault();
  }
});
welcomeOverlayEl?.addEventListener('click', (event) => {
  event.stopPropagation();
});
tourButtonEl?.addEventListener('click', () => {
  startTour();
});
tourSkipEl?.addEventListener('click', () => {
  finishTour();
});
tourBackEl?.addEventListener('click', () => {
  goTourBack();
});
tourNextEl?.addEventListener('click', () => {
  goTourNext();
});
tourDownloadEl?.addEventListener('click', () => {
  void chooseTourDownload();
});
tourLaterEl?.addEventListener('click', () => {
  showTourStep(1);
});
tourBlockerEl?.addEventListener('mousedown', (event) => {
  event.preventDefault();
  event.stopPropagation();
});
tourOverlayEl?.addEventListener('click', (event) => {
  event.stopPropagation();
});
document.addEventListener('keydown', onTourKeyDown, true);
window.addEventListener('resize', () => {
  if (isTourVisible()) {
    positionTour();
  }
});

window.Glaux.Tour = {
  start: startTour,
  startIfFirstLaunch: startTourIfFirstLaunch,
};
