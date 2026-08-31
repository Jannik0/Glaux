// Shared helper for panel status messages that auto-clear after a delay.
// Exposed as window.Glaux.Status so panels/*.js can share one clearing timer map.
(function () {
  window.Glaux = window.Glaux || {};

  const STATUS_MESSAGE_CLEAR_MS = 5000;
  /** @type {Map<string, ReturnType<typeof setTimeout>>} */
  const statusMessageClearTimers = new Map();

  /**
   * @param {HTMLElement | null} element
   * @param {string} key
   * @param {string} message
   * @param {boolean} isError
   * @param {string} errorClass
   */
  function setTimedPanelStatus(element, key, message, isError, errorClass) {
    if (!element) {
      return;
    }

    const pending = statusMessageClearTimers.get(key);
    if (pending) {
      clearTimeout(pending);
      statusMessageClearTimers.delete(key);
    }

    const text = message || '';
    element.textContent = text;
    element.classList.toggle(errorClass, Boolean(isError));

    if (text) {
      const timer = setTimeout(() => {
        statusMessageClearTimers.delete(key);
        element.textContent = '';
        element.classList.remove(errorClass);
      }, STATUS_MESSAGE_CLEAR_MS);
      statusMessageClearTimers.set(key, timer);
    }
  }

  window.Glaux.Status = { setTimedPanelStatus };
})();
