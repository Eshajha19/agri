/**
 * Google Translate Website Translator synchronization utilities.
 *
 * These functions manage the legacy `google.translate.TranslateElement` widget
 * that provides on-the-fly page translation for languages not covered by the
 * i18next localisation bundles.
 */

const GOOGLE_TRANSLATE_TIMEOUT = 15000;
const GOOGLE_TRANSLATE_SYNC_DELAY = 1200;
const MAX_RETRIES = 3;

let googleTranslateObserver = null;
let googleTranslateRetryTimeout = null;
let googleTranslateRefreshTimeout = null;
let lastGoogleTranslateRefreshAt = 0;
let lastAppliedLanguage = null;
let translateInitializationInProgress = false;

/**
 * Apply a language code to the Google Translate combo box.
 * Returns true on success, false when the widget is unavailable.
 */
export const applyGoogleTranslate = (langCode) => {
  try {
    const select = document.querySelector(".goog-te-combo");

    if (!select) {
      return false;
    }

    // Prevent redundant re-application
    if (select.value === langCode && lastAppliedLanguage === langCode) {
      return true;
    }

    select.value = langCode;

    select.dispatchEvent(
      new Event("change", { bubbles: true })
    );

    lastAppliedLanguage = langCode;

    return true;
  } catch (error) {
    console.error(
      "Google Translate apply error:",
      error
    );

    return false;
  }
};

/**
 * Remove the `.notranslate` CSS class from elements that do NOT carry an
 * explicit `translate="no"` attribute.  This lets Google Translate
 * localise generic UI copy while still protecting brand names and other
 * proper nouns that are marked with `translate="no"`.
 */
export const allowUserInterfaceTranslation = () => {
  if (typeof document === "undefined") return;
  try {
    document
      .querySelectorAll(".notranslate:not([translate='no'])")
      .forEach((element) => element.classList.remove("notranslate"));
  } catch (error) {
    console.error("Error allowing UI translation:", error);
  }
};

/**
 * Re-trigger Google Translate so that lazy-loaded content is picked up.
 */
export const refreshGoogleTranslation = (langCode) => {
  if (langCode === "en") return;
  if (typeof document === "undefined") return;

  const now = Date.now();
  if (now - lastGoogleTranslateRefreshAt < 1000) return;
  lastGoogleTranslateRefreshAt = now;

  clearTimeout(googleTranslateRefreshTimeout);
  googleTranslateRefreshTimeout = setTimeout(() => {
    allowUserInterfaceTranslation();
    // Reset the guard so Google Translate rescans text added by lazy routes.
    lastAppliedLanguage = null;
    applyGoogleTranslate(langCode);
  }, 250);
};

/**
 * Wait for the Google Translate widget combo box to appear in the DOM.
 */
export const waitForGoogleTranslateWidget = (
  timeoutMs = GOOGLE_TRANSLATE_TIMEOUT
) => {
  return new Promise((resolve, reject) => {
    if (typeof document === "undefined") {
      reject(new Error("Document is not available"));
      return;
    }

    const existingWidget = document.querySelector(
      ".goog-te-combo"
    );

    if (existingWidget) {
      resolve(existingWidget);
      return;
    }

    const timeoutId = setTimeout(() => {
      cleanup();
      reject(
        new Error(
          "Google Translate widget initialization timeout"
        )
      );
    }, timeoutMs);

    const cleanup = () => {
      clearTimeout(timeoutId);

      if (googleTranslateObserver) {
        googleTranslateObserver.disconnect();
        googleTranslateObserver = null;
      }
    };

    googleTranslateObserver = new MutationObserver(() => {
      const widget = document.querySelector(
        ".goog-te-combo"
      );

      if (widget) {
        cleanup();
        resolve(widget);
      }
    });

    googleTranslateObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
  });
};

/**
 * Robust translation application with retry-on-timeout fallback.
 */
export const applyGoogleTranslateRobust = async (
  langCode,
  options = {}
) => {
  const {
    retry = true,
    onReady,
    onError,
  } = options;

  // Prevent overlapping initialization calls
  if (translateInitializationInProgress) {
    return;
  }

  translateInitializationInProgress = true;

  try {
    await waitForGoogleTranslateWidget();

    const applied = applyGoogleTranslate(langCode);

    if (!applied) {
      throw new Error(
        "Failed to apply translation state"
      );
    }

    onReady?.();
  } catch (error) {
    console.warn(
      "Google Translate synchronization failed:",
      error.message
    );

    // Retry once after delayed script injection
    if (retry) {
      clearTimeout(googleTranslateRetryTimeout);

      googleTranslateRetryTimeout = setTimeout(() => {
        void applyGoogleTranslateRobust(langCode, {
          retry: false,
        });
      }, GOOGLE_TRANSLATE_SYNC_DELAY);
    }

    onError?.(error);
  } finally {
    translateInitializationInProgress = false;
  }
};

/**
 * Synchronize the Google Translate widget with the desired language.
 *
 * Accepts injectable `applyFn` and `robustFn` so the retry logic can be
 * unit-tested without a DOM or a live Google Translate widget.
 *
 * @param {string} langCode - Target language code (e.g. "hi", "bn").
 * @param {Function} [applyFn] - Fast-path function; returns true on success.
 * @param {Function} [robustFn] - Fallback function called with (langCode, {onReady, onError}).
 */
export const synchronizeTranslation = async (
  langCode,
  applyFn = applyGoogleTranslate,
  robustFn = applyGoogleTranslateRobust
) => {
  if (!langCode) return;

  let retryCount = 0;

  const attempt = async () => {
    if (!langCode) return;

    // Allow Google Translate to localise UI copy that was previously
    // marked with `.notranslate` (older route components).  Brand names
    // and proper nouns carry `translate="no"` and are therefore preserved.
    allowUserInterfaceTranslation();

    try {
      // Fast path – widget is ready, apply directly.
      if (await applyFn(langCode)) {
        refreshGoogleTranslation(langCode);
        console.log("Google Translate initialized successfully");
        return;
      }

      // Robust fallback – wait for the widget then apply.
      await robustFn(langCode, {
        onReady: () => {
          console.log("Google Translate synchronized successfully");
        },
        onError: () => {
          console.warn("Translation fallback active");
          if (retryCount < MAX_RETRIES) {
            retryCount++;
            attempt();
          }
        },
      });
    } catch (error) {
      console.error("Google Translate init failed:", error);
      if (retryCount < MAX_RETRIES) {
        retryCount++;
        attempt();
      }
    }
  };

  await attempt();
};

/**
 * Tear down all timers and observers created by this module.
 * Call from a component's useEffect cleanup.
 */
export const cleanupGoogleTranslate = () => {
  if (googleTranslateObserver) {
    googleTranslateObserver.disconnect();
    googleTranslateObserver = null;
  }

  if (googleTranslateRetryTimeout) {
    clearTimeout(googleTranslateRetryTimeout);
    googleTranslateRetryTimeout = null;
  }

  if (googleTranslateRefreshTimeout) {
    clearTimeout(googleTranslateRefreshTimeout);
    googleTranslateRefreshTimeout = null;
  }
};
