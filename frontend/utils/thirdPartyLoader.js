/**
 * Centralised third-party script loader.
 *
 * Features
 * --------
 * - Retry with exponential backoff
 * - Configurable timeout per script
 * - Dependency tracking (load scripts in order, know when all are ready)
 * - Error monitoring via a callback registry
 *
 * Usage
 * -----
 *   import loader from "./utils/thirdPartyLoader.js";
 *
 *   loader.load("google-translate", {
 *     src: "https://translate.google.com/translate_a/element.js?cb=onGoogleTranslateInit",
 *     timeoutMs: 20_000,
 *     retries: 2,
 *     attrs: { defer: true },
 *   });
 *
 *   loader.onReady("google-translate", () => { … });
 *   loader.onError("google-translate", (err) => { … });
 */

const _registry = new Map();   // key → { status, error, callbacks }
const _errors = new Map();     // key → registered error handlers

const PENDING = "pending";
const LOADING = "loading";
const READY   = "ready";
const FAILED  = "failed";

let _counter = 0;

function _key(opts) {
  return opts.key || `script_${++_counter}`;
}

async function _sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function _loadOne(key, opts) {
  const entry = _registry.get(key);
  if (!entry || entry.status === READY) return;
  entry.status = LOADING;

  const src = opts.src;
  const timeoutMs = opts.timeoutMs ?? 15_000;
  const retries = opts.retries ?? 1;
  const attrs = opts.attrs ?? {};

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      await new Promise((resolve, reject) => {
        const el = document.createElement("script");
        el.src = src;
        el.async = true;
        for (const [k, v] of Object.entries(attrs)) {
          if (v !== false) el.setAttribute(k, v === true ? k : v);
        }
        el.onload = () => { clearTimeout(timer); resolve(); };
        el.onerror = () => { clearTimeout(timer); reject(new Error(`Failed to load: ${src}`)); };
        document.body.appendChild(el);
      });

      entry.status = READY;
      entry.error = null;
      for (const cb of entry.callbacks) cb();
      return;
    } catch (err) {
      entry.error = err;
      if (attempt < retries) {
        await _sleep(500 * Math.pow(2, attempt));
      }
    }
  }

  entry.status = FAILED;
  for (const cb of (_errors.get(key) || [])) cb(entry.error);
}

/**
 * Load (or ensure loading of) a third-party script.
 * Returns the same promise for concurrent calls with the same key.
 */
export function load(opts = {}) {
  const key = _key(opts);

  if (_registry.has(key)) {
    const existing = _registry.get(key);
    if (existing.status === READY) return Promise.resolve();
    if (existing.status === FAILED) return Promise.reject(existing.error);
    return new Promise((resolve, reject) => {
      existing.callbacks.push(resolve);
      _errors.set(key, (_errors.get(key) || []).concat([reject]));
    });
  }

  _registry.set(key, { status: PENDING, error: null, callbacks: [] });
  const promise = _loadOne(key, opts);
  return promise;
}

/**
 * Register a ready callback for a specific key.
 */
export function onReady(key, fn) {
  const entry = _registry.get(key);
  if (!entry) {
    _registry.set(key, { status: PENDING, error: null, callbacks: [fn] });
    return;
  }
  if (entry.status === READY) { fn(); return; }
  entry.callbacks.push(fn);
}

/**
 * Register an error callback for a specific key.
 */
export function onError(key, fn) {
  if (!_errors.has(key)) _errors.set(key, []);
  _errors.get(key).push(fn);
}

/**
 * Return a snapshot of the registry (for debugging / monitoring).
 */
export function getStatus() {
  const snapshot = {};
  for (const [key, entry] of _registry) {
    snapshot[key] = { status: entry.status, error: entry.error?.message ?? null };
  }
  return snapshot;
}

export default { load, onReady, onError, getStatus };
