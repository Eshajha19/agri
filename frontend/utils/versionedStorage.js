const DEFAULT_VERSION = 1;

const safeParse = (value) => {
  if (typeof value !== "string") {
    return null;
  }

  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

const isVersionedEnvelope = (value) =>
  value && typeof value === "object" && !Array.isArray(value) && "v" in value;

const trimCollection = (items, maxItems) => {
  if (!Array.isArray(items)) {
    return [];
  }

  if (!Number.isFinite(maxItems) || maxItems <= 0) {
    return items.slice();
  }

  return items.slice(0, maxItems);
};

export const loadVersionedArray = (key, { version = DEFAULT_VERSION, fallback = [], maxItems } = {}) => {
  if (typeof window === "undefined") {
    return fallback.slice();
  }

  const parsed = safeParse(localStorage.getItem(key));
  const items = Array.isArray(parsed)
    ? parsed
    : isVersionedEnvelope(parsed) && Array.isArray(parsed.items)
      ? parsed.items
      : fallback;

  return trimCollection(items, maxItems);
};

export const saveVersionedArray = (key, items, { version = DEFAULT_VERSION, maxItems } = {}) => {
  if (typeof window === "undefined") {
    return false;
  }

  const payload = {
    v: version,
    items: trimCollection(items, maxItems),
    updatedAt: new Date().toISOString(),
  };

  try {
    localStorage.setItem(key, JSON.stringify(payload));
    return true;
  } catch (error) {
    const isQuotaError =
      error?.name === "QuotaExceededError" ||
      error?.code === 22 ||
      error?.code === 1014;

    if (isQuotaError && Number.isFinite(maxItems) && maxItems > 0) {
      try {
        const reducedPayload = {
          ...payload,
          items: trimCollection(items, Math.max(1, Math.floor(maxItems / 2))),
        };
        localStorage.setItem(key, JSON.stringify(reducedPayload));
        return true;
      } catch {
        // Fall through to false.
      }
    }

    return false;
  }
};

export const loadVersionedObject = (key, { version = DEFAULT_VERSION, fallback = {} } = {}) => {
  if (typeof window === "undefined") {
    return { ...fallback };
  }

  const parsed = safeParse(localStorage.getItem(key));
  if (isVersionedEnvelope(parsed) && parsed.data && typeof parsed.data === "object" && !Array.isArray(parsed.data)) {
    return parsed.data;
  }

  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return parsed;
  }

  return { ...fallback };
};

export const saveVersionedObject = (key, data, { version = DEFAULT_VERSION } = {}) => {
  if (typeof window === "undefined") {
    return false;
  }

  try {
    localStorage.setItem(
      key,
      JSON.stringify({
        v: version,
        data,
        updatedAt: new Date().toISOString(),
      })
    );
    return true;
  } catch {
    return false;
  }
};
