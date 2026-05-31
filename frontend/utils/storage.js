export const secureStorage = {
  get(key) {
    try {
      const value = sessionStorage.getItem(key);

      if (value === null) {
        return null;
      }

      try {
        return JSON.parse(value);
      } catch {
        return value;
      }
    } catch {
      return null;
    }
  },

  set(key, value) {
    try {
      const serialized =
        typeof value === "string"
          ? value
          : JSON.stringify(value);

      sessionStorage.setItem(key, serialized);
    } catch (err) {
      console.error("Storage error:", err);
    }
  },

  remove(key) {
    try {
      sessionStorage.removeItem(key);
    } catch {}
  },

  clear() {
    try {
      sessionStorage.clear();
    } catch {}
  },
};