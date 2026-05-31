export const secureStorage = {
  get(key) {
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  },

  set(key, value) {
    try {
      sessionStorage.setItem(key, value);
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
  }
};