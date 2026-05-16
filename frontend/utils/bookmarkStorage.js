/**
 * Bookmark storage helper.
 * Uses localStorage to persist bookmarked crops and articles.
 */
const STORAGE_KEYS = {
  crops: "agri:bookmarkedCrops",
  articles: "agri:bookmarkedArticles",
};

const safeParse = (value) => {
  try {
    return JSON.parse(value);
  } catch (err) {
    return null;
  }
};

export const getBookmarks = (type) => {
  const key = STORAGE_KEYS[type];
  if (!key) return [];
  const stored = typeof window !== "undefined" ? localStorage.getItem(key) : null;
  return safeParse(stored) || [];
};

export const setBookmarks = (type, items) => {
  const key = STORAGE_KEYS[type];
  if (!key) return;
  localStorage.setItem(key, JSON.stringify(items || []));
};

export const isBookmarked = (type, id) => {
  const items = getBookmarks(type);
  return items.some((item) => item.id === id);
};

export const toggleBookmark = (type, item) => {
  const items = getBookmarks(type);
  const exists = items.some((entry) => entry.id === item.id);
  const updated = exists ? items.filter((entry) => entry.id !== item.id) : [...items, item];
  setBookmarks(type, updated);
  return updated;
};

export const getAllBookmarks = () => ({
  crops: getBookmarks("crops"),
  articles: getBookmarks("articles"),
});
