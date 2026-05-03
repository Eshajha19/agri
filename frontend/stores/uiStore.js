import { create } from "zustand";

/* =========================
   CONSTANTS
========================= */

const LANGUAGE_OPTIONS = [
  { value: "en", label: "🌍 English" },
  { value: "hi", label: "🇮🇳 हिंदी" },
  { value: "mr", label: "🇮🇳 मराठी" },
  { value: "bn", label: "🇮🇳 বাংলা" },
  { value: "ta", label: "🇮🇳 தமிழ்" },
  { value: "te", label: "🇮🇳 తెలుగు" },
  { value: "gu", label: "🇮🇳 ગુજરાતી" },
  { value: "pa", label: "🇮🇳 ਪੰਜਾਬੀ" },
  { value: "kn", label: "🇮🇳 ಕನ್ನಡ" },
  { value: "ml", label: "🇮🇳 മലയാളം" },
  { value: "or", label: "🇮🇳 ଓଡ଼ିଆ" },
  { value: "as", label: "🇮🇳 অসমীয়া" },
];

/* =========================
   SAFE STORAGE HELPERS
========================= */

const safeGet = (key, fallback) => {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
};

const safeSet = (key, value) => {
  try {
    localStorage.setItem(key, value);
  } catch {}
};

/* =========================
   INITIAL VALUES
========================= */

const getInitialTheme = () => safeGet("theme", "light");

const getInitialAccessibilityMode = () =>
  safeGet("accessibilityMode", "off") === "on";

const getInitialFarmerName = () => safeGet("farmerName", "");

/* =========================
   STORE
========================= */

export const useUiStore = create((set, get) => ({
  /* =====================
     THEME SYSTEM
  ===================== */

  theme: getInitialTheme(),

  setTheme: (theme) => {
    safeSet("theme", theme);
    set({ theme });
  },

  /* =====================
     ACCESSIBILITY MODE
     (Eye comfort / sunlight mode)
  ===================== */

  isAccessibilityMode: getInitialAccessibilityMode(),

  setAccessibilityMode: (enabled) => {
    safeSet("accessibilityMode", enabled ? "on" : "off");

    set({ isAccessibilityMode: enabled });
  },

  /* =====================
     LANGUAGE
  ===================== */

  languageOptions: LANGUAGE_OPTIONS,

  /* =====================
     NAVIGATION
  ===================== */

  isNavOpen: false,

  toggleNav: () =>
    set((state) => ({ isNavOpen: !state.isNavOpen })),

  setNavOpen: (isOpen) => set({ isNavOpen: isOpen }),

  /* =====================
     FARMER PROFILE
  ===================== */

  farmerName: getInitialFarmerName(),

  setFarmerName: (name) => {
    safeSet("farmerName", name);
    set({ farmerName: name });
  },

  inputName: "",

  setInputName: (name) => set({ inputName: name }),

  /* =====================
     API LOADING STATE
  ===================== */

  apiPendingRequests: 0,

  isApiLoading: false,

  incrementApiPendingRequests: () => {
    const next = get().apiPendingRequests + 1;

    set({
      apiPendingRequests: next,
      isApiLoading: true,
    });
  },

  decrementApiPendingRequests: () => {
    const next = Math.max(0, get().apiPendingRequests - 1);

    set({
      apiPendingRequests: next,
      isApiLoading: next > 0,
    });
  },
}));