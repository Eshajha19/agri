import { create } from 'zustand';
import { LANGUAGE_OPTIONS } from '../lib/languageOptions';

const getInitialTheme = () => {
  try {
    return localStorage.getItem('agri:theme') || 'light';
  } catch {
    return 'light';
  }
};

const getInitialAccessibilityMode = () => {
  try {
    return localStorage.getItem('agri:accessibilityMode') === 'sunlight';
  } catch {
    return false;
  }
};

export const useUiStore = create((set) => ({
  // Theme state
  theme: getInitialTheme(),
  setTheme: (theme) => {
    localStorage.setItem('agri:theme', theme);
    set({ theme });
  },

  // Accessibility / Sunlight mode state
  isAccessibilityMode: getInitialAccessibilityMode(),
  setAccessibilityMode: (enabled) => {
    document.documentElement.classList.toggle('sunlight', enabled);
    localStorage.setItem('agri:accessibilityMode', enabled ? 'sunlight' : 'light');
    set({ isAccessibilityMode: enabled });
  },

  // Language options (for reference - actual language change is handled by i18n)
  languageOptions: LANGUAGE_OPTIONS,

  // Navigation sidebar
  isNavOpen: false,
  toggleNav: () => set((state) => ({ isNavOpen: !state.isNavOpen })),
  setNavOpen: (isOpen) => set({ isNavOpen: isOpen }),

  inputName: '',
  setInputName: (name) => set({ inputName: name }),

  // Global API loading state
  apiPendingRequests: 0,
  isApiLoading: false,
  incrementApiPendingRequests: () =>
    set((state) => {
      const nextPending = state.apiPendingRequests + 1;
      return {
        apiPendingRequests: nextPending,
        isApiLoading: nextPending > 0,
      };
    }),
  decrementApiPendingRequests: () =>
    set((state) => {
      const nextPending = Math.max(0, state.apiPendingRequests - 1);
      return {
        apiPendingRequests: nextPending,
        isApiLoading: nextPending > 0,
      };
    }),
}));
