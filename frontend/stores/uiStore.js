import { create } from 'zustand';

const LANGUAGE_OPTIONS = [
  { value: 'en', label: '🌍 English' },
  { value: 'hi', label: '🇮🇳 हिंदी' },
  { value: 'mr', label: '🇮🇳 मराठी' },
  { value: 'bn', label: '🇮🇳 বাংলা' },
  { value: 'ta', label: '🇮🇳 தமிழ்' },
  { value: 'te', label: '🇮🇳 తెలుగు' },
  { value: 'gu', label: '🇮🇳 ગુજરાતી' },
  { value: 'pa', label: '🇮🇳 ਪੰਜਾਬੀ' },
  { value: 'kn', label: '🇮🇳 ಕನ್ನಡ' },
  { value: 'ml', label: '🇮🇳 മലയാളം' },
  { value: 'or', label: '🇮🇳 ଓଡ଼ିଆ' },
  { value: 'as', label: '🇮🇳 অসमीय' },
];

const getInitialTheme = () => {
  try {
    return localStorage.getItem('theme') || 'light';
  } catch {
    return 'light';
  }
};

const getInitialAccessibilityMode = () => {
  try {
    return localStorage.getItem('accessibilityMode') === 'sunlight';
  } catch {
    return false;
  }
};

export const useUiStore = create((set) => ({
  // Theme state
  theme: getInitialTheme(),
  setTheme: (theme) => {
    localStorage.setItem('theme', theme);
    set({ theme });
  },

  // Accessibility / Sunlight mode state
  isAccessibilityMode: getInitialAccessibilityMode(),
  setAccessibilityMode: (enabled) => {
    document.documentElement.classList.toggle('sunlight', enabled);
    localStorage.setItem('accessibilityMode', enabled ? 'sunlight' : 'light');
    set({ isAccessibilityMode: enabled });
  },

  // Language options (for reference - actual language change is handled by i18n)
  languageOptions: LANGUAGE_OPTIONS,

  // Navigation sidebar
  isNavOpen: false,
  toggleNav: () => set((state) => ({ isNavOpen: !state.isNavOpen })),
  setNavOpen: (isOpen) => set({ isNavOpen: isOpen }),

  // Farmer profile
  farmerName: localStorage.getItem('farmerName') || '',
  setFarmerName: (name) => {
    localStorage.setItem('farmerName', name);
    set({ farmerName: name });
  },
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
