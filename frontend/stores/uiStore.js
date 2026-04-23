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
  { value: 'as', label: '🇮🇳 অসমীয়া' },
];

const getInitialLanguage = () => {
  try {
    const stored = localStorage.getItem('preferredLanguage');
    return LANGUAGE_OPTIONS.some((l) => l.value === stored) ? stored : 'en';
  } catch {
    return 'en';
  }
};

const getInitialTheme = () => {
  try {
    return localStorage.getItem('theme') || 'light';
  } catch {
    return 'light';
  }
};

export const useUiStore = create((set) => ({
  // Theme state
  theme: getInitialTheme(),
  setTheme: (theme) => {
    document.documentElement.classList.toggle('theme-dark', theme === 'dark');
    localStorage.setItem('theme', theme);
    set({ theme });
  },

  // Language state
  preferredLang: getInitialLanguage(),
  setPreferredLang: (lang) => {
    localStorage.setItem('preferredLanguage', lang);
    set({ preferredLang: lang });
  },

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
}));
