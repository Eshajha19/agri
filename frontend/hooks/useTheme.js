import { useCallback, useEffect } from 'react';

import { useUiStore } from '../stores/uiStore';

export const useTheme = () => {
  const { theme, setTheme } = useUiStore();

  useEffect(() => {
    document.documentElement.classList.toggle('theme-dark', theme === 'dark');
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  }, [theme, setTheme]);

  return {
    theme,
    setTheme,
    toggleTheme,
    isDarkTheme: theme === 'dark',
  };
};