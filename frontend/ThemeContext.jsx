import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';

const ThemeContext = createContext();

/**
 * ThemeProvider manages the application's visual theme (light/dark/night).
 * It centralizes theme state and ensures synchronization with the DOM and localStorage,
 * following React's state-driven lifecycle to avoid inconsistencies.
 */
export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    try {
      const storedTheme = localStorage.getItem('agri:theme');
      return storedTheme === 'dark' || storedTheme === 'night' ? storedTheme : 'light';
    } catch {
      return 'light';
    }
  });

  // Centralized side-effect to sync React state with the DOM
  useEffect(() => {
    const root = document.documentElement;
    const isDarkTheme = theme !== 'light';

    root.classList.toggle('theme-dark', isDarkTheme);
    root.classList.toggle('theme-night', theme === 'night');
    
    // Also set data attribute for future-proofing and better selector performance
    root.setAttribute('data-theme', theme);
    root.style.colorScheme = isDarkTheme ? 'dark' : 'light';
    
    try {
      localStorage.setItem('agri:theme', theme);
    } catch (e) {
      console.warn('Failed to persist theme to localStorage:', e);
    }
  }, [theme]);

  const value = useMemo(() => ({
    theme,
    setTheme,
    toggleTheme: () => setTheme(prev => (prev === 'light' ? 'dark' : prev === 'dark' ? 'night' : 'light')),
    isDark: theme !== 'light',
    isNight: theme === 'night'
  }), [theme]);

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
