import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';

const ThemeContext = createContext();

/**
 * ThemeProvider manages the application's visual theme (light/dark).
 * It centralizes theme state and ensures synchronization with the DOM and localStorage,
 * following React's state-driven lifecycle to avoid inconsistencies.
 */
export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('agri:theme') || 'light';
    } catch {
      return 'light';
    }
  });

  // Centralized side-effect to sync React state with the DOM
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('theme-dark');
    } else {
      root.classList.remove('theme-dark');
    }
    
    // Also set data attribute for future-proofing and better selector performance
    root.setAttribute('data-theme', theme);
    
    try {
      localStorage.setItem('agri:theme', theme);
    } catch (e) {
      console.warn('Failed to persist theme to localStorage:', e);
    }
  }, [theme]);

  const value = useMemo(() => ({
    theme,
    setTheme,
    toggleTheme: () => setTheme(prev => (prev === 'dark' ? 'light' : 'dark')),
    isDark: theme === 'dark'
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
