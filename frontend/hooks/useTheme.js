import { useTheme as useThemeContext } from "../ThemeContext";

/**
 * useTheme Hook
 * 
 * A wrapper around ThemeContext to provide a consistent interface for components.
 * This ensures all theme-related logic is centralized and managed through the 
 * React Context API, avoiding direct DOM manipulation outside of the ThemeProvider.
 */
export const useTheme = () => {
  const { theme, setTheme, toggleTheme, isDark } = useThemeContext();

  return {
    theme,
    setTheme,
    toggleTheme,
    isDarkTheme: isDark,
  };
};