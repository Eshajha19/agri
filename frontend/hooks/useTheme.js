import { useCallback, useEffect } from "react";
import { useUiStore } from "../stores/uiStore";

export const useTheme = () => {
  const { theme, setTheme } = useUiStore();

  // Apply theme to DOM + persist
  useEffect(() => {
    const root = document.documentElement;

    // Recommended: use attribute instead of class
    root.setAttribute("data-theme", theme);

    // Save to localStorage
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Load theme on first mount
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");

    if (savedTheme) {
      setTheme(savedTheme);
    } else {
      // Detect system preference
      const systemDark = window.matchMedia(
        "(prefers-color-scheme: dark)"
      ).matches;

      setTheme(systemDark ? "dark" : "light");
    }
  }, [setTheme]);

  // Safer toggle (no stale state issue)
  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, [setTheme]);

  return {
    theme,
    setTheme,
    toggleTheme,
    isDarkTheme: theme === "dark",
  };
};