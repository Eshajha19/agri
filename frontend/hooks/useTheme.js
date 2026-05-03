import { useCallback, useEffect, useMemo } from "react";
import { useUiStore } from "../stores/uiStore";

export const useTheme = () => {
  const { theme, setTheme } = useUiStore();

  
  useEffect(() => {
    const root = document.documentElement;

    // remove all possible themes first
    root.classList.remove(
      "theme-light",
      "theme-dark",
      "theme-comfort"
    );

    // apply current theme
    root.classList.add(`theme-${theme}`);
  }, [theme]);


  const isDarkTheme = theme === "dark";
  const isLightTheme = theme === "light";
  const isComfortTheme = theme === "comfort";

  /* =========================
     ACTIONS
  ========================= */

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      if (prev === "light") return "dark";
      if (prev === "dark") return "comfort";
      return "light";
    });
  }, [setTheme]);

  const changeTheme = useCallback(
    (newTheme) => setTheme(newTheme),
    [setTheme]
  );

  
  return useMemo(
    () => ({
      theme,
      setTheme: changeTheme,
      toggleTheme,

      isDarkTheme,
      isLightTheme,
      isComfortTheme,
    }),
    [
      theme,
      changeTheme,
      toggleTheme,
      isDarkTheme,
      isLightTheme,
      isComfortTheme,
    ]
  );
};