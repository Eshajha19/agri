import React, { useEffect } from "react";
import { useLocation } from "react-router-dom";

/**
 * NavigationManager Component
 * 
 * This component handles two critical aspects of Single Page Application (SPA) navigation:
 * 1. Scroll Management: Resets the scroll position to the top of the page on route changes.
 * 2. Focus Management: Shifts the keyboard focus to the primary heading (h1) of the new page.
 * 
 * Why this is important for accessibility (A11y):
 * In traditional multi-page websites, the browser automatically resets scroll and focus on navigation.
 * In SPAs, the browser stays on the same document, so we must manually manage focus to ensure 
 * keyboard and screen reader users aren't left stranded on the navigation menu or at the bottom 
 * of the "new" page.
 * 
 * @returns {null} This component doesn't render any UI, it only handles side effects.
 */
const NavigationManager = () => {
  // Access the current location (pathname, search, hash) from React Router
  const { pathname, hash } = useLocation();

  useEffect(() => {
    // 1. Scroll to top on every route change (unless there is a hash fragment)
    if (!hash) {
      window.scrollTo({ top: 0, behavior: "instant" });
    }

    // 2. Focus Management Logic
    // We use a small timeout to ensure the new component has finished rendering
    // and the DOM is updated with the new content.
    const focusTimeout = setTimeout(() => {
      // Find the first h1 element on the page
      const primaryHeading = document.querySelector("h1");

      if (primaryHeading) {
        /**
         * By default, h1 elements are not focusable via script unless they have a tabIndex.
         * We set tabIndex to -1 so it can be focused programmatically but doesn't 
         * interfere with the normal tabbing order.
         */
        primaryHeading.setAttribute("tabindex", "-1");
        
        // Remove the focus outline if we don't want it to look "selected" 
        // but still want the screen reader to announce it.
        // CSS can also handle this: h1:focus { outline: none; }
        primaryHeading.style.outline = "none";

        // Shift focus to the heading
         primaryHeading.focus();
         
         if (process.env.NODE_ENV !== 'production') {
           console.log(`[A11y] Focus shifted to: ${primaryHeading.innerText}`);
         }
      } else {
        // Fallback: If no h1 is found, focus the main container or body
        const mainContent = document.querySelector("main") || document.body;
        mainContent.setAttribute("tabindex", "-1");
        mainContent.focus();
      }
    }, 100); // 100ms is usually enough for the component tree to settle

    return () => clearTimeout(focusTimeout);
  }, [pathname, hash]);

  // Handle hash fragments (e.g., /about#team)
  useEffect(() => {
    if (hash) {
      const id = hash.replace("#", "");
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
        element.setAttribute("tabindex", "-1");
        element.focus();
      }
    }
  }, [hash]);

  return null;
};

/**
 * SkipLink Component
 * 
 * A 'Skip to Main Content' link is essential for keyboard users.
 * It allows them to bypass the repetitive navigation menu and jump 
 * straight to the primary content of the page.
 */
export const SkipLink = () => {
  return (
    <a 
      href="#main-content" 
      className="skip-link"
      style={{
        position: 'absolute',
        top: '-100px', // Hide it off-screen
        left: '0',
        background: '#2ecc71',
        color: 'white',
        padding: '10px 20px',
        zIndex: '10000',
        transition: 'top 0.3s',
        textDecoration: 'none',
        fontWeight: 'bold',
        borderRadius: '0 0 5px 0'
      }}
      // Show it when it receives focus via Tab key
      onFocus={(e) => e.target.style.top = '0'}
      onBlur={(e) => e.target.style.top = '-100px'}
    >
      Skip to Main Content
    </a>
  );
};

export default NavigationManager;
