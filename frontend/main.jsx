import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'
import NavigationManager from './NavigationManager.jsx'
import './index.css'
import './lib/i18n.js'
import { registerSW } from 'virtual:pwa-register'
import { initializeFirebase } from './lib/firebase'

console.log('Fasal Saathi: Booting application...');

// Initialize Firebase early (supports runtime config fetch from backend)
void initializeFirebase();

// Register service worker with auto-update
const updateSW = registerSW({
  onNeedRefresh() {
    console.log('New content available, reloading...');
    if (confirm('New version available. Reload now?')) {
      updateSW(true);
    }
  },
  onOfflineReady() {
    console.log('App ready for offline usage.');
  }
});

// Forcibly remove preloader once React starts
const removePreloader = () => {
  const preloader = document.querySelector('.preloader');
  if (preloader) {
    preloader.style.opacity = '0';
    setTimeout(() => preloader.remove(), 300);
  }
};

import { ThemeProvider } from './ThemeContext.jsx'

const rootElement = document.getElementById("root");
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(
    <StrictMode>
      <BrowserRouter>
        <ThemeProvider>
          <NavigationManager />
          <ErrorBoundary>
            <App />
          </ErrorBoundary>
        </ThemeProvider>
      </BrowserRouter>
    </StrictMode>
  );
  removePreloader();
} else {
  console.error("Critical Error: Root element #root not found in index.html");
}
