import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

// Global error catcher for debugging silent crashes
window.addEventListener('error', (e) => {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:red;color:white;z-index:99999;padding:20px;font-family:monospace;white-space:pre-wrap;overflow:auto;';
  overlay.innerHTML = '<h2>Global Error Caught!</h2><p>' + e.message + '</p><pre>' + (e.error?.stack || '') + '</pre>';
  document.body.appendChild(overlay);
});

window.addEventListener('unhandledrejection', (e) => {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:darkred;color:white;z-index:99999;padding:20px;font-family:monospace;white-space:pre-wrap;overflow:auto;';
  overlay.innerHTML = '<h2>Unhandled Promise Rejection!</h2><p>' + e.reason + '</p><pre>' + (e.reason?.stack || '') + '</pre>';
  document.body.appendChild(overlay);
});

import { BrowserRouter } from 'react-router-dom'
import './index.css'
import './lib/i18n' // Initialize i18n
import App from './App.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'
import NavigationManager from './NavigationManager.jsx'

// Import i18n configuration for offline translation support
import './lib/i18n.js'

import { registerSW } from 'virtual:pwa-register'

// Register service worker for offline support
registerSW({ immediate: true })

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <NavigationManager />
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </BrowserRouter>
  </StrictMode>,
)
