import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import './lib/i18n' // Initialize i18n
import App from './App.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'
import NavigationManager from './NavigationManager.jsx'

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
