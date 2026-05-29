# Seasonal Farming Strategy Guide

This document describes the Seasonal Farming Strategy Guide UI added to the frontend. It provides concise, actionable strategies for Kharif (monsoon), Rabi (winter) and Zaid (summer) seasons.

Files added
- `frontend/SeasonalFarmingStrategyGuide.jsx` — React component rendered as a modal from the Advisor page.
- `frontend/SeasonalFarmingStrategyGuide.css` — Responsive styling for the guide.

Behavior
- The guide is opened from the Advisor features grid ("Seasonal Farming Strategy Guide" card).
- It appears as a modal overlay using the existing `weather-overlay` / `weather-popup` styles provided in `Advisor.jsx`.
- Content is organized into three cards (Kharif, Rabi, Zaid) with brief bullets for sowing, irrigation, soil & nutrition, pests & harvest.
- Actions: Close button and Print button (triggers `window.print()`).

Accessibility & Responsiveness
- Uses semantic headings and lists for screen reader clarity.
- Responsive grid: 3 columns desktop, 2 columns tablet, 1 column on small screens.

Testing
- Open the Advisor view and click the "Seasonal Farming Strategy Guide" card. The modal should open without console errors.
- Resize the browser to confirm responsive behavior.

If you'd like more details (links to crop-specific timing, or downloadable PDF versions), I can extend the guide.
