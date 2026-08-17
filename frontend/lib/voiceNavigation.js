/**
 * Voice Navigation command parsing for website navigation.
 *
 * This module is intentionally frontend-only.
 * It parses a transcript into a navigation action that the UI can execute.
 */

const NORMALIZATION_REGEXPS = [
  // Collapse whitespace
  /\s+/g,
];

function normalizeTranscript(transcript) {
  return String(transcript || "")
    .toLowerCase()
    .replace(NORMALIZATION_REGEXPS[0], " ")
    .trim();
}

const NAV_KEYWORDS = [
  // path, array of keywords/phrases to match
  { path: "/", keywords: ["open home", "go to home", "home page", "home"] },
  { path: "/dashboard", keywords: ["open dashboard", "go to dashboard", "dashboard", "open dash"] },
  { path: "/about", keywords: ["open about", "go to about", "about us", "about"] },
  { path: "/how-it-works", keywords: ["open how it works", "go to how it works", "how it works"] },
  { path: "/resources", keywords: ["open resources", "go to resources", "resources"] },
  { path: "/crop-guide", keywords: ["open crop guide", "go to crop guide", "crop guide", "cropguid"] },
  { path: "/community", keywords: ["open community", "go to community", "community"] },
  { path: "/weather", keywords: ["open weather", "go to weather", "weather"] },
  { path: "/faq", keywords: ["open faq", "go to faq", "faq"] },
  { path: "/glossary", keywords: ["open glossary", "go to glossary", "glossary"] },
  { path: "/leaderboard", keywords: ["open leaderboard", "go to leaderboard", "leaderboard"] },
  { path: "/farm-finance", keywords: ["open farm finance", "go to farm finance", "farm finance"] },
  { path: "/soil-analysis", keywords: ["open soil analysis", "go to soil analysis", "soil analysis"] },
  { path: "/soil-guide", keywords: ["open soil guide", "go to soil guide", "soil guide"] },
  { path: "/disease-awareness", keywords: ["open disease awareness", "go to disease awareness", "disease awareness"] },
  { path: "/pest-detection", keywords: ["open pest detection", "go to pest detection", "pest detection"] },
  { path: "/equipment-management", keywords: ["open equipment management", "go to equipment management", "equipment management"] },
  { path: "/helpline", keywords: ["open helpline", "go to helpline", "helpline"] },
  { path: "/advisor", keywords: ["open advisor", "go to advisor", "advisor"] },
  { path: "/calendar", keywords: ["open calendar", "go to calendar", "calendar"] },
  { path: "/market-prices", keywords: ["open market prices", "go to market prices", "market prices", "marketprice", "open mi", "go to mi", "mi", "open market information", "go to market information", "market information"] },
  { path: "/farming-map", keywords: ["open farming map", "go to farming map", "farming map"] },
  { path: "/profit-calculator", keywords: ["open profit calculator", "go to profit calculator", "profit calculator"] },
  { path: "/profile-settings", keywords: ["open profile settings", "go to profile settings", "profile settings"] },
  { path: "/login", keywords: ["open login", "go to login", "login", "sign in"] },
  { path: "/voice-assistant", keywords: ["open voice assistant", "go to voice assistant", "voice assistant"] },
  { path: "/blog", keywords: ["open blog", "go to blog", "blog"] },
  { path: "/crop-planner", keywords: ["open crop planner", "go to crop planner", "crop planner"] },
  { path: "/risk-index", keywords: ["open risk index", "go to risk index", "risk index"] },
  { path: "/seed-verifier", keywords: ["open seed verifier", "go to seed verifier", "seed verifier"] },
  { path: "/pest-calendar", keywords: ["open pest calendar", "go to pest calendar", "pest calendar", "seasonal pest calendar"] },
  { path: "/yield-predictor", keywords: ["open yield predictor", "go to yield predictor", "yield predictor"] },
  { path: "/smart-farm-autopilot", keywords: ["open smart farm autopilot", "go to smart farm autopilot", "smart farm autopilot"] },
  { path: "/sustainability-analytics", keywords: ["open sustainability analytics", "go to sustainability analytics", "sustainability analytics"] },
  { path: "/spray-scheduler", keywords: ["open spray scheduler", "go to spray scheduler", "spray scheduler"] },
  { path: "/myth-checker", keywords: ["open myth checker", "go to myth checker", "myth checker"] },
  { path: "/crop-comparison", keywords: ["open crop comparison", "go to crop comparison", "crop comparison"] },
  { path: "/insurance-claim", keywords: ["open insurance claim", "go to insurance claim", "insurance claim"] },
];

function matchRoute(transcriptNormalized) {
  // Prefer longer/more specific phrases first
  const sorted = [...NAV_KEYWORDS].sort(
    (a, b) => String(b.path).length - String(a.path).length
  );

  for (const entry of sorted) {
    for (const kw of entry.keywords) {
      if (kw && transcriptNormalized.includes(String(kw))) {
        return entry.path;
      }
    }
  }
  return null;
}

/**
 * Parse a transcript into a navigation intent.
 *
 * Output:
 *   { type: 'navigate', path: string }
 *   { type: 'back' }
 *   { type: 'none' }
 */
export function parseVoiceNavigation(transcript) {
  const t = normalizeTranscript(transcript);
  if (!t) return { type: "none" };

  // Back
  if (
    t === "back" ||
    t.includes("go back") ||
    t.includes("go back please") ||
    t.includes("go back to previous") ||
    t.includes("previous page")
  ) {
    return { type: "back" };
  }

  // Go to <keyword>
  const hasNavigateIntent =
    t.includes("open ") ||
    t.includes("go to ") ||
    t.includes("navigate to ") ||
    t.startsWith("open ") ||
    t.startsWith("go to ") ||
    t.startsWith("navigate to ");

  const route = matchRoute(t);
  if (route && (hasNavigateIntent || route === "/dashboard" || route === "/about")) {
    return { type: "navigate", path: route };
  }

  // If user just says the page name (e.g., “dashboard”), still navigate.
  if (route) return { type: "navigate", path: route };

  return { type: "none" };
}

