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
  { path: "/dashboard", keywords: ["open dashboard", "go to dashboard", "dashboard", "open dash"] },
  { path: "/about", keywords: ["open about", "go to about", "about"] },
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

