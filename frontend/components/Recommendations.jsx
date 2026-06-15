import React, { useState } from "react";
import "./Recommendations.css";

export default function Recommendations({ results }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      const formattedText = results
        .map((r, i) => `${i + 1}. ${r.crop} — ${r.reason}`)
        .join("\n");

      await navigator.clipboard.writeText(formattedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.error("Copy failed", err);
    }
  };

  if (!results || results.length === 0) {
    return <p className="no-results">No recommendations available.</p>;
  }

  return (
    <div className="recommendations">
      <ul className="recommendations-list">
        {results.map((r, i) => (
          <li key={i} className="recommendation-item">
            <strong>{r.crop}</strong>: {r.reason}
          </li>
        ))}
      </ul>
      <button className="copy-btn" onClick={handleCopy}>
        {copied ? "Copied!" : "Copy Results"}
      </button>
    </div>
  );
}
