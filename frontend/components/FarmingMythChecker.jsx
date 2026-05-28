import React from "react";
import "./FarmingMythChecker.css";

const myths = [
  {
    myth: "More fertilizer = more yield",
    fact: "Excess fertilizer harms soil and reduces yield long-term.",
    verdict: "false",
  },
  {
    myth: "Drip irrigation always increases yield",
    fact: "It depends on crop type, soil, and water quality.",
    verdict: "depends",
  },
  {
    myth: "Organic farming cannot feed the world",
    fact: "Studies show organic methods can be productive with sustainable practices.",
    verdict: "false",
  },
];

export default function FarmingMythChecker() {
  return (
    <div className="myth-checker">
      <h2>🌱 Farming Myth vs Fact Checker</h2>
      <ul>
        {myths.map((item, idx) => (
          <li key={idx} className={`verdict-${item.verdict}`}>
            <strong>Myth:</strong> {item.myth}
            <br />
            <strong>Fact:</strong> {item.fact}
          </li>
        ))}
      </ul>
    </div>
  );
}
