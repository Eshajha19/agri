import React, { useState } from "react";
import "./CropComparison.css";

const cropsData = {
  wheat: { profitability: "High", water: "Medium", risk: "Low", duration: "120 days" },
  rice: { profitability: "Medium", water: "High", risk: "Medium", duration: "150 days" },
  maize: { profitability: "Medium", water: "Low", risk: "Low", duration: "90 days" },
  cotton: { profitability: "High", water: "High", risk: "High", duration: "180 days" },
};

export default function CropComparison() {
  const [crop1, setCrop1] = useState("wheat");
  const [crop2, setCrop2] = useState("rice");

  const c1 = cropsData[crop1];
  const c2 = cropsData[crop2];

  // helper to highlight better metric
  const highlight = (metric, val1, val2) => {
    if (val1 === val2) return ["", ""];
    if (metric === "risk" || metric === "water") {
      return val1 === "Low" ? ["highlight", ""] : val2 === "Low" ? ["", "highlight"] : ["", ""];
    }
    if (metric === "profitability") {
      return val1 === "High" ? ["highlight", ""] : val2 === "High" ? ["", "highlight"] : ["", ""];
    }
    if (metric === "duration") {
      const d1 = parseInt(val1);
      const d2 = parseInt(val2);
      return d1 < d2 ? ["highlight", ""] : ["", "highlight"];
    }
    return ["", ""];
  };

  return (
    <div className="comparison-tool">
      <h2>🌾 Crop Comparison Tool</h2>
      <p className="subtitle">Compare crops side‑by‑side by key metrics</p>

      <div className="selectors">
        <select value={crop1} onChange={(e) => setCrop1(e.target.value)}>
          {Object.keys(cropsData).map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <select value={crop2} onChange={(e) => setCrop2(e.target.value)}>
          {Object.keys(cropsData).map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      <table className="comparison-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>{crop1}</th>
            <th>{crop2}</th>
          </tr>
        </thead>
        <tbody>
          {[
            ["Profitability", c1.profitability, c2.profitability, "profitability"],
            ["Water Usage", c1.water, c2.water, "water"],
            ["Risk Level", c1.risk, c2.risk, "risk"],
            ["Growth Duration", c1.duration, c2.duration, "duration"],
          ].map(([metric, val1, val2, key], idx) => {
            const [h1, h2] = highlight(key, val1, val2);
            return (
              <tr key={idx}>
                <td>{metric}</td>
                <td className={h1}>{val1}</td>
                <td className={h2}>{val2}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
