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

  return (
    <div className="comparison-tool">
      <h2>🌾 Crop Comparison Tool</h2>
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
          <tr><td>Profitability</td><td>{c1.profitability}</td><td>{c2.profitability}</td></tr>
          <tr><td>Water Usage</td><td>{c1.water}</td><td>{c2.water}</td></tr>
          <tr><td>Risk Level</td><td>{c1.risk}</td><td>{c2.risk}</td></tr>
          <tr><td>Growth Duration</td><td>{c1.duration}</td><td>{c2.duration}</td></tr>
        </tbody>
      </table>
    </div>
  );
}