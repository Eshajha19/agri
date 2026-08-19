import React from "react";
import { useTranslation } from 'react-i18next';
import "./IrrigationCard.css";

const methods = [
  {
    title: "💧 Drip Irrigation",
    description: "Delivers water directly to plant roots. Saves water and reduces weed growth. Best for row crops and orchards.",
  },
  {
    title: "🌊 Flood Irrigation",
    description: "Traditional method where fields are flooded. Simple but uses large amounts of water. Common for rice paddies.",
  },
  {
    title: "🌦️ Sprinkler Systems",
    description: "Sprays water like rainfall. Provides uniform coverage, suitable for sandy soils and diverse crops.",
  },
];

export default function IrrigationCard() {
  const { t } = useTranslation();
  return (
    <div className="irrigation-card">
      <h2>🚜 Irrigation Method Learning System</h2>
      <div className="card-grid">
        {methods.map((m, idx) => (
          <div key={idx} className="card">
            <h3>{m.title}</h3>
            <p>{m.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
