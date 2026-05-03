import React from "react";
import { CloudSun, Sun, CloudRain, Cloud, CloudLightning } from "lucide-react";

const FORECAST_DATA = [
  { day: "Monday", temp: "32°C", status: "Sunny", icon: <Sun size={24} color="#f59e0b" /> },
  { day: "Tuesday", temp: "30°C", status: "Partly Cloudy", icon: <CloudSun size={24} color="#fb923c" /> },
  { day: "Wednesday", temp: "28°C", status: "Light Rain", icon: <CloudRain size={24} color="#60a5fa" /> },
  { day: "Thursday", temp: "27°C", status: "Thunderstorm", icon: <CloudLightning size={24} color="#4b5563" /> },
  { day: "Friday", temp: "29°C", status: "Cloudy", icon: <Cloud size={24} color="#9ca3af" /> },
  { day: "Saturday", temp: "31°C", status: "Sunny", icon: <Sun size={24} color="#f59e0b" /> },
  { day: "Sunday", temp: "33°C", status: "Clear Sky", icon: <Sun size={24} color="#f59e0b" /> },
];

export default function Forecast() {
  return (
    <div className="forecast-modal-content">
      <div style={{ padding: "20px", color: "#1f2937" }}>
        <h2 style={{ marginBottom: "20px", display: "flex", alignItems: "center", gap: "10px", color: "#166534" }}>
          <CloudSun size={28} /> 7-Day Weather Forecast
        </h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {FORECAST_DATA.map((item, index) => (
            <div
              key={index}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "12px 20px",
                background: "rgba(0, 0, 0, 0.03)",
                borderRadius: "12px",
                border: "1px solid rgba(0, 0, 0, 0.05)"
              }}
            >
              <div style={{ width: "100px", fontWeight: "600" }}>{item.day}</div>
              <div style={{ display: "flex", alignItems: "center", gap: "12px", flex: 1, justifyContent: "center" }}>
                {item.icon}
                <span style={{ fontSize: "0.9rem", color: "#6b7280" }}>{item.status}</span>
              </div>
              <div style={{ width: "80px", textAlign: "right", fontWeight: "700", color: "#166534" }}>
                {item.temp}
              </div>
            </div>
          ))}
        </div>
        <div style={{ 
          marginTop: "20px", 
          padding: "15px", 
          background: "#f0fdf4", 
          borderRadius: "10px", 
          border: "1px solid #bbf7d0" 
        }}>
          <p style={{ margin: 0, fontSize: "0.85rem", color: "#166534", lineHeight: "1.5" }}>
            <strong>🌱 Agricultural Insight:</strong> Rain is expected mid-week. Consider delaying irrigation on Tuesday and check for drainage readiness in low-lying areas.
          </p>
        </div>
      </div>
    </div>
  );
}