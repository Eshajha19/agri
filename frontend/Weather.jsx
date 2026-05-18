import React from "react";
import WeatherCard from "./weather/WeatherCard";
import WeatherQuickWidget from "./weather/WeatherQuickWidget";
import WeatherAlertBar from "./weather/WeatherAlertBar";
import "./Home.css";

export default function Weather() {
  return (
    <div className="weather-page">
      <WeatherAlertBar />
      <div className="weather-page-header">
        <h1>Weather Insights</h1>
        <p>Real-time weather forecasts and custom alerts for your farm</p>
      </div>
      <div className="weather-page-content">
        <WeatherQuickWidget />
        <WeatherCard />
      </div>
    </div>
  );
}
