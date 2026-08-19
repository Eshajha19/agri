import React from "react";
import { useTranslation } from "react-i18next";
import WeatherCard from "./weather/WeatherCard";
import WeatherQuickWidget from "./weather/WeatherQuickWidget";
import WeatherAlertBar from "./weather/WeatherAlertBar";
import "./Home.css";

export default function Weather() {
  const { t } = useTranslation();

  return (
    <div className="weather-page">
      <WeatherAlertBar />
      <div className="weather-page-header">
        <h1>{t("weather.title")}</h1>
        <p>{t("weather.subtitle")}</p>
      </div>
      <div className="weather-page-content">
        <WeatherQuickWidget />
        <WeatherCard />
      </div>
    </div>
  );
}
