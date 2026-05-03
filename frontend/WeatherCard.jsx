"use client";
import React, { useState, useEffect } from "react";
import "./WeatherCard.css";

export default function WeatherCard({ onClose }) {
  const [city, setCity] = useState("");
  const [weather, setWeather] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // 🔑 Use env variable instead of hardcoding
  const API_KEY = process.env.NEXT_PUBLIC_WEATHER_KEY;

  // ================= AUTO LOAD LOCATION =================
  useEffect(() => {
    getWeatherByLocation();
  }, []);

  // ================= FETCH WEATHER =================
  const fetchWeather = async (url) => {
    setLoading(true);
    setError("");
    setWeather(null);

    try {
      if (!API_KEY) {
        throw new Error("API key missing. Add NEXT_PUBLIC_WEATHER_KEY");
      }

      const res = await fetch(url);
      if (!res.ok) throw new Error("City not found or API error");

      const data = await res.json();
      setWeather(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ================= CITY SEARCH =================
  const getWeather = () => {
    if (!city.trim()) {
      setError("Please enter a city name");
      return;
    }

    fetchWeather(
      `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${API_KEY}&units=metric`
    );
  };

  // ================= LOCATION =================
  const getWeatherByLocation = () => {
    if (!navigator.geolocation) {
      setError("Geolocation not supported");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        fetchWeather(
          `https://api.openweathermap.org/data/2.5/weather?lat=${coords.latitude}&lon=${coords.longitude}&appid=${API_KEY}&units=metric`
        );
      },
      () => setError("Location access denied")
    );
  };

  // ================= FARMING INSIGHT =================
  const getFarmingAdvice = () => {
    if (!weather) return "";

    const temp = weather.main.temp;
    const humidity = weather.main.humidity;
    const wind = weather.wind.speed;

    if (temp > 35)
      return "🌡 High temperature! Increase irrigation frequency.";
    if (temp < 10)
      return "❄️ Low temperature. Protect crops from cold damage.";
    if (humidity > 80)
      return "💧 High humidity. Risk of fungal diseases.";
    if (wind > 8)
      return "🌬 Strong winds. Avoid spraying pesticides.";

    return "✅ Weather conditions are favorable for farming.";
  };

  // ================= DYNAMIC STYLE =================
  const getWeatherClass = () => {
    if (!weather) return "";
    const main = weather.weather[0].main.toLowerCase();

    if (main.includes("rain")) return "rainy";
    if (main.includes("cloud")) return "cloudy";
    if (main.includes("clear")) return "sunny";
    return "default";
  };

  // ================= UI =================
  return (
    <div className={`weather-card ${getWeatherClass()}`}>
      <button className="close-btn" onClick={onClose}>
        ✖
      </button>

      <h2>🌤 Weather Forecast</h2>
      <p className="subtitle">
        Search city or use your location for live data
      </p>

      {/* INPUT */}
      <div className="input-group">
        <input
          type="text"
          placeholder="Enter city..."
          value={city}
          onChange={(e) => setCity(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && getWeather()}
        />
        <button onClick={getWeather}>Search</button>
      </div>

      <button className="location-btn" onClick={getWeatherByLocation}>
        📍 Use My Location
      </button>

      {/* STATES */}
      {loading && <p className="loading">⏳ Fetching weather...</p>}
      {error && <p className="error">{error}</p>}

      {/* WEATHER DISPLAY */}
      {weather && (
        <div className="weather-info">
          <h3>
            {weather.name}, {weather.sys.country}
          </h3>

          <div className="icon-temp">
            <img
              src={`https://openweathermap.org/img/wn/${weather.weather[0].icon}@2x.png`}
              alt="icon"
            />
            <h1>{Math.round(weather.main.temp)}°C</h1>
          </div>

          <p className="desc">{weather.weather[0].description}</p>

          <div className="details">
            <p>💧 Humidity: {weather.main.humidity}%</p>
            <p>🌬 Wind: {weather.wind.speed} m/s</p>
            <p>🌡 Feels like: {Math.round(weather.main.feels_like)}°C</p>
            <p>ضغط: {weather.main.pressure} hPa</p>
            <p>
              🌅 Sunrise:{" "}
              {new Date(weather.sys.sunrise * 1000).toLocaleTimeString()}
            </p>
            <p>
              🌇 Sunset:{" "}
              {new Date(weather.sys.sunset * 1000).toLocaleTimeString()}
            </p>
          </div>

          {/* 🌱 FARMING ADVICE */}
          <div className="advice">
            <h4>🌱 Farming Insight</h4>
            <p>{getFarmingAdvice()}</p>
          </div>
        </div>
      )}
    </div>
  );
}