

import React, { useState } from "react";
import "../css/WeatherCard.css";

export default function WeatherCard({ onClose }) {   // ✅ accept onClose
  const [city, setCity] = useState("");
  const [weather, setWeather] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const API_KEY = ""; // Replace with your OpenWeatherMap API key
  // Fetch weather by city
  const getWeather = async () => {
    if (!city.trim()) {
      setError("Please enter a city name");
      setWeather(null);
      return;
    }
    fetchWeather(
      `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${API_KEY}&units=metric`
    );
  };

  // Fetch weather by geolocation
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

  const fetchWeather = async (url) => {
    setLoading(true);
    setError("");
    setWeather(null);
    try {
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

  return (
    <div className="weather-card">
      {/* ✅ Close Button */}
      <button className="close-btn" onClick={onClose}>
        ✖
      </button>

      <h2>🌤 Weather Forecast</h2>
      <p className="subtitle">Search for any city or use your location</p>

      <div className="input-group">
        <input
          id="weather-city"
          name="city"
          type="text"
          placeholder="Enter city..."
          value={city}
          onChange={(e) => setCity(e.target.value)}
        />
        <button onClick={getWeather}>Search</button>
      </div>

      <button className="location-btn" onClick={getWeatherByLocation}>
        📍 Use My Location
      </button>

      {loading && <p className="loading">⏳ Fetching weather...</p>}
      {error && <p className="error">{error}</p>}

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
          </div>
        </div>
      )}
    </div>
  );
}

