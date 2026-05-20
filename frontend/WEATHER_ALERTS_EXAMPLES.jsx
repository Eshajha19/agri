/**
 * Example: How to integrate WeatherAlerts into your Dashboard
 * 
 * This file shows the recommended way to add weather alerts
 * to your existing dashboard or farm page.
 */

import React, { useState, useEffect } from "react";
import WeatherAlerts from "./WeatherAlerts";
import useWeatherLocation from "./hooks/useWeatherLocation";

/**
 * Example 1: Basic Integration
 * Simply display weather alerts for a known location
 */
export function DashboardWithWeatherAlerts() {
  return (
    <div className="dashboard">
      <h1>Farm Dashboard</h1>
      
      {/* Add the WeatherAlerts component */}
      <WeatherAlerts
        latitude={31.1471}  // Punjab latitude
        longitude={75.3412} // Punjab longitude
        location="Punjab, India"
        crop="rice"
      />
      
      {/* Rest of your dashboard content */}
    </div>
  );
}

/**
 * Example 2: Dynamic Location Selection
 * Allow users to search for and select their farm location
 */
export function DashboardWithLocationPicker() {
  const [location, setLocation] = useState(null);
  const [searchInput, setSearchInput] = useState("");
  const [searching, setSearching] = useState(false);
  const [crop, setCrop] = useState("rice");

  const handleLocationSearch = async (e) => {
    e.preventDefault();
    if (!searchInput.trim()) return;

    setSearching(true);
    try {
      const response = await fetch("/api/weather/geocode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ location: searchInput }),
      });

      if (response.ok) {
        const data = await response.json();
        setLocation({
          name: data.location,
          latitude: data.latitude,
          longitude: data.longitude,
        });
      }
    } catch (error) {
      console.error("Location search failed:", error);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="dashboard">
      <h1>Farm Dashboard</h1>

      {/* Location Search */}
      <form onSubmit={handleLocationSearch} className="location-search">
        <input
          type="text"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="Enter your location (e.g., Punjab, New Delhi)"
          disabled={searching}
        />
        <button type="submit" disabled={searching}>
          {searching ? "Searching..." : "Search"}
        </button>
      </form>

      {/* Crop Selection */}
      <select value={crop} onChange={(e) => setCrop(e.target.value)}>
        <option value="">Select Crop</option>
        <option value="rice">Rice</option>
        <option value="wheat">Wheat</option>
        <option value="maize">Maize</option>
        <option value="cotton">Cotton</option>
        <option value="sugarcane">Sugarcane</option>
        <option value="potato">Potato</option>
        <option value="tomato">Tomato</option>
        <option value="onion">Onion</option>
      </select>

      {/* Display Weather Alerts */}
      {location ? (
        <WeatherAlerts
          latitude={location.latitude}
          longitude={location.longitude}
          location={location.name}
          crop={crop || undefined}
        />
      ) : (
        <div className="placeholder">
          📍 Search for your farm location to view weather alerts
        </div>
      )}
    </div>
  );
}

/**
 * Example 3: Using with Firebase User Data
 * Fetch user's saved farm location from Firestore
 */
export function DashboardWithUserLocation() {
  const [farmData, setFarmData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFarmLocation = async () => {
      try {
        // Assuming you have user ID and Firebase auth set up
        const userId = "user_123"; // Get from auth context
        const docRef = doc(db, "users", userId, "farms", "primary");
        const unsubscribe = onSnapshot(docRef, (docSnap) => {
          if (docSnap.exists()) {
            setFarmData(docSnap.data());
          }
          setLoading(false);
        });

        return () => unsubscribe();
      } catch (error) {
        console.error("Failed to fetch farm location:", error);
        setLoading(false);
      }
    };

    fetchFarmLocation();
  }, []);

  if (loading) {
    return <div>Loading farm data...</div>;
  }

  if (!farmData?.latitude || !farmData?.longitude) {
    return <div>Please set up your farm location first</div>;
  }

  return (
    <div className="dashboard">
      <h1>{farmData.name || "My Farm"} Dashboard</h1>

      <WeatherAlerts
        latitude={farmData.latitude}
        longitude={farmData.longitude}
        location={farmData.location || "Unknown"}
        crop={farmData.primaryCrop}
      />

      {/* Rest of your dashboard */}
    </div>
  );
}

/**
 * Example 4: Multiple Crops Monitoring
 * Show weather alerts for multiple crops at the same location
 */
export function MultiCropWeatherMonitor() {
  const crops = ["rice", "wheat", "maize"];
  const location = {
    latitude: 31.1471,
    longitude: 75.3412,
    name: "Punjab, India",
  };

  return (
    <div className="dashboard">
      <h1>Multi-Crop Weather Monitoring</h1>

      <div className="crops-grid">
        {crops.map((crop) => (
          <WeatherAlerts
            key={crop}
            latitude={location.latitude}
            longitude={location.longitude}
            location={location.name}
            crop={crop}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * Example 5: Alert Integration with Notifications
 * Fetch alerts and combine with existing notification system
 */
export function DashboardWithAlertNotifications() {
  const [weatherAlerts, setWeatherAlerts] = useState([]);
  const [farmAlerts, setFarmAlerts] = useState([]);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        // Fetch weather alerts
        const weatherRes = await fetch("/api/weather/alerts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: 31.1471,
            longitude: 75.3412,
            location: "Punjab",
            crop: "rice",
          }),
        });

        if (weatherRes.ok) {
          const weatherData = await weatherRes.json();
          setWeatherAlerts(weatherData.alerts.alerts);
        }

        // Fetch existing farm alerts
        const farmRes = await fetch("/api/notifications?crop=rice");
        if (farmRes.ok) {
          const farmData = await farmRes.json();
          setFarmAlerts(farmData.data);
        }
      } catch (error) {
        console.error("Failed to fetch alerts:", error);
      }
    };

    fetchAlerts();
    
    // Auto-refresh every 30 minutes
    const interval = setInterval(fetchAlerts, 30 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const allAlerts = [...weatherAlerts, ...farmAlerts];
  const criticalAlerts = allAlerts.filter((a) => a.severity === "critical");
  const highAlerts = allAlerts.filter((a) => a.severity === "high");

  return (
    <div className="dashboard">
      <h1>Farm Dashboard</h1>

      {/* Alert Summary */}
      <div className="alert-summary">
        <div className="alert-stat critical">
          <span className="count">{criticalAlerts.length}</span>
          <span className="label">Critical Alerts</span>
        </div>
        <div className="alert-stat high">
          <span className="count">{highAlerts.length}</span>
          <span className="label">High Alerts</span>
        </div>
      </div>

      {/* Weather Alerts Component */}
      <WeatherAlerts
        latitude={31.1471}
        longitude={75.3412}
        location="Punjab, India"
        crop="rice"
      />

      {/* Rest of dashboard */}
    </div>
  );
}

export default DashboardWithWeatherAlerts;
