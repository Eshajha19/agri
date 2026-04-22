import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  FaMapMarkerAlt,
  FaCloud,
  FaLeaf,
  FaExclamationTriangle,
  FaTimes,
  FaLocationArrow,
} from 'react-icons/fa';
import './FarmingMap.css';

// Custom icons
const createWeatherIcon = () =>
  L.divIcon({
    html: '<div style="font-size: 24px; display: flex; align-items: center; justify-content: center;">🌤️</div>',
    iconSize: [30, 30],
    className: 'weather-marker',
  });

const createCropIcon = () =>
  L.divIcon({
    html: '<div style="font-size: 24px; display: flex; align-items: center; justify-content: center;">🌾</div>',
    iconSize: [30, 30],
    className: 'crop-marker',
  });

const createUserIcon = () =>
  L.divIcon({
    html: '<div style="font-size: 28px; display: flex; align-items: center; justify-content: center;">📍</div>',
    iconSize: [32, 32],
    className: 'user-marker',
  });

const createAlertIcon = () =>
  L.divIcon({
    html: '<div style="font-size: 24px; display: flex; align-items: center; justify-content: center;">⚠️</div>',
    iconSize: [30, 30],
    className: 'alert-marker',
  });

export default function FarmingMap() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const markersRef = useRef({});

  const [userLocation, setUserLocation] = useState(null);
  const [mapError, setMapError] = useState(null);
  const [selectedMarker, setSelectedMarker] = useState(null);
  const [showWeatherLayer, setShowWeatherLayer] = useState(true);
  const [showCropLayer, setShowCropLayer] = useState(true);
  const [showAlertLayer, setShowAlertLayer] = useState(true);

  // Initialize map
  useEffect(() => {
    if (mapContainer.current && !map.current) {
      // Create map centered on India
      map.current = L.map(mapContainer.current).setView([20.5937, 78.9629], 5);

      // Add tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
      }).addTo(map.current);
    }

    return () => {
      // Cleanup
    };
  }, []);

  // Fetch user location
  useEffect(() => {
    if (navigator.geolocation) {
      const timeoutId = setTimeout(() => {
        setMapError('Location request timed out');
      }, 5000);

      navigator.geolocation.getCurrentPosition(
        (position) => {
          clearTimeout(timeoutId);
          const { latitude, longitude } = position.coords;
          setUserLocation([latitude, longitude]);
          if (map.current) {
            map.current.setView([latitude, longitude], 12);
          }
        },
        (error) => {
          clearTimeout(timeoutId);
          console.error('Geolocation error:', error.message);
          setMapError('Location access denied. Using default location.');
          setUserLocation([20.5937, 78.9629]);
        }
      );
    } else {
      setMapError('Geolocation not supported');
      setUserLocation([20.5937, 78.9629]);
    }
  }, []);

  // Add/remove user location marker
  useEffect(() => {
    if (map.current && userLocation) {
      // Remove existing user marker
      if (markersRef.current.userMarker) {
        map.current.removeLayer(markersRef.current.userMarker);
      }

      // Add new user marker
      const userMarker = L.marker(userLocation, {
        icon: createUserIcon(),
      })
        .addTo(map.current)
        .bindPopup(`
          <div class="map-popup">
            <strong>Your Location</strong>
            <p>Lat: ${userLocation[0].toFixed(4)}</p>
            <p>Lng: ${userLocation[1].toFixed(4)}</p>
          </div>
        `);

      markersRef.current.userMarker = userMarker;
    }
  }, [userLocation]);

  // Add weather markers
  useEffect(() => {
    if (!map.current || !userLocation) return;

    // Remove old weather markers
    if (markersRef.current.weatherMarkers) {
      markersRef.current.weatherMarkers.forEach((marker) => {
        map.current.removeLayer(marker);
      });
    }

    if (showWeatherLayer) {
      const weatherPoints = [
        {
          id: 'weather_1',
          lat: userLocation[0] + 0.02,
          lng: userLocation[1] + 0.02,
          title: 'Weather Station 1',
          temp: 28,
          humidity: 65,
          condition: 'Partly Cloudy',
        },
        {
          id: 'weather_2',
          lat: userLocation[0] - 0.02,
          lng: userLocation[1] + 0.02,
          title: 'Weather Station 2',
          temp: 26,
          humidity: 70,
          condition: 'Cloudy',
        },
      ];

      const newMarkers = weatherPoints.map((point) => {
        const marker = L.marker([point.lat, point.lng], {
          icon: createWeatherIcon(),
        })
          .addTo(map.current)
          .bindPopup(`
            <div class="map-popup weather-popup">
              <strong>${point.title}</strong>
              <p>🌡️ Temp: ${point.temp}°C</p>
              <p>💧 Humidity: ${point.humidity}%</p>
              <p>☁️ ${point.condition}</p>
            </div>
          `);

        marker.on('click', () => {
          setSelectedMarker(point);
        });

        return marker;
      });

      markersRef.current.weatherMarkers = newMarkers;
    }
  }, [showWeatherLayer, userLocation]);

  // Add crop markers
  useEffect(() => {
    if (!map.current || !userLocation) return;

    // Remove old crop markers
    if (markersRef.current.cropMarkers) {
      markersRef.current.cropMarkers.forEach((marker) => {
        map.current.removeLayer(marker);
      });
    }

    if (showCropLayer) {
      const cropPoints = [
        {
          id: 'crop_1',
          lat: userLocation[0] - 0.02,
          lng: userLocation[1] - 0.02,
          title: 'Paddy Field A',
          crop: 'Paddy',
          area: '5 acres',
          status: 'Good',
        },
        {
          id: 'crop_2',
          lat: userLocation[0] + 0.02,
          lng: userLocation[1] + 0.02,
          title: 'Wheat Field B',
          crop: 'Wheat',
          area: '3 acres',
          status: 'Good',
        },
        {
          id: 'crop_3',
          lat: userLocation[0] + 0.01,
          lng: userLocation[1] - 0.03,
          title: 'Vegetable Plot C',
          crop: 'Vegetables',
          area: '2 acres',
          status: 'Needs Attention',
        },
      ];

      const newMarkers = cropPoints.map((point) => {
        const marker = L.marker([point.lat, point.lng], {
          icon: createCropIcon(),
        })
          .addTo(map.current)
          .bindPopup(`
            <div class="map-popup crop-popup">
              <strong>${point.title}</strong>
              <p>🌾 Crop: ${point.crop}</p>
              <p>📍 Area: ${point.area}</p>
              <p>✅ Status: ${point.status}</p>
            </div>
          `);

        marker.on('click', () => {
          setSelectedMarker(point);
        });

        return marker;
      });

      markersRef.current.cropMarkers = newMarkers;
    }
  }, [showCropLayer, userLocation]);

  // Add alert markers
  useEffect(() => {
    if (!map.current || !userLocation) return;

    // Remove old alert markers
    if (markersRef.current.alertMarkers) {
      markersRef.current.alertMarkers.forEach((marker) => {
        map.current.removeLayer(marker);
      });
    }

    if (showAlertLayer) {
      const alertPoints = [
        {
          id: 'alert_1',
          lat: userLocation[0] - 0.03,
          lng: userLocation[1] + 0.03,
          title: 'Heavy Rain Alert',
          severity: 'High',
          message: 'Heavy rainfall expected in 2 hours',
        },
      ];

      const newMarkers = alertPoints.map((point) => {
        const marker = L.marker([point.lat, point.lng], {
          icon: createAlertIcon(),
        })
          .addTo(map.current)
          .bindPopup(`
            <div class="map-popup alert-popup">
              <strong>⚠️ ${point.title}</strong>
              <p>Severity: ${point.severity}</p>
              <p>${point.message}</p>
            </div>
          `);

        marker.on('click', () => {
          setSelectedMarker(point);
        });

        return marker;
      });

      markersRef.current.alertMarkers = newMarkers;
    }
  }, [showAlertLayer, userLocation]);

  const handleLocateUser = () => {
    if (userLocation && map.current) {
      map.current.setView(userLocation, 12);
    }
  };

  return (
    <div className="farming-map-container">
      <div className="map-controls">
        <button className="map-control-btn locate-btn" onClick={handleLocateUser} title="Locate me">
          <FaLocationArrow /> Locate Me
        </button>
      </div>

      <div className="map-layers-panel">
        <h3>Map Layers</h3>
        <div className="layer-toggle">
          <label>
            <input
              type="checkbox"
              checked={showWeatherLayer}
              onChange={(e) => setShowWeatherLayer(e.target.checked)}
            />
            <FaCloud /> Weather Data
          </label>
        </div>
        <div className="layer-toggle">
          <label>
            <input
              type="checkbox"
              checked={showCropLayer}
              onChange={(e) => setShowCropLayer(e.target.checked)}
            />
            <FaLeaf /> Crop Fields
          </label>
        </div>
        <div className="layer-toggle">
          <label>
            <input
              type="checkbox"
              checked={showAlertLayer}
              onChange={(e) => setShowAlertLayer(e.target.checked)}
            />
            <FaExclamationTriangle /> Alerts
          </label>
        </div>
      </div>

      {mapError && <div className="map-error">{mapError}</div>}

      <div ref={mapContainer} className="map-container" style={{ height: '100%', width: '100%' }} />

      {selectedMarker && (
        <div className="marker-details-panel">
          <button className="close-details-btn" onClick={() => setSelectedMarker(null)}>
            <FaTimes />
          </button>
          <h3>{selectedMarker.title || selectedMarker.id}</h3>
          <div className="marker-details">
            {Object.entries(selectedMarker).map(
              ([key, value]) =>
                key !== 'id' &&
                key !== 'lat' &&
                key !== 'lng' &&
                key !== 'title' && (
                  <div key={key} className="detail-item">
                    <strong>{key}:</strong> {String(value)}
                  </div>
                )
            )}
          </div>
        </div>
      )}
    </div>
  );
}
