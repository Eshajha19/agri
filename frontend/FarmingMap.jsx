import React, { useEffect, useRef, useState, useCallback } from 'react';
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
  const [isLiteMode, setIsLiteMode] = useState(false);

  const updateTileLayer = useCallback(() => {
    if (!map.current) return;
    
    // Remove existing tile layers
    map.current.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) {
        map.current.removeLayer(layer);
      }
    });

    try {
      if (isLiteMode) {
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; OpenStreetMap',
          maxZoom: 19,
        }).addTo(map.current);
      } else {
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
          attribution: 'Tiles &copy; Esri',
          maxZoom: 19,
        }).addTo(map.current);
      }
    } catch (err) {
      console.error("Tile layer error:", err);
    }
  }, [isLiteMode]);

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current) return;

    // Robust check for existing map
    if (map.current) {
      map.current.remove();
      map.current = null;
    }

    try {
      const mapInstance = L.map(mapContainer.current, {
        preferCanvas: true,
        zoomControl: false,
      }).setView([20.5937, 78.9629], 5);

      L.control.zoom({ position: 'bottomright' }).addTo(mapInstance);
      map.current = mapInstance;
      
      updateTileLayer();
    } catch (err) {
      console.error("Map initialization error:", err);
      setMapError("Failed to initialize map. Please refresh.");
    }

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []); // Run once on mount

  // Handle Lite Mode toggle separately
  useEffect(() => {
    if (map.current) {
      updateTileLayer();
    }
  }, [isLiteMode, updateTileLayer]);

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
          setMapError(null);
          if (map.current) {
            map.current.setView([latitude, longitude], 12);
          }
        },
        (error) => {
          clearTimeout(timeoutId);
          let errorMsg = 'Location access denied. Using default location.';
          if (error.code === error.PERMISSION_DENIED) {
            errorMsg = 'Please enable location access to see your position on the map.';
          }
          setMapError(errorMsg);
          setUserLocation([20.5937, 78.9629]);
          if (map.current) {
            map.current.setView([20.5937, 78.9629], 5);
          }
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
        <div className="layer-toggle performance-toggle">
          <label className="lite-mode-label">
            <input
              type="checkbox"
              checked={isLiteMode}
              onChange={(e) => setIsLiteMode(e.target.checked)}
            />
            <FaLocationArrow /> High Performance (Lite Mode)
          </label>
        </div>
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
