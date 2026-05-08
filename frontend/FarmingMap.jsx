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
    html: `<div style="font-size: 24px; display: flex; align-items: center; justify-content: center; color: #3498db;">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="M20 12h2"/><path d="m19.07 4.93-1.41 1.41"/><path d="M15.947 12.65a4 4 0 0 0-5.925-4.128"/><path d="M13 22H7a5 5 0 1 1 4.9-6H13a3 3 0 0 1 0 6Z"/></svg>
    </div>`,
    iconSize: [30, 30],
    className: 'weather-marker',
  });

const createCropIcon = () =>
  L.divIcon({
    html: `<div style="font-size: 24px; display: flex; align-items: center; justify-content: center; color: #2ecc71;">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C10.9 14.36 12 15 12 18c0 2.5-2.03 4-5 4-.74 0-1.55-.18-3-.5L2 21Z"/></svg>
    </div>`,
    iconSize: [30, 30],
    className: 'crop-marker',
  });

const createUserIcon = () =>
  L.divIcon({
    html: `<div style="font-size: 28px; display: flex; align-items: center; justify-content: center; color: #e74c3c;">
      <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>
    </div>`,
    iconSize: [32, 32],
    className: 'user-marker',
  });

const createAlertIcon = () =>
  L.divIcon({
    html: `<div style="font-size: 24px; display: flex; align-items: center; justify-content: center; color: #f1c40f;">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
    </div>`,
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


  const updateTileLayer = useCallback(() => {
    if (!map.current) return;
    
    // Remove existing tile layers
    map.current.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) {
        map.current.removeLayer(layer);
      }
    });

    try {
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 19,
      }).addTo(map.current);
    } catch (err) {
      console.error("Tile layer error:", err);
    }
  }, []);

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
              <p>Temp: ${point.temp}°C</p>
              <p>Humidity: ${point.humidity}%</p>
              <p>${point.condition}</p>
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
              <p>Crop: ${point.crop}</p>
              <p>Area: ${point.area}</p>
              <p>Status: ${point.status}</p>
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
              <strong>${point.title}</strong>
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
