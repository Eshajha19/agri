import React, { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  FaMapMarkerAlt, FaCloud, FaLeaf, FaExclamationTriangle,
  FaTimes, FaLocationArrow, FaDownload, FaWifi, FaDatabase,
  FaTrash, FaDrawPolygon, FaCheck, FaSatellite, FaMap,
  FaInfoCircle, FaSync, FaUsers, FaShieldAlt
} from 'react-icons/fa';
import * as geofire from "geofire-common";
import { db, auth, isFirebaseConfigured } from "./lib/firebase";
import { 
  collection, doc, setDoc, query, orderBy, 
  startAt, endAt, onSnapshot, serverTimestamp, deleteDoc 
} from "firebase/firestore";
import offlineMapService from './services/offlineMapService';
import './FarmingMap.css';

// ── Icons ────────────────────────────────────────────────────────────────────
const mkIcon = (emoji, cls) =>
  L.divIcon({ html: `<div class="fm-icon">${emoji}</div>`, iconSize: [32, 32], className: cls });

const ICONS = {
  weather: () => mkIcon('🌤️', 'weather-marker'),
  crop:    () => mkIcon('🌾', 'crop-marker'),
  user:    () => mkIcon('📍', 'user-marker'),
  farmer:  (name) => L.divIcon({ 
    html: `<div class="fm-farmer-marker">
             <div class="farmer-avatar">🚜</div>
             <div class="farmer-label">${name}</div>
           </div>`, 
    iconSize: [40, 40], 
    className: 'farmer-marker-container' 
  }),
  alert:   (severity) => mkIcon('⚠️', `alert-marker severity-${severity?.toLowerCase() || 'medium'}`),
  field:   () => mkIcon('🟢', 'field-marker'),
};

const TILE_URLS = {
  satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  street:    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
};

const TILE_ATTR = {
  satellite: 'Tiles &copy; Esri',
  street:    '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a>',
};

// ── Constants ─────────────────────────────────────────────────────────────────
const QUERY_RADIUS_M = 10000; // 10km search radius

// ── Component ─────────────────────────────────────────────────────────────────
export default function FarmingMap() {
  const mapContainer = useRef(null);
  const map          = useRef(null);
  const markersRef   = useRef({
    weather: [], crops: [], alerts: [], fieldPolygons: [], 
    farmers: {}, remoteAlerts: {}, remoteBoundaries: {}
  });
  const drawLayer    = useRef(null);
  const drawPoints   = useRef([]);
  const tileLayers   = useRef({});
  const syncTimeout  = useRef(null);

  // Core state
  const [userLocation,    setUserLocation]    = useState(null);
  const [mapError,        setMapError]        = useState(null);
  const [selectedMarker,  setSelectedMarker]  = useState(null);
  const [mapStyle,        setMapStyle]        = useState('satellite');
  const [isOnline,        setIsOnline]        = useState(navigator.onLine);

  // Layer toggles
  const [showWeather,  setShowWeather]  = useState(false); // Default false for performance
  const [showFarmers,  setShowFarmers]  = useState(true);
  const [showAlerts,   setShowAlerts]   = useState(true);
  const [showFields,   setShowFields]   = useState(true);
  const [showCrops,    setShowCrops]    = useState(false);

  // Real-time data state
  const [nearbyFarmers,    setNearbyFarmers]    = useState([]);
  const [nearbyAlerts,     setNearbyAlerts]     = useState([]);
  const [sharedBoundaries, setSharedBoundaries] = useState([]);

  // Offline / download
  const [offlineRegions,  setOfflineRegions]  = useState([]);
  const [cacheStats,      setCacheStats]      = useState(null);
  const [downloading,     setDownloading]     = useState(false);
  const [downloadProgress,setDownloadProgress]= useState(null);
  const [showOfflinePanel,setShowOfflinePanel]= useState(false);
  const [downloadError,   setDownloadError]   = useState(null);

  // Field drawing
  const [drawMode,     setDrawMode]     = useState(false);
  const [fieldName,    setFieldName]    = useState('');
  const [savedFields,  setSavedFields]  = useState([]);
  const [showFieldForm,setShowFieldForm]= useState(false);

  // GPS tracking
  const [tracking, setTracking] = useState(false);
  const watchIdRef = useRef(null);

  // ── Online/offline listener ──────────────────────────────────────────────
  useEffect(() => {
    const on  = () => { setIsOnline(true);  handleSync(); };
    const off = () => setIsOnline(false);
    window.addEventListener('online',  on);
    window.addEventListener('offline', off);
    return () => { window.removeEventListener('online', on); window.removeEventListener('offline', off); };
  }, []);

  // ── Load saved data on mount ─────────────────────────────────────────────
  useEffect(() => {
    loadOfflineData();
    offlineMapService.pruneExpiredTiles().catch(() => {});
  }, []);

  async function loadOfflineData() {
    const [regions, fields, stats] = await Promise.all([
      offlineMapService.getAllRegionMeta(),
      offlineMapService.getAllFields(),
      offlineMapService.getCacheStats(),
    ]);
    setOfflineRegions(regions);
    setSavedFields(fields);
    setCacheStats(stats);
  }

  // ── Map initialise ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapContainer.current) return;
    if (map.current) { map.current.remove(); map.current = null; }

    try {
      const m = L.map(mapContainer.current, { preferCanvas: true, zoomControl: false })
        .setView([20.5937, 78.9629], 5);
      L.control.zoom({ position: 'bottomright' }).addTo(m);
      map.current = m;
      addTileLayer(m, 'satellite');
    } catch {
      setMapError('Failed to initialise map. Please refresh.');
    }

    return () => {
      if (watchIdRef.current) navigator.geolocation.clearWatch(watchIdRef.current);
      if (map.current) { map.current.remove(); map.current = null; }
    };
  }, []);

  // ── Tile layer helpers ───────────────────────────────────────────────────
  function addTileLayer(m, style) {
    Object.values(tileLayers.current).forEach(l => m.removeLayer(l));
    tileLayers.current = {};
    try {
      const l = L.tileLayer(TILE_URLS[style], { attribution: TILE_ATTR[style], maxZoom: 19 }).addTo(m);
      tileLayers.current[style] = l;
    } catch {
      // Ignore tile load errors
    }
  }

  useEffect(() => {
    if (map.current) addTileLayer(map.current, mapStyle);
  }, [mapStyle]);

  // ── Initial GPS location ─────────────────────────────────────────────────
  useEffect(() => {
    if (!navigator.geolocation) {
      setMapError('Geolocation not supported.');
      setUserLocation([20.5937, 78.9629]);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      ({ coords: { latitude: lat, longitude: lng } }) => {
        setUserLocation([lat, lng]);
        setMapError(null);
        map.current?.setView([lat, lng], 13);
      },
      () => {
        setMapError('Location access denied. Showing India overview.');
        setUserLocation([20.5937, 78.9629]);
      }
    );
  }, []);

  // ── Live Location Broadcast ──────────────────────────────────────────────
  useEffect(() => {
    if (!tracking || !userLocation || !isFirebaseConfigured() || !isOnline) return;

    const broadcastLocation = async () => {
      try {
        const uid = auth.currentUser.uid;
        const [lat, lng] = userLocation;
        const hash = geofire.geohashForLocation([lat, lng]);
        
        await setDoc(doc(db, "farmer_locations", uid), {
          uid,
          lat,
          lng,
          geohash: hash,
          name: auth.currentUser.displayName || "Farmer",
          lastSeen: serverTimestamp()
        }, { merge: true });
      } catch (err) {
        console.error("Failed to broadcast location:", err);
      }
    };

    const interval = setInterval(broadcastLocation, 10000); // Every 10s
    broadcastLocation();

    return () => clearInterval(interval);
  }, [tracking, userLocation, isOnline]);

  // ── Real-time Multi-farmer & Disaster Subscription ──────────────────────
  useEffect(() => {
    if (!userLocation || !isFirebaseConfigured() || !isOnline) return;

    const center = [userLocation[0], userLocation[1]];
    const bounds = geofire.geohashQueryBounds(center, QUERY_RADIUS_M);
    const unsubs = [];

    // Helper for geofire queries
    const createGeoQuery = (collectionName, stateSetter, filterRadius = QUERY_RADIUS_M) => {
      const resultsMap = new Map();
      
      for (const b of bounds) {
        const q = query(
          collection(db, collectionName),
          orderBy("geohash"),
          startAt(b[0]),
          endAt(b[1])
        );

        const unsub = onSnapshot(q, (snapshot) => {
          snapshot.docChanges().forEach((change) => {
            const docData = change.doc.data();
            if (change.type === "removed") {
              resultsMap.delete(change.doc.id);
            } else {
              const distanceInKm = geofire.distanceBetween([docData.lat, docData.lng], center);
              if (distanceInKm <= filterRadius / 1000) {
                resultsMap.set(change.doc.id, { id: change.doc.id, ...docData, distanceInKm });
              }
            }
          });
          stateSetter(Array.from(resultsMap.values()));
        });
        unsubs.push(unsub);
      }
    };

    if (showFarmers) createGeoQuery("farmer_locations", setNearbyFarmers);
    if (showAlerts)  createGeoQuery("disaster_alerts", setNearbyAlerts);
    if (showFields)  createGeoQuery("farm_boundaries", setSharedBoundaries);

    return () => unsubs.forEach(unsub => unsub());
  }, [userLocation, isOnline, showFarmers, showAlerts, showFields]);

  // ── Render Nearby Farmers ────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current) return;
    
    // Cleanup old markers
    Object.keys(markersRef.current.farmers).forEach(id => {
      if (!nearbyFarmers.find(f => f.uid === id)) {
        map.current.removeLayer(markersRef.current.farmers[id]);
        delete markersRef.current.farmers[id];
      }
    });

    if (!showFarmers) {
      Object.values(markersRef.current.farmers).forEach(m => map.current.removeLayer(m));
      markersRef.current.farmers = {};
      return;
    }

    nearbyFarmers.forEach(farmer => {
      if (farmer.uid === auth.currentUser?.uid) return; // Skip self

      const pos = [farmer.lat, farmer.lng];
      if (markersRef.current.farmers[farmer.uid]) {
        markersRef.current.farmers[farmer.uid].setLatLng(pos);
      } else {
        markersRef.current.farmers[farmer.uid] = L.marker(pos, { icon: ICONS.farmer(farmer.name) })
          .addTo(map.current)
          .bindPopup(`<div class="map-popup"><strong>🚜 ${farmer.name}</strong><p>Live nearby</p></div>`);
      }
    });
  }, [nearbyFarmers, showFarmers]);

  // ── Render Disaster Alerts ───────────────────────────────────────────────
  useEffect(() => {
    if (!map.current) return;
    
    Object.keys(markersRef.current.remoteAlerts).forEach(id => {
      if (!nearbyAlerts.find(a => a.id === id)) {
        map.current.removeLayer(markersRef.current.remoteAlerts[id]);
        delete markersRef.current.remoteAlerts[id];
      }
    });

    if (!showAlerts) {
      Object.values(markersRef.current.remoteAlerts).forEach(m => map.current.removeLayer(m));
      markersRef.current.remoteAlerts = {};
      return;
    }

    nearbyAlerts.forEach(alert => {
      const pos = [alert.lat, alert.lng];
      if (!markersRef.current.remoteAlerts[alert.id]) {
        const marker = L.marker(pos, { icon: ICONS.alert(alert.severity) })
          .addTo(map.current)
          .bindPopup(`
            <div class="map-popup alert-popup severity-${alert.severity?.toLowerCase()}">
              <strong>⚠️ ${alert.type}</strong>
              <p>Severity: ${alert.severity}</p>
              <p>${alert.notes || ''}</p>
              <small>Reported by ${alert.reporterName}</small>
            </div>
          `);
        markersRef.current.remoteAlerts[alert.id] = marker;
        
        // Add severity circle
        const color = alert.severity === 'High' ? '#ef4444' : '#f59e0b';
        L.circle(pos, { radius: 500, color, fillColor: color, fillOpacity: 0.15 }).addTo(map.current);
      }
    });
  }, [nearbyAlerts, showAlerts]);

  // ── Render Shared Farm Boundaries ────────────────────────────────────────
  useEffect(() => {
    if (!map.current) return;

    Object.keys(markersRef.current.remoteBoundaries).forEach(id => {
      if (!sharedBoundaries.find(b => b.id === id)) {
        map.current.removeLayer(markersRef.current.remoteBoundaries[id]);
        delete markersRef.current.remoteBoundaries[id];
      }
    });

    if (!showFields) {
      Object.values(markersRef.current.remoteBoundaries).forEach(m => map.current.removeLayer(m));
      markersRef.current.remoteBoundaries = {};
      return;
    }

    sharedBoundaries.forEach(field => {
      if (markersRef.current.remoteBoundaries[field.id]) return;

      const poly = L.polygon(field.points, { 
        color: '#4ade80', 
        fillColor: '#4ade80', 
        fillOpacity: 0.1, 
        weight: 1.5,
        dashArray: '5, 5'
      })
      .addTo(map.current)
      .bindPopup(`<div class="map-popup"><strong>🌾 ${field.name}</strong><p>Farmer: ${field.reporterName}</p></div>`);
      
      markersRef.current.remoteBoundaries[field.id] = poly;
    });
  }, [sharedBoundaries, showFields]);

  // ── Continuous GPS tracking ──────────────────────────────────────────────
  const toggleTracking = useCallback(() => {
    if (tracking) {
      if (watchIdRef.current) navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
      setTracking(false);
      // Remove location from firestore when stopping
      if (isOnline && isFirebaseConfigured()) {
        try { deleteDoc(doc(db, "farmer_locations", auth.currentUser.uid)); } catch { /* Ignore cleanup error */ }
      }
    } else {
      watchIdRef.current = navigator.geolocation.watchPosition(
        ({ coords: { latitude: lat, longitude: lng } }) => {
          setUserLocation([lat, lng]);
          map.current?.panTo([lat, lng]);
        },
        () => {},
        { enableHighAccuracy: true, maximumAge: 3000 }
      );
      setTracking(true);
    }
  }, [tracking, isOnline]);

  // ── User marker ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current || !userLocation) return;
    if (markersRef.current.user) map.current.removeLayer(markersRef.current.user);
    markersRef.current.user = L.marker(userLocation, { icon: ICONS.user() })
      .addTo(map.current)
      .bindPopup(`<div class="map-popup"><strong>📍 Your Location</strong><p>Lat: ${userLocation[0].toFixed(5)}</p><p>Lng: ${userLocation[1].toFixed(5)}</p></div>`);
  }, [userLocation]);

  // ── Weather markers ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current || !userLocation) return;
    (markersRef.current.weather || []).forEach(m => map.current.removeLayer(m));
    if (!showWeather) return;
    const pts = [
      { lat: userLocation[0]+0.02, lng: userLocation[1]+0.02, title: 'Station N', temp: 28, humidity: 65, cond: 'Partly Cloudy' },
      { lat: userLocation[0]-0.02, lng: userLocation[1]+0.02, title: 'Station S', temp: 26, humidity: 70, cond: 'Cloudy' },
    ];
    markersRef.current.weather = pts.map(p =>
      L.marker([p.lat, p.lng], { icon: ICONS.weather() }).addTo(map.current)
        .bindPopup(`<div class="map-popup weather-popup"><strong>${p.title}</strong><p>🌡️ ${p.temp}°C</p><p>💧 ${p.humidity}%</p><p>☁️ ${p.cond}</p></div>`)
        .on('click', () => setSelectedMarker(p))
    );
  }, [showWeather, userLocation]);

  // ── Saved field boundaries on map ────────────────────────────────────────
  useEffect(() => {
    if (!map.current) return;
    (markersRef.current.fieldPolygons || []).forEach(l => map.current.removeLayer(l));
    if (!showFields) return;
    markersRef.current.fieldPolygons = savedFields.map(f => {
      if (!f.points?.length) return null;
      const poly = L.polygon(f.points, { color: '#4caf50', fillColor: '#4caf50', fillOpacity: 0.2, weight: 2 })
        .addTo(map.current)
        .bindPopup(`<div class="map-popup crop-popup"><strong>🟢 ${f.name}</strong><p>Points: ${f.points.length}</p><p>Saved: ${new Date(f.savedAt).toLocaleDateString()}</p></div>`);
      return poly;
    }).filter(Boolean);
  }, [savedFields, showFields]);

  // ── Draw mode ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current) return;
    if (drawMode) {
      map.current.getContainer().style.cursor = 'crosshair';
      map.current.on('click', handleMapClick);
    } else {
      map.current.getContainer().style.cursor = '';
      map.current.off('click', handleMapClick);
      // Clear temp draw layer
      if (drawLayer.current) { map.current.removeLayer(drawLayer.current); drawLayer.current = null; }
      drawPoints.current = [];
    }
    return () => { map.current?.off('click', handleMapClick); };
  }, [drawMode]);

  function handleMapClick(e) {
    drawPoints.current.push([e.latlng.lat, e.latlng.lng]);
    if (drawLayer.current) map.current.removeLayer(drawLayer.current);
    if (drawPoints.current.length >= 2) {
      drawLayer.current = L.polygon(drawPoints.current, { color: '#ff9800', dashArray: '6,6', fillOpacity: 0.15 })
        .addTo(map.current);
    } else {
      L.circleMarker(drawPoints.current[0], { radius: 5, color: '#ff9800' }).addTo(map.current);
    }
    if (drawPoints.current.length >= 3) setShowFieldForm(true);
  }

  async function saveField() {
    if (!fieldName.trim() || drawPoints.current.length < 3) return;
    const field = { id: Date.now().toString(), name: fieldName.trim(), points: [...drawPoints.current], savedAt: Date.now() };
    await offlineMapService.saveFieldBoundary(field);
    
    // Also share to firestore if online
    if (isOnline && isFirebaseConfigured()) {
      try {
        const center = drawPoints.current[0];
        const hash = geofire.geohashForLocation([center[0], center[1]]);
        await setDoc(doc(db, "farm_boundaries", field.id), {
          ...field,
          geohash: hash,
          lat: center[0],
          lng: center[1],
          reporterId: auth.currentUser.uid,
          reporterName: auth.currentUser.displayName || "Farmer",
          createdAt: serverTimestamp()
        });
      } catch {
        // Shared boundary sync failed
      }
    }

    setSavedFields(prev => [...prev, field]);
    setFieldName('');
    setDrawMode(false);
    setShowFieldForm(false);
  }

  async function removeField(id) {
    await offlineMapService.deleteField(id);
    if (isOnline && isFirebaseConfigured()) {
      try { await deleteDoc(doc(db, "farm_boundaries", id)); } catch { /* Ignore delete error */ }
    }
    setSavedFields(prev => prev.filter(f => f.id !== id));
  }

  // ── Offline download ─────────────────────────────────────────────────────
  async function downloadCurrentView() {
    if (!map.current || downloading) return;
    setDownloadError(null);
    setDownloading(true);
    setDownloadProgress({ downloaded: 0, total: 0, percent: 0 });

    const bounds = map.current.getBounds();
    const zoom   = map.current.getZoom();
    const zLevels = Array.from({ length: 4 }, (_, i) => Math.min(zoom + i - 1, 17)).filter(z => z >= 8);
    const region  = `region-${Date.now()}`;

    const b = {
      north: bounds.getNorth(), south: bounds.getSouth(),
      east:  bounds.getEast(),  west:  bounds.getWest(),
    };

    const estimate = offlineMapService.estimateDownloadSize(b, zLevels);
    if (estimate.tileCount > 800) {
      setDownloadError(`Too many tiles (${estimate.tileCount}). Zoom in closer before downloading.`);
      setDownloading(false);
      return;
    }

    try {
      await offlineMapService.downloadRegion({
        bounds: b, zoomLevels: zLevels, region, style: mapStyle,
        onProgress: setDownloadProgress,
      });
      await loadOfflineData();
    } catch (err) {
      setDownloadError('Download failed: ' + err.message);
    } finally {
      setDownloading(false);
    }
  }

  async function deleteRegion(key) {
    await offlineMapService.deleteRegion(key);
    await loadOfflineData();
  }

  // ── Background sync ──────────────────────────────────────────────────────
  function handleSync() {
    clearTimeout(syncTimeout.current);
    syncTimeout.current = setTimeout(() => {
      loadOfflineData();
    }, 2000);
  }

  const locateMe = () => { if (userLocation && map.current) map.current.setView(userLocation, 14); };

  // ────────────────────────────────────────────────────────────────────────
  return (
    <div className="farming-map-container">

      {/* Status bar */}
      <div className={`fm-status-bar ${isOnline ? 'online' : 'offline'}`}>
        {isOnline ? <><FaWifi /> Online — collaborative mode active</> : <><FaDatabase /> Offline — cached maps active</>}
      </div>

      {/* Left panel — layers */}
      <div className="map-layers-panel">
        <h3>🗺️ Collaborative Layers</h3>

        {/* Map style */}
        <div className="fm-style-toggle">
          <button className={mapStyle === 'satellite' ? 'active' : ''} onClick={() => setMapStyle('satellite')}><FaSatellite /> Satellite</button>
          <button className={mapStyle === 'street'    ? 'active' : ''} onClick={() => setMapStyle('street')}><FaMap /> Street</button>
        </div>

        <div className="layer-toggle"><label><input type="checkbox" checked={showFarmers} onChange={e => setShowFarmers(e.target.checked)} /><FaUsers /> Nearby Farmers</label></div>
        <div className="layer-toggle"><label><input type="checkbox" checked={showAlerts}  onChange={e => setShowAlerts(e.target.checked)}   /><FaShieldAlt /> Disaster Alerts</label></div>
        <div className="layer-toggle"><label><input type="checkbox" checked={showFields}  onChange={e => setShowFields(e.target.checked)}   /><FaDrawPolygon /> Shared Boundaries</label></div>
        <div className="layer-toggle"><label><input type="checkbox" checked={showWeather} onChange={e => setShowWeather(e.target.checked)} /><FaCloud /> Weather</label></div>
        <div className="layer-toggle"><label><input type="checkbox" checked={showCrops}   onChange={e => setShowCrops(e.target.checked)}   /><FaLeaf /> Local Crop Fields</label></div>

        {/* Nearby Farmers List Summary */}
        {showFarmers && nearbyFarmers.length > 0 && (
          <div className="fm-nearby-list">
            <p className="fm-section-label">Nearby Farmers ({nearbyFarmers.length})</p>
            <div className="farmer-scroll">
              {nearbyFarmers.map(f => (
                <div key={f.uid} className="fm-farmer-item" onClick={() => map.current?.panTo([f.lat, f.lng])}>
                  <span className="farmer-dot"></span>
                  <span className="farmer-name-mini">{f.name}</span>
                  <span className="farmer-dist">{f.distanceInKm.toFixed(1)}km</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Saved fields list */}
        {savedFields.length > 0 && (
          <div className="fm-fields-list">
            <p className="fm-section-label">My Local Fields</p>
            {savedFields.map(f => (
              <div key={f.id} className="fm-field-item">
                <span onClick={() => map.current?.panTo(f.points[0])}>🟢 {f.name}</span>
                <button onClick={() => removeField(f.id)} title="Delete field"><FaTrash /></button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Top-right controls */}
      <div className="map-controls">
        <button className="map-control-btn locate-btn" onClick={locateMe} title="Locate me">
          <FaLocationArrow /> Locate Me
        </button>
        <button className={`map-control-btn track-btn ${tracking ? 'active' : ''}`} onClick={toggleTracking} title="Live GPS">
          <FaSync /> {tracking ? 'Broadcasting Location' : 'Live GPS tracking'}
        </button>
        <button className={`map-control-btn draw-btn ${drawMode ? 'active' : ''}`} onClick={() => setDrawMode(!drawMode)} title="Draw field">
          <FaDrawPolygon /> {drawMode ? 'Cancel Draw' : 'Share Farm Boundary'}
        </button>
        <button className="map-control-btn offline-btn" onClick={() => setShowOfflinePanel(!showOfflinePanel)} title="Offline maps">
          <FaDownload /> Offline Maps
        </button>
      </div>

      {/* Draw instructions */}
      {drawMode && (
        <div className="fm-draw-hint">
          <FaInfoCircle /> Mark your farm boundary to share it with nearby farmers. (min 3 points)
        </div>
      )}

      {/* Field name form */}
      {showFieldForm && (
        <div className="fm-field-form">
          <h4>Save & Share Boundary</h4>
          <input
            type="text" placeholder="e.g. West Wheat Field"
            value={fieldName} onChange={e => setFieldName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && saveField()}
            autoFocus
          />
          <div className="fm-form-btns">
            <button className="btn-save" onClick={saveField} disabled={!fieldName.trim()}><FaCheck /> Save & Share</button>
            <button className="btn-cancel" onClick={() => { setShowFieldForm(false); setDrawMode(false); }}><FaTimes /> Cancel</button>
          </div>
        </div>
      )}

      {/* Offline panel */}
      {showOfflinePanel && (
        <div className="fm-offline-panel">
          <div className="fm-offline-header">
            <h3><FaDatabase /> Offline Maps</h3>
            <button onClick={() => setShowOfflinePanel(false)}><FaTimes /></button>
          </div>

          {cacheStats && (
            <div className="fm-cache-stats">
              <div className="stat"><span>{cacheStats.tileCount}</span><label>Tiles</label></div>
              <div className="stat"><span>{cacheStats.totalMB}MB</span><label>Size</label></div>
              <div className="stat"><span>{cacheStats.regionCount}</span><label>Regions</label></div>
            </div>
          )}

          <div className="fm-download-section">
            <p className="fm-section-label">Download current view:</p>
            {!isOnline && <p className="fm-offline-warn">⚠️ Offline. Cannot download.</p>}
            {downloadError && <p className="fm-error-msg">❌ {downloadError}</p>}

            {downloading ? (
              <div className="fm-progress">
                <div className="fm-progress-bar">
                  <div className="fm-progress-fill" style={{ width: `${downloadProgress?.percent || 0}%` }} />
                </div>
                <span>{downloadProgress?.percent || 0}% • {downloadProgress?.downloaded} tiles</span>
              </div>
            ) : (
              <button className="btn-download" onClick={downloadCurrentView} disabled={!isOnline}>
                <FaDownload /> Download View
              </button>
            )}
          </div>

          {/* Cached regions */}
          {offlineRegions.length > 0 && (
            <div className="fm-regions-list">
              <p className="fm-section-label">Cached Regions</p>
              {offlineRegions.map(r => (
                <div key={r.key} className="fm-region-item">
                  <div>
                    <strong>{r.key}</strong>
                    <span>{new Date(r.timestamp).toLocaleDateString()}</span>
                  </div>
                  <button onClick={() => deleteRegion(r.key)} title="Delete"><FaTrash /></button>
                </div>
              ))}
            </div>
          )}

          <button className="btn-sync" onClick={loadOfflineData}><FaSync /> Refresh</button>
        </div>
      )}

      {/* Error banner */}
      {mapError && <div className="map-error"><FaInfoCircle /> {mapError}</div>}

      {/* The map */}
      <div ref={mapContainer} className="map-container" />

      {/* Selected marker details */}
      {selectedMarker && (
        <div className="marker-details-panel">
          <button className="close-details-btn" onClick={() => setSelectedMarker(null)}><FaTimes /></button>
          <h3>{selectedMarker.title || selectedMarker.id}</h3>
          <div className="marker-details">
            {Object.entries(selectedMarker).filter(([k]) => !['id','lat','lng','title','distanceInKm'].includes(k)).map(([k, v]) => (
              <div key={k} className="detail-item">
                <strong>{k}:</strong> <span>{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}