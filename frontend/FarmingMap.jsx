import React, { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  FaMapMarkerAlt, FaCloud, FaLeaf, FaExclamationTriangle,
  FaTimes, FaLocationArrow, FaDownload, FaWifi, FaDatabase,
  FaTrash, FaDrawPolygon, FaCheck, FaSatellite, FaMap,
  FaInfoCircle, FaSync,
} from 'react-icons/fa';
import offlineMapService from './services/offlineMapService';
import './FarmingMap.css';

// ── Icons ────────────────────────────────────────────────────────────────────
const mkIcon = (emoji, cls) =>
  L.divIcon({ html: `<div class="fm-icon">${emoji}</div>`, iconSize: [32, 32], className: cls });

const ICONS = {
  weather: () => mkIcon('🌤️', 'weather-marker'),
  crop:    () => mkIcon('🌾', 'crop-marker'),
  user:    () => mkIcon('📍', 'user-marker'),
  alert:   () => mkIcon('⚠️', 'alert-marker'),
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

// ── Component ─────────────────────────────────────────────────────────────────
export default function FarmingMap() {
  const mapContainer = useRef(null);
  const map          = useRef(null);
  const markersRef   = useRef({});
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
  const [showWeather, setShowWeather] = useState(true);
  const [showCrops,   setShowCrops]   = useState(true);
  const [showAlerts,  setShowAlerts]  = useState(true);
  const [showFields,  setShowFields]  = useState(true);

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
      // Ignore tile layer errors; they're non-critical
    }
  }

  useEffect(() => {
    if (map.current) addTileLayer(map.current, mapStyle);
  }, [mapStyle]);

  // ── GPS location ─────────────────────────────────────────────────────────
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

  // ── Continuous GPS tracking ──────────────────────────────────────────────
  const toggleTracking = useCallback(() => {
    if (tracking) {
      if (watchIdRef.current) navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
      setTracking(false);
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
  }, [tracking]);

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

  // ── Crop markers ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current || !userLocation) return;
    (markersRef.current.crops || []).forEach(m => map.current.removeLayer(m));
    if (!showCrops) return;
    const pts = [
      { lat: userLocation[0]-0.02, lng: userLocation[1]-0.02, title: 'Paddy Field A', crop: 'Paddy', area: '5 acres', status: 'Good' },
      { lat: userLocation[0]+0.02, lng: userLocation[1]+0.02, title: 'Wheat Field B', crop: 'Wheat', area: '3 acres', status: 'Good' },
      { lat: userLocation[0]+0.01, lng: userLocation[1]-0.03, title: 'Vegetable Plot C', crop: 'Vegetables', area: '2 acres', status: 'Needs Attention' },
    ];
    markersRef.current.crops = pts.map(p =>
      L.marker([p.lat, p.lng], { icon: ICONS.crop() }).addTo(map.current)
        .bindPopup(`<div class="map-popup crop-popup"><strong>${p.title}</strong><p>🌾 ${p.crop}</p><p>📍 ${p.area}</p><p>✅ ${p.status}</p></div>`)
        .on('click', () => setSelectedMarker(p))
    );
  }, [showCrops, userLocation]);

  // ── Alert markers ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current || !userLocation) return;
    (markersRef.current.alerts || []).forEach(m => map.current.removeLayer(m));
    if (!showAlerts) return;
    const pts = [
      { lat: userLocation[0]-0.03, lng: userLocation[1]+0.03, title: 'Heavy Rain Alert', severity: 'High', message: 'Heavy rainfall expected in 2 hrs' },
    ];
    markersRef.current.alerts = pts.map(p =>
      L.marker([p.lat, p.lng], { icon: ICONS.alert() }).addTo(map.current)
        .bindPopup(`<div class="map-popup alert-popup"><strong>⚠️ ${p.title}</strong><p>Severity: ${p.severity}</p><p>${p.message}</p></div>`)
        .on('click', () => setSelectedMarker(p))
    );
  }, [showAlerts, userLocation]);

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
    setSavedFields(prev => [...prev, field]);
    setFieldName('');
    setDrawMode(false);
    setShowFieldForm(false);
  }

  async function removeField(id) {
    await offlineMapService.deleteField(id);
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
        {isOnline ? <><FaWifi /> Online — live data active</> : <><FaDatabase /> Offline — cached maps active</>}
      </div>

      {/* Left panel — layers */}
      <div className="map-layers-panel">
        <h3>🗺️ Map Layers</h3>

        {/* Map style */}
        <div className="fm-style-toggle">
          <button className={mapStyle === 'satellite' ? 'active' : ''} onClick={() => setMapStyle('satellite')}><FaSatellite /> Satellite</button>
          <button className={mapStyle === 'street'    ? 'active' : ''} onClick={() => setMapStyle('street')}><FaMap /> Street</button>
        </div>

        <div className="layer-toggle"><label><input type="checkbox" checked={showWeather} onChange={e => setShowWeather(e.target.checked)} /><FaCloud /> Weather</label></div>
        <div className="layer-toggle"><label><input type="checkbox" checked={showCrops}   onChange={e => setShowCrops(e.target.checked)}   /><FaLeaf /> Crop Fields</label></div>
        <div className="layer-toggle"><label><input type="checkbox" checked={showAlerts}  onChange={e => setShowAlerts(e.target.checked)}   /><FaExclamationTriangle /> Alerts</label></div>
        <div className="layer-toggle"><label><input type="checkbox" checked={showFields}  onChange={e => setShowFields(e.target.checked)}   /><FaDrawPolygon /> My Fields</label></div>

        {/* Saved fields list */}
        {savedFields.length > 0 && (
          <div className="fm-fields-list">
            <p className="fm-section-label">Saved Fields</p>
            {savedFields.map(f => (
              <div key={f.id} className="fm-field-item">
                <span>🟢 {f.name}</span>
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
          <FaSync /> {tracking ? 'Stop GPS' : 'Live GPS'}
        </button>
        <button className={`map-control-btn draw-btn ${drawMode ? 'active' : ''}`} onClick={() => setDrawMode(!drawMode)} title="Draw field">
          <FaDrawPolygon /> {drawMode ? 'Cancel Draw' : 'Draw Field'}
        </button>
        <button className="map-control-btn offline-btn" onClick={() => setShowOfflinePanel(!showOfflinePanel)} title="Offline maps">
          <FaDownload /> Offline Maps
        </button>
      </div>

      {/* Draw instructions */}
      {drawMode && (
        <div className="fm-draw-hint">
          <FaInfoCircle /> Click on the map to mark field boundaries (min 3 points), then save.
        </div>
      )}

      {/* Field name form */}
      {showFieldForm && (
        <div className="fm-field-form">
          <h4>Save Field Boundary</h4>
          <input
            type="text" placeholder="Field name (e.g. North Paddy Field)"
            value={fieldName} onChange={e => setFieldName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && saveField()}
            autoFocus
          />
          <div className="fm-form-btns">
            <button className="btn-save" onClick={saveField} disabled={!fieldName.trim()}><FaCheck /> Save</button>
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
              <div className="stat"><span>{cacheStats.tileCount}</span><label>Cached Tiles</label></div>
              <div className="stat"><span>{cacheStats.totalMB} MB</span><label>Storage Used</label></div>
              <div className="stat"><span>{cacheStats.regionCount}</span><label>Regions</label></div>
            </div>
          )}

          <div className="fm-download-section">
            <p className="fm-section-label">Download current map view for offline use:</p>
            {!isOnline && <p className="fm-offline-warn">⚠️ You are offline. Cannot download new maps.</p>}
            {downloadError && <p className="fm-error-msg">❌ {downloadError}</p>}

            {downloading ? (
              <div className="fm-progress">
                <div className="fm-progress-bar">
                  <div className="fm-progress-fill" style={{ width: `${downloadProgress?.percent || 0}%` }} />
                </div>
                <span>{downloadProgress?.percent || 0}% — {downloadProgress?.downloaded}/{downloadProgress?.total} tiles</span>
              </div>
            ) : (
              <button className="btn-download" onClick={downloadCurrentView} disabled={!isOnline}>
                <FaDownload /> Download Current View
              </button>
            )}
          </div>

          {/* Cached regions */}
          {offlineRegions.length > 0 && (
            <div className="fm-regions-list">
              <p className="fm-section-label">Downloaded Regions</p>
              {offlineRegions.map(r => (
                <div key={r.key} className="fm-region-item">
                  <div>
                    <strong>{r.key}</strong>
                    <span>{r.downloaded}/{r.total} tiles • {new Date(r.timestamp).toLocaleDateString()}</span>
                    <span className={`fm-region-status ${r.status}`}>{r.status}</span>
                  </div>
                  <button onClick={() => deleteRegion(r.key)} title="Delete region"><FaTrash /></button>
                </div>
              ))}
            </div>
          )}

          <button className="btn-sync" onClick={loadOfflineData}><FaSync /> Refresh Stats</button>
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
            {Object.entries(selectedMarker).filter(([k]) => !['id','lat','lng','title'].includes(k)).map(([k, v]) => (
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
