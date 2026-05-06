import React, { useState, useEffect } from "react";
import * as geofire from "geofire-common";
import { db, auth, isFirebaseConfigured } from "./lib/firebase";
import { collection, addDoc, query, orderBy, startAt, endAt, Timestamp, onSnapshot } from "firebase/firestore";
import { AlertTriangle, MapPin, X, Flame, Bug, CloudLightning } from "lucide-react";

export default function GeoAlertMesh({ onClose }) {
  const [alerts, setAlerts] = useState([]);
  const [location, setLocation] = useState(null);
  const [error, setError] = useState(null);
  
  // Form state
  const [alertType, setAlertType] = useState("Pest Outbreak");
  const [severity, setSeverity] = useState("High");
  const [notes, setNotes] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // 1. Get User Location
  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
        },
        (err) => {
          setError("Location access required for geo-mesh alerts. Please enable GPS.");
        }
      );
    } else {
      setError("Geolocation is not supported by your browser.");
    }
  }, []);

  // 2. Fetch Nearby Alerts (Real-time using geohash bounds)
  useEffect(() => {
    if (!location || !isFirebaseConfigured()) return;

    // Radius in meters (5km)
    const radiusInM = 5000;
    const center = [location.lat, location.lng];
    const bounds = geofire.geohashQueryBounds(center, radiusInM);
    
    const unsubs = [];
    const localAlertsMap = new Map();

    for (const b of bounds) {
      const q = query(
        collection(db, "disaster_alerts"),
        orderBy("geohash"),
        startAt(b[0]),
        endAt(b[1])
      );

      const unsub = onSnapshot(q, (snapshot) => {
        snapshot.docChanges().forEach((change) => {
          const doc = change.doc;
          if (change.type === "removed") {
            localAlertsMap.delete(doc.id);
          } else {
            const data = doc.data();
            // Validate data coordinates
            if (typeof data.lat === 'number' && typeof data.lng === 'number') {
              const distanceInKm = geofire.distanceBetween(
                [data.lat, data.lng], 
                center
              );
              // Filter false positives from bounding box
              if (distanceInKm <= radiusInM / 1000) {
                localAlertsMap.set(doc.id, { id: doc.id, distanceInKm, ...data });
              }
            }
          }
        });
        
        // Sort by recency
        const sorted = Array.from(localAlertsMap.values())
          .sort((a, b) => b.createdAt?.toMillis() - a.createdAt?.toMillis());
        setAlerts(sorted);
      });
      unsubs.push(unsub);
    }

    return () => {
      unsubs.forEach((u) => u());
    };
  }, [location]);

  // 3. Report Event
  const handleReport = async (e) => {
    e.preventDefault();
    if (!location || !auth.currentUser || !isFirebaseConfigured()) {
      if (!auth.currentUser) setError("You must be logged in to report disasters.");
      return;
    }
    
    setIsSubmitting(true);
    try {
      const hash = geofire.geohashForLocation([location.lat, location.lng]);
      
      await addDoc(collection(db, "disaster_alerts"), {
        geohash: hash,
        lat: location.lat,
        lng: location.lng,
        type: alertType,
        severity: severity,
        notes: notes,
        reporterId: auth.currentUser.uid,
        reporterName: auth.currentUser.displayName || auth.currentUser.email?.split("@")[0] || "Farmer",
        createdAt: Timestamp.now()
      });
      
      setNotes("");
      // Success feedback can be inferred by the alert popping up instantly in the feed below
    } catch (err) {
      console.error(err);
      setError("Failed to broadcast alert. Check your connection.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const getIcon = (type) => {
    if (type.includes("Fire")) return <Flame color="#ef4444" size={24} />;
    if (type.includes("Pest")) return <Bug color="#f59e0b" size={24} />;
    return <CloudLightning color="#3b82f6" size={24} />;
  };

  return (
    <div className="geo-alerts-modal" onClick={e => e.stopPropagation()}>
      <div className="geo-alerts-header">
        <div className="header-title">
          <AlertTriangle color="#ef4444" />
          <h2>Local Disaster Mesh</h2>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close"><X /></button>
      </div>

      <div className="geo-alerts-content">
        {error && <div className="geo-error">{error}</div>}
        
        <div className="report-section">
          <h3>Report a Disaster (5km Radius Broadcast)</h3>
          <form onSubmit={handleReport} className="report-form">
            <div className="form-row">
              <select value={alertType} onChange={(e) => setAlertType(e.target.value)} className="alert-select">
                <option>Pest Outbreak</option>
                <option>Wildfire</option>
                <option>Extreme Weather</option>
              </select>
              <select value={severity} onChange={(e) => setSeverity(e.target.value)} className="severity-select">
                <option value="High">High Severity</option>
                <option value="Medium">Medium Severity</option>
              </select>
            </div>
            <input 
              type="text" 
              className="notes-input"
              placeholder="Additional details (e.g. Locust Swarm moving north)" 
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              maxLength={100}
            />
            <button type="submit" disabled={!location || isSubmitting} className="broadcast-btn">
              {isSubmitting ? "Broadcasting..." : "Broadcast Alert to Nearby Farmers"}
            </button>
          </form>
        </div>

        <div className="nearby-alerts">
          <div className="nearby-header">
            <h3>Nearby Active Alerts <span className="alert-count">{alerts.length}</span></h3>
            <span className="radius-badge">within 5 km</span>
          </div>
          
          {!location ? (
            <div className="loading-loc">
              <MapPin className="pulse" /> Acquiring precise GPS location...
            </div>
          ) : alerts.length === 0 ? (
            <div className="no-alerts">
              <ShieldCheck color="#22c55e" size={32} />
              <p>No disasters reported in your area.</p>
            </div>
          ) : (
            <div className="alert-list">
              {alerts.map(alert => (
                <div key={alert.id} className={`alert-card severity-${alert.severity.toLowerCase()}`}>
                  <div className="alert-icon-container">{getIcon(alert.type)}</div>
                  <div className="alert-info">
                    <h4>{alert.type}</h4>
                    <p className="alert-meta">
                      <MapPin size={12}/> {alert.distanceInKm.toFixed(1)} km away • Reported by {alert.reporterName}
                    </p>
                    {alert.notes && <p className="alert-notes">"{alert.notes}"</p>}
                  </div>
                  <div className="alert-actions">
                    <span className="alert-time">
                      {alert.createdAt?.toDate().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                    </span>
                    <button className="ack-btn">Acknowledge</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Dummy shield component for "no alerts" state
const ShieldCheck = ({ color, size }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    <path d="m9 12 2 2 4-4"/>
  </svg>
);
