import React, { useState } from "react";

function EmergencyReport() {
  const [event, setEvent] = useState("");
  const [photo, setPhoto] = useState(null);
  const [location, setLocation] = useState(null);
  const [submitted, setSubmitted] = useState(false);

  const events = ["Flood", "Hailstorm", "Cyclone", "Pest Outbreak", "Drought"];

  const detectLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setLocation({
            lat: pos.coords.latitude,
            lon: pos.coords.longitude,
          });
        },
        (err) => console.error("Location error:", err)
      );
    }
  };

  const handleSubmit = () => {
    const report = {
      event,
      photo,
      location,
      timestamp: new Date().toISOString(),
    };
    console.log("Emergency Report:", report);
    setSubmitted(true);
  };

  return (
    <div className="emergency-report">
      <h2>One‑Tap Emergency Crop Loss Reporting</h2>

      {!submitted ? (
        <>
          <select onChange={(e) => setEvent(e.target.value)}>
            <option value="">Select Event</option>
            {events.map(ev => <option key={ev} value={ev}>{ev}</option>)}
          </select>

          <input
            type="file"
            accept="image/*"
            onChange={(e) => setPhoto(e.target.files[0])}
          />

          <button onClick={detectLocation}>Detect Location</button>
          {location && <p>Lat: {location.lat}, Lon: {location.lon}</p>}

          <button onClick={handleSubmit} disabled={!event || !photo || !location}>
            Submit Quick Report
          </button>
        </>
      ) : (
        <p>✅ Report submitted successfully!</p>
      )}
    </div>
  );
}

export default EmergencyReport;
