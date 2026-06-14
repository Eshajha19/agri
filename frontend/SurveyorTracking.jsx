import React, { useState, useEffect } from "react";

function SurveyorTracking({ surveyData }) {
  const [status, setStatus] = useState(surveyData.status);

  useEffect(() => {
    // Simulate auto-updates (later connect to backend via WebSocket or polling)
    const interval = setInterval(() => {
      // fetch updated status from backend
      // setStatus(newStatus);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="surveyor-tracking">
      <h2>Surveyor Tracking Interface</h2>
      <div className="card">
        <p><strong>Surveyor:</strong> {surveyData.name}</p>
        <p><strong>Contact:</strong> {surveyData.contact}</p>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Scheduled Visit:</strong> {surveyData.visitTime}</p>
        <p><strong>ETA:</strong> {surveyData.eta}</p>
      </div>
    </div>
  );
}

export default SurveyorTracking;
