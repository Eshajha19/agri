import React from "react";
import "./ClaimTimeline.css";

function ClaimTimeline({ stages, currentStage }) {
  return (
    <div className="timeline-container">
      {stages.map((stage, index) => {
        const isActive = stage === currentStage;
        const isCompleted = stages.indexOf(currentStage) > index;

        return (
          <div key={stage} className={`timeline-step ${isActive ? "active" : ""} ${isCompleted ? "completed" : ""}`}>
            <div className="circle">{index + 1}</div>
            <p className="label">{stage}</p>
            {isActive && <p className="timestamp">Now</p>}
          </div>
        );
      })}
    </div>
  );
}

export default ClaimTimeline;
