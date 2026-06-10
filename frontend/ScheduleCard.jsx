import React from "react";
import "./ScheduleCard.css";

const ScheduleCard = ({ schedule, onSelect }) => {
  const handleSelect = () => {
    if (onSelect) onSelect(schedule);
  };

  return (
    <div
      className={`schedule-card status-${schedule.status}`}
      role="button"
      tabIndex={0}
      aria-label={`Schedule for ${schedule.crop}, product ${schedule.product}, date ${schedule.date}, status ${schedule.status}`}
      onClick={handleSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleSelect();
        }
      }}
    >
      <h3>{schedule.crop}</h3>
      <p>Disease/Pest: {schedule.pest}</p>
      <p>Product: {schedule.product}</p>
      <p>Date: {schedule.date}</p>
      <span className={`badge ${schedule.status}`}>{schedule.status}</span>
    </div>
  );
};


export default ScheduleCard;