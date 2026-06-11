import React from "react";
import "./ScheduleCard.css";

const ScheduleCard = ({ schedule }) => {
  return (
    <div className={`schedule-card status-${schedule.status}`}>
      <h3>{schedule.crop}</h3>
      <p>Disease/Pest: {schedule.pest}</p>
      <p>Product: {schedule.product}</p>
      <p>Date: {schedule.date}</p>
      <span className={`badge ${schedule.status}`}>{schedule.status}</span>
    </div>
  );
};

export default ScheduleCard;
