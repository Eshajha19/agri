import React, { useState } from "react";
import Calendar from "react-calendar";  
import "react-calendar/dist/Calendar.css"; 
import ScheduleCard from "./ScheduleCard";
import "./SprayScheduler.css";

const SprayScheduler = ({ schedules }) => {
  const [viewDate, setViewDate] = useState(new Date());
  const [filter, setFilter] = useState("");

  return (
    <div className="spray-scheduler">
      <h2>Spray Scheduler</h2>

      {/* Calendar View */}
      <Calendar
        onChange={setViewDate}
        value={viewDate}
        className="calendar-view"
      />

      {/* Filters */}
      <div className="filters">
        <input
          type="text"
          placeholder="Search by crop or pest..."
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      {/* Dashboard Summary */}
      <div className="summary">
        <div className="summary-card">Total: {schedules.length}</div>
        <div className="summary-card">
          Upcoming: {schedules.filter(s => s.status === "upcoming").length}
        </div>
        <div className="summary-card">
          Overdue: {schedules.filter(s => s.status === "overdue").length}
        </div>
        <div className="summary-card">
          Completed: {schedules.filter(s => s.status === "completed").length}
        </div>
      </div>

      {/* Schedule Cards */}
      <div className="schedule-list">
        {schedules
          .filter(s =>
            s.crop.toLowerCase().includes(filter.toLowerCase()) ||
            s.pest.toLowerCase().includes(filter.toLowerCase())
          )
          .map((schedule, idx) => (
            <ScheduleCard key={idx} schedule={schedule} />
          ))}
      </div>
    </div>
  );
};

export default SprayScheduler;
