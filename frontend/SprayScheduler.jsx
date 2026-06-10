import React, { useState, useEffect } from "react";
import Calendar from "react-calendar";  
import "react-calendar/dist/Calendar.css"; 
import ScheduleCard from "./ScheduleCard";
import "./SprayScheduler.css";
import { useTranslation } from "react-i18next";

const SprayScheduler = ({ schedules = [], weatherData: _weatherData, location: _location }) => {
  const { t } = useTranslation();
  const [viewDate, setViewDate] = useState(new Date());
  const [filter, setFilter] = useState("");

  const normalizeSchedule = (item) => {
    if (!item) return null;
    const dateSource = item.date || item.scheduledAt || item.createdAt;
    const scheduleDate = dateSource ? new Date(dateSource) : null;
    const today = new Date();
    const dateLabel = scheduleDate
      ? scheduleDate.toLocaleDateString()
      : "No date";

    let status = item.status;
    if (!status) {
      if (!scheduleDate) status = "upcoming";
      else if (scheduleDate.toDateString() === today.toDateString()) status = "today";
      else if (scheduleDate < today) status = "overdue";
      else status = "upcoming";
    }

    return {
      id: item.id,
      crop: item.crop || item.cropName || "",
      pest: item.sprayType || item.type || "",
      product: item.product || item.type || "",
      date: dateLabel,
      status,
      createdAt: item.createdAt || dateLabel,
    };
  };

  const loadPersistedSchedules = () => {
    try {
      const raw = localStorage.getItem("agri_spray_schedules");
      if (!raw) return [];
      const data = JSON.parse(raw);
      return Array.isArray(data)
        ? data.map(normalizeSchedule).filter(Boolean)
        : [];
    } catch (_err) {
      return [];
    }
  };

  const [savedSchedules, setSavedSchedules] = useState(() => loadPersistedSchedules());

  useEffect(() => {
    setSavedSchedules(loadPersistedSchedules());
  }, []);

  const merged = (schedules && schedules.length ? schedules : savedSchedules)
    .map(normalizeSchedule)
    .filter(Boolean);

  return (
    <div className="spray-scheduler">
      <h2>{t("sprayScheduler.title")}</h2>

      <Calendar
        onChange={setViewDate}
        value={viewDate}
        className="calendar-view"
        aria-label={t("sprayScheduler.calendarAria")}
      />

      <div className="filters">
        <input
          type="text"
          placeholder={t("sprayScheduler.searchPlaceholder")}
          aria-label={t("sprayScheduler.searchAria")}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      <div className="summary">
        <div className="summary-card" role="status" aria-label={t("sprayScheduler.totalAria")}>
            {t("sprayScheduler.total")}: {merged.length}
        </div>
        <div className="summary-card" role="status" aria-label={t("sprayScheduler.upcomingAria")}>
          {t("sprayScheduler.upcoming")}: {merged.filter((s) => s.status === "upcoming").length}
        </div>
        <div className="summary-card" role="status" aria-label={t("sprayScheduler.overdueAria")}>
          {t("sprayScheduler.overdue")}: {merged.filter((s) => s.status === "overdue").length}
        </div>
        <div className="summary-card" role="status" aria-label={t("sprayScheduler.completedAria")}>
         {t("sprayScheduler.completed")}: {merged.filter((s) => s.status === "completed").length}
        </div>
      </div>

      <div className="schedule-list">
        {(() => {
          const term = filter.toLowerCase().trim();
          const filtered = term
            ? merged.filter((s) => {
                const haystack = `${s.crop} ${s.pest} ${s.product} ${s.status}`.toLowerCase();
                return haystack.includes(term);
              })
            : merged;

          if (!filtered.length) {
            return (
              <div className="empty-state">
                <p>No spray schedules yet.</p>
              </div>
            );
          }

          try {
  return filtered
    .slice()
    .sort((a, b) => String(a.date).localeCompare(String(b.date)))
    .map((schedule, idx) => (
      <ScheduleCard
        key={schedule.id || idx}
        schedule={schedule}
        onSelect={(s) => console.log("Selected", s)}
      />
    ));
} catch (error) {
  reportErrorToBackend({
    error,
    context: "SprayScheduler rendering schedules",
    timestamp: new Date().toISOString(),
    severity: "medium"
  });
  return (
    <div className="error-state">
      <p>Something went wrong while loading schedules.</p>
    </div>
  );
}
        })()}
      </div>
    </div>
  );
};

import React, { useState, useEffect } from "react";
import Calendar from "react-calendar";  
import "react-calendar/dist/Calendar.css"; 
import ScheduleCard from "./ScheduleCard";
import "./SprayScheduler.css";

const SprayScheduler = ({ schedules = [], weatherData: _weatherData, location: _location }) => {
  const [viewDate, setViewDate] = useState(new Date());
  const [filter, setFilter] = useState("");

  const normalizeSchedule = (item) => {
    if (!item) return null;
    const dateSource = item.date || item.scheduledAt || item.createdAt;
    const scheduleDate = dateSource ? new Date(dateSource) : null;
    const today = new Date();
    const dateLabel = scheduleDate
      ? scheduleDate.toLocaleDateString()
      : "No date";

    let status = item.status;
    if (!status) {
      if (!scheduleDate) status = "upcoming";
      else if (scheduleDate.toDateString() === today.toDateString()) status = "today";
      else if (scheduleDate < today) status = "overdue";
      else status = "upcoming";
    }

    return {
      id: item.id,
      crop: item.crop || item.cropName || "",
      pest: item.sprayType || item.type || "",
      product: item.product || item.type || "",
      date: dateLabel,
      status,
      createdAt: item.createdAt || dateLabel,
    };
  };

  const loadPersistedSchedules = () => {
    try {
      const raw = localStorage.getItem("agri_spray_schedules");
      if (!raw) return [];
      const data = JSON.parse(raw);
      return Array.isArray(data)
        ? data.map(normalizeSchedule).filter(Boolean)
        : [];
    } catch (_err) {
      return [];
    }
  };

  const [savedSchedules, setSavedSchedules] = useState(() => loadPersistedSchedules());

  useEffect(() => {
    setSavedSchedules(loadPersistedSchedules());
  }, []);

  const merged = (schedules && schedules.length ? schedules : savedSchedules)
    .map(normalizeSchedule)
    .filter(Boolean);

  return (
    <div className="spray-scheduler">
      <h2>Spray Scheduler</h2>

      <Calendar
        onChange={setViewDate}
        value={viewDate}
        className="calendar-view"
      />

      <div className="filters">
        <input
          type="text"
          placeholder="Search by crop or pest..."
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      <div className="summary">
        <div className="summary-card">Total: {merged.length}</div>
        <div className="summary-card">
          Upcoming: {merged.filter((s) => s.status === "upcoming").length}
        </div>
        <div className="summary-card">
          Overdue: {merged.filter((s) => s.status === "overdue").length}
        </div>
        <div className="summary-card">
          Completed: {merged.filter((s) => s.status === "completed").length}
        </div>
      </div>

      <div className="schedule-list">
        {(() => {
          const term = filter.toLowerCase().trim();
          const filtered = term
            ? merged.filter((s) => {
                const haystack = `${s.crop} ${s.pest} ${s.product} ${s.status}`.toLowerCase();
                return haystack.includes(term);
              })
            : merged;

          if (!filtered.length) {
            return (
              <div className="empty-state">
                <p>No spray schedules yet.</p>
              </div>
            );
          }

          return filtered
            .slice()
            .sort((a, b) => String(a.date).localeCompare(String(b.date)))
            .map((schedule, idx) => (
              <ScheduleCard key={schedule.id || idx} schedule={schedule} />
            ));
        })()}
      </div>
    </div>
  );
};

export default SprayScheduler;