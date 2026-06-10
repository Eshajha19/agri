import React, { useState, useEffect } from "react";
import Calendar from "react-calendar";  
import "react-calendar/dist/Calendar.css"; 
import ScheduleCard from "./ScheduleCard";
import "./SprayScheduler.css";
import { db, auth } from "./firebase";
import { collection, getDocs, addDoc } from "firebase/firestore";

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

  const loadPersistedSchedules = async () => {
  const userId = auth.currentUser?.uid;
  if (!userId) return [];
  const snapshot = await getDocs(collection(db, "users", userId, "schedules"));
  return snapshot.docs.map(d => ({ id: d.id, ...d.data() }));
};

const [savedSchedules, setSavedSchedules] = useState([]);

useEffect(() => {
  loadPersistedSchedules().then(setSavedSchedules);
}, []);

// Example: add new schedule
const handleAddSchedule = async (schedule) => {
  const userId = auth.currentUser?.uid;
  if (!userId) return;
  await addDoc(collection(db, "users", userId, "schedules"), schedule);
  const updated = await loadPersistedSchedules();
  setSavedSchedules(updated);
};

// Example: delete schedule
const handleDeleteSchedule = async (id) => {
  const userId = auth.currentUser?.uid;
  if (!userId) return;
  await deleteDoc(doc(db, "users", userId, "schedules", id));
  const updated = await loadPersistedSchedules();
  setSavedSchedules(updated);
};

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