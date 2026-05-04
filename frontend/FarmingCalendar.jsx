import React, { useState, useEffect, useMemo } from "react";
import {
  format,
  addMonths,
  subMonths,
  startOfMonth,
  endOfMonth,
  startOfWeek,
  endOfWeek,
  isSameMonth,
  isSameDay,
  addDays,
  isToday,
  parseISO
} from "date-fns";

import {
  Calendar as CalendarIcon,
  Plus,
  Clock,
  Droplets,
  Sprout,
  Trash2,
  CheckCircle2,
  AlertCircle
} from "lucide-react";

import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import {
  collection,
  query,
  where,
  onSnapshot,
  addDoc,
  deleteDoc,
  doc,
  updateDoc
} from "firebase/firestore";

import "./FarmingCalendar.css";
import Loader from "./Loader";

/* ---------------- CONFIG ---------------- */

const ACTIVITY_TYPES = [
  { id: "sowing", label: "Sowing", icon: <Sprout size={16} />, color: "#10b981" },
  { id: "irrigation", label: "Irrigation", icon: <Droplets size={16} />, color: "#3b82f6" },
  { id: "fertilizer", label: "Fertilizer", icon: <AlertCircle size={16} />, color: "#f59e0b" },
  { id: "harvest", label: "Harvest", icon: <CheckCircle2 size={16} />, color: "#8b5cf6" },
  { id: "other", label: "Other", icon: <CalendarIcon size={16} />, color: "#6b7280" }
];

const getToday = () => new Date();

/* ---------------- COMPONENT ---------------- */

const FarmingCalendar = () => {
  const [currentMonth, setCurrentMonth] = useState(getToday());
  const [selectedDate, setSelectedDate] = useState(getToday());
  const [activities, setActivities] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [loading, setLoading] = useState(true);

  const [newActivity, setNewActivity] = useState({
    title: "",
    type: "sowing",
    time: "09:00",
    description: ""
  });

  /* ---------------- FIREBASE ---------------- */

  useEffect(() => {
    if (!isFirebaseConfigured()) return setLoading(false);

    const user = auth?.currentUser;
    if (!user) return setLoading(false);

    const q = query(collection(db, "activities"), where("userId", "==", user.uid));

    const unsub = onSnapshot(q, (snap) => {
      const data = snap.docs.map(d => ({
        id: d.id,
        ...d.data(),
        parsedDate: parseISO(d.data().date) // 🔥 parse once only
      }));
      setActivities(data);
      setLoading(false);
    });

    return () => unsub();
  }, []);

  /* ---------------- OPTIMIZED GROUPING ---------------- */

  const activitiesByDay = useMemo(() => {
    const map = new Map();

    activities.forEach(act => {
      const key = format(act.parsedDate, "yyyy-MM-dd");

      if (!map.has(key)) map.set(key, []);
      map.get(key).push(act);
    });

    return map;
  }, [activities]);

  /* ---------------- CALENDAR GRID ---------------- */

  const calendarCells = useMemo(() => {
    const monthStart = startOfMonth(currentMonth);
    const monthEnd = endOfMonth(monthStart);
    const startDate = startOfWeek(monthStart);
    const endDate = endOfWeek(monthEnd);

    const rows = [];
    let days = [];
    let day = startDate;

    while (day <= endDate) {
      for (let i = 0; i < 7; i++) {
        const key = format(day, "yyyy-MM-dd");
        const dayActivities = activitiesByDay.get(key) || [];

        const cloneDay = day;

        days.push(
          <div
            key={key}
            className={`calendar-cell 
              ${!isSameMonth(day, monthStart) ? "disabled" : ""} 
              ${isSameDay(day, selectedDate) ? "selected" : ""} 
              ${isToday(day) ? "today" : ""}`}
            onClick={() => setSelectedDate(cloneDay)}
          >
            <span className="cell-number">{format(day, "d")}</span>

            <div className="cell-indicators">
              {dayActivities.slice(0, 3).map((act) => {
                const type = ACTIVITY_TYPES.find(t => t.id === act.type);
                return (
                  <div
                    key={act.id}
                    className="activity-dot"
                    style={{ backgroundColor: type?.color }}
                  />
                );
              })}
              {dayActivities.length > 3 && (
                <span className="more-count">+{dayActivities.length - 3}</span>
              )}
            </div>
          </div>
        );

        day = addDays(day, 1);
      }

      rows.push(
        <div className="calendar-row" key={day.toString()}>
          {days}
        </div>
      );
      days = [];
    }

    return rows;
  }, [currentMonth, selectedDate, activitiesByDay]);

  /* ---------------- SELECTED DAY ---------------- */

  const selectedDayActivities = useMemo(() => {
    const key = format(selectedDate, "yyyy-MM-dd");
    return activitiesByDay.get(key) || [];
  }, [selectedDate, activitiesByDay]);

  /* ---------------- FIREBASE ACTIONS ---------------- */

  const handleAddActivity = async (e) => {
    e.preventDefault();
    const user = auth?.currentUser;
    if (!user) return;

    await addDoc(collection(db, "activities"), {
      userId: user.uid,
      ...newActivity,
      date: selectedDate.toISOString(),
      completed: false,
      createdAt: new Date().toISOString()
    });

    setNewActivity({
      title: "",
      type: "sowing",
      time: "09:00",
      description: ""
    });

    setShowAddModal(false);
  };

  const handleDelete = async (id) => {
    await deleteDoc(doc(db, "activities", id));
  };

  const toggleComplete = async (act) => {
    await updateDoc(doc(db, "activities", act.id), {
      completed: !act.completed
    });
  };

  /* ---------------- UI ---------------- */

  if (loading) return <Loader message="Loading schedule..." />;

  return (
    <div className="farming-calendar-container">

      {/* HEADER */}
      <div className="calendar-header">
        <h2>{format(currentMonth, "MMMM yyyy")}</h2>

        <div className="header-nav">
          <button onClick={() => setCurrentMonth(subMonths(currentMonth, 1))}>‹</button>
          <button onClick={() => setCurrentMonth(getToday())}>Today</button>
          <button onClick={() => setCurrentMonth(addMonths(currentMonth, 1))}>›</button>
        </div>
      </div>

      {/* GRID */}
      <div className="calendar-body">{calendarCells}</div>

      {/* DETAILS */}
      <div className="details-col">
        <h3>{format(selectedDate, "do MMMM yyyy")}</h3>

        {selectedDayActivities.length === 0 ? (
          <p>No activities</p>
        ) : (
          selectedDayActivities.map(act => {
            const type = ACTIVITY_TYPES.find(t => t.id === act.type);

            return (
              <div key={act.id} className="activity-item">
                <span onClick={() => toggleComplete(act)}>
                  {act.completed ? "✔" : "○"}
                </span>

                <div>
                  <b>{act.title}</b>
                  <p>{type?.label} • {act.time}</p>
                </div>

                <button onClick={() => handleDelete(act.id)}>
                  <Trash2 size={16} />
                </button>
              </div>
            );
          })
        )}

        <button onClick={() => setShowAddModal(true)}>
          <Plus /> Add
        </button>
      </div>

      {/* ADD MODAL */}
      {showAddModal && (
        <div className="modal-overlay">
          <form className="modal-card" onSubmit={handleAddActivity}>
            <input
              placeholder="Title"
              value={newActivity.title}
              onChange={(e) =>
                setNewActivity({ ...newActivity, title: e.target.value })
              }
              required
            />

            <select
              value={newActivity.type}
              onChange={(e) =>
                setNewActivity({ ...newActivity, type: e.target.value })
              }
            >
              {ACTIVITY_TYPES.map(t => (
                <option key={t.id} value={t.id}>{t.label}</option>
              ))}
            </select>

            <button type="submit">Save</button>
          </form>
        </div>
      )}
    </div>
  );
};

export default FarmingCalendar;