import React, { useState, useEffect } from "react";
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
import { auth, db } from "./lib/firebase";
import { collection, query, where, onSnapshot, addDoc, deleteDoc, doc, updateDoc } from "firebase/firestore";
import "./FarmingCalendar.css";

const ACTIVITY_TYPES = [
  { id: "sowing", label: "Sowing", icon: <Sprout size={16} />, color: "#10b981" },
  { id: "irrigation", label: "Irrigation", icon: <Droplets size={16} />, color: "#3b82f6" },
  { id: "fertilizer", label: "Fertilizer", icon: <AlertCircle size={16} />, color: "#f59e0b" },
  { id: "harvest", label: "Harvest", icon: <CheckCircle2 size={16} />, color: "#8b5cf6" },
  { id: "other", label: "Other", icon: <CalendarIcon size={16} />, color: "#6b7280" },
];

const FarmingCalendar = () => {
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [activities, setActivities] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newActivity, setNewActivity] = useState({
    title: "",
    type: "sowing",
    time: "09:00",
    description: ""
  });
  const [loading, setLoading] = useState(true);

  // Fetch activities from Firestore
  useEffect(() => {
    const user = auth.currentUser;
    if (!user) {
      setLoading(false);
      return;
    }

    const q = query(collection(db, "activities"), where("userId", "==", user.uid));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const docs = snapshot.docs.map(snap => ({
        id: snap.id,
        ...snap.data()
      }));
      setActivities(docs);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const renderHeader = () => {
    return (
      <div className="calendar-header">
        <div className="header-info">
          <h2>{format(currentMonth, "MMMM yyyy")}</h2>
          <p>Plan your agricultural activities</p>
        </div>
        <div className="header-nav">
          <button onClick={() => setCurrentMonth(subMonths(currentMonth, 1))} className="nav-btn">
            &#8249;
          </button>
          <button onClick={() => setCurrentMonth(new Date())} className="today-btn">Today</button>
          <button onClick={() => setCurrentMonth(addMonths(currentMonth, 1))} className="nav-btn">
            &#8250;
          </button>
        </div>
      </div>
    );
  };

  const renderDays = () => {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    return (
      <div className="calendar-days">
        {days.map((day, i) => (
          <div key={i} className="day-name">{day}</div>
        ))}
      </div>
    );
  };

  const renderCells = () => {
    const monthStart = startOfMonth(currentMonth);
    const monthEnd = endOfMonth(monthStart);
    const startDate = startOfWeek(monthStart);
    const endDate = endOfWeek(monthEnd);

    const dateFormat = "d";
    const rows = [];
    let days = [];
    let day = startDate;
    let formattedDate = "";

    while (day <= endDate) {
      for (let i = 0; i < 7; i++) {
        formattedDate = format(day, dateFormat);
        const cloneDay = day;
        const dayActivities = activities.filter(act => isSameDay(parseISO(act.date), cloneDay));

        days.push(
          <div
            className={`calendar-cell ${!isSameMonth(day, monthStart) ? "disabled" : ""} 
              ${isSameDay(day, selectedDate) ? "selected" : ""} 
              ${isToday(day) ? "today" : ""}`}
            key={day.toString()}
            onClick={() => setSelectedDate(cloneDay)}
          >
            <span className="cell-number">{formattedDate}</span>
            <div className="cell-indicators">
              {dayActivities.slice(0, 3).map((act, index) => (
                <div 
                  key={index} 
                  className="activity-dot" 
                  style={{ backgroundColor: ACTIVITY_TYPES.find(t => t.id === act.type)?.color }}
                />
              ))}
              {dayActivities.length > 3 && <span className="more-count">+{dayActivities.length - 3}</span>}
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
    return <div className="calendar-body">{rows}</div>;
  };

  const handleAddActivity = async (e) => {
    e.preventDefault();
    const user = auth.currentUser;
    if (!user) return;

    try {
      await addDoc(collection(db, "activities"), {
        userId: user.uid,
        title: newActivity.title,
        type: newActivity.type,
        time: newActivity.time,
        description: newActivity.description,
        date: selectedDate.toISOString(),
        completed: false,
        createdAt: new Date().toISOString()
      });
      setShowAddModal(false);
      setNewActivity({ title: "", type: "sowing", time: "09:00", description: "" });
    } catch (err) {
      console.error("Error adding activity:", err);
    }
  };

  const handleDeleteActivity = async (id) => {
    try {
      await deleteDoc(doc(db, "activities", id));
    } catch (err) {
      console.error("Error deleting activity:", err);
    }
  };

  const toggleComplete = async (activity) => {
    try {
      await updateDoc(doc(db, "activities", activity.id), {
        completed: !activity.completed
      });
    } catch (err) {
      console.error("Error toggling activity:", err);
    }
  };

  const selectedDayActivities = activities.filter(act => isSameDay(parseISO(act.date), selectedDate));

  return (
    <div className="farming-calendar-container">
      <div className="calendar-main-glass">
        <div className="calendar-col">
          {renderHeader()}
          {renderDays()}
          {renderCells()}
        </div>
        
        <div className="details-col">
          <div className="details-header">
            <h3>{format(selectedDate, "do MMMM, yyyy")}</h3>
            <button className="add-activity-btn" onClick={() => setShowAddModal(true)}>
              <Plus size={18} /> Add Activity
            </button>
          </div>

          <div className="activities-list">
            {selectedDayActivities.length === 0 ? (
              <div className="no-activities">
                <CalendarIcon size={48} className="empty-icon" />
                <p>No activities planned for this day.</p>
              </div>
            ) : (
              selectedDayActivities.map((act) => (
                <div key={act.id} className={`activity-item ${act.completed ? 'completed' : ''}`}>
                  <div className="activity-status" onClick={() => toggleComplete(act)}>
                    {act.completed ? <CheckCircle2 size={20} className="done" /> : <div className="pending-circle" />}
                  </div>
                  <div className="activity-info">
                    <div className="activity-type-badge" style={{ backgroundColor: ACTIVITY_TYPES.find(t => t.id === act.type)?.color + '20', color: ACTIVITY_TYPES.find(t => t.id === act.type)?.color }}>
                      {ACTIVITY_TYPES.find(t => t.id === act.type)?.icon}
                      {ACTIVITY_TYPES.find(t => t.id === act.type)?.label}
                    </div>
                    <h4>{act.title}</h4>
                    <div className="activity-metadata">
                      <span><Clock size={14} /> {act.time}</span>
                      {act.description && <p>{act.description}</p>}
                    </div>
                  </div>
                  <button className="delete-btn" onClick={() => handleDeleteActivity(act.id)}>
                    <Trash2 size={16} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {showAddModal && (
        <div className="modal-overlay">
          <div className="modal-card">
            <h3>Add New Activity</h3>
            <form onSubmit={handleAddActivity}>
              <div className="form-group">
                <label>Activity Title</label>
                <input 
                  type="text" 
                  placeholder="e.g. Rice Sowing" 
                  value={newActivity.title}
                  onChange={(e) => setNewActivity({...newActivity, title: e.target.value})}
                  required
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Type</label>
                  <select 
                    value={newActivity.type}
                    onChange={(e) => setNewActivity({...newActivity, type: e.target.value})}
                  >
                    {ACTIVITY_TYPES.map(t => (
                      <option key={t.id} value={t.id}>{t.label}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>Time</label>
                  <input 
                    type="time" 
                    value={newActivity.time}
                    onChange={(e) => setNewActivity({...newActivity, time: e.target.value})}
                  />
                </div>
              </div>
              <div className="form-group">
                <label>Description (Optional)</label>
                <textarea 
                  rows="3" 
                  placeholder="Details about the task..."
                  value={newActivity.description}
                  onChange={(e) => setNewActivity({...newActivity, description: e.target.value})}
                ></textarea>
              </div>
              <div className="modal-actions">
                <button type="button" onClick={() => setShowAddModal(false)} className="cancel-btn">Cancel</button>
                <button type="submit" className="submit-btn">Save Activity</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default FarmingCalendar;
