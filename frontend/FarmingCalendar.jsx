import React, { useEffect, useMemo, useRef, useState } from "react";
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
  parseISO,
} from "date-fns";
import {
  Calendar as CalendarIcon,
  Plus,
  Clock,
  Droplets,
  Sprout,
  Trash2,
  CheckCircle2,
  AlertCircle,
  PencilLine,
  Download,
  MessageSquareText,
  Sparkles,
} from "lucide-react";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import { collection, query, where, onSnapshot, addDoc, deleteDoc, doc, updateDoc } from "firebase/firestore";
import "./FarmingCalendar.css";
import Loader from "./Loader";

const ACTIVITY_TYPES = [
  { id: "sowing", label: "Sowing", icon: <Sprout size={16} />, color: "#10b981" },
  { id: "irrigation", label: "Irrigation", icon: <Droplets size={16} />, color: "#3b82f6" },
  { id: "spraying", label: "Spraying", icon: <AlertCircle size={16} />, color: "#f59e0b" },
  { id: "harvest", label: "Harvest", icon: <CheckCircle2 size={16} />, color: "#8b5cf6" },
  { id: "other", label: "Other", icon: <CalendarIcon size={16} />, color: "#6b7280" },
];

const CROP_REMINDER_TEMPLATES = {
  rice: {
    label: "Rice",
    timeline: [
      { daysAfter: 0, type: "sowing", title: "Rice sowing and nursery prep", time: "07:00", description: "Prepare the field, seed bed, and transplant plan." },
      { daysAfter: 18, type: "irrigation", title: "Maintain standing water", time: "06:30", description: "Keep shallow water and watch for weed pressure." },
      { daysAfter: 35, type: "spraying", title: "Scout for stem borer and blast", time: "07:00", description: "Inspect leaves and apply crop-safe spray only if symptoms appear." },
      { daysAfter: 60, type: "irrigation", title: "Critical panicle-stage irrigation", time: "06:30", description: "Avoid moisture stress during panicle initiation." },
      { daysAfter: 120, type: "harvest", title: "Rice harvest window", time: "08:00", description: "Start harvesting when grain moisture and maturity align." },
    ],
  },
  wheat: {
    label: "Wheat",
    timeline: [
      { daysAfter: 0, type: "sowing", title: "Wheat sowing", time: "07:30", description: "Use a uniform sowing depth and good seed placement." },
      { daysAfter: 21, type: "irrigation", title: "CRI irrigation", time: "06:30", description: "First critical irrigation at crown root initiation." },
      { daysAfter: 45, type: "spraying", title: "Rust and aphid scouting", time: "07:00", description: "Check lower leaves and spray only if pest pressure builds." },
      { daysAfter: 75, type: "irrigation", title: "Tillering irrigation", time: "06:30", description: "Keep the crop active through tillering and stem elongation." },
      { daysAfter: 150, type: "harvest", title: "Wheat harvest", time: "08:00", description: "Schedule harvest once grains are physiologically mature." },
    ],
  },
  maize: {
    label: "Maize",
    timeline: [
      { daysAfter: 0, type: "sowing", title: "Maize sowing", time: "08:00", description: "Plant after soil moisture is ready and field is leveled." },
      { daysAfter: 18, type: "irrigation", title: "Knee-high irrigation", time: "06:30", description: "Avoid moisture stress during vegetative growth." },
      { daysAfter: 35, type: "spraying", title: "Fall armyworm monitoring", time: "07:00", description: "Inspect whorls and treat early if damage appears." },
      { daysAfter: 55, type: "irrigation", title: "Tasseling irrigation", time: "06:30", description: "Critical water window for pollination success." },
      { daysAfter: 90, type: "harvest", title: "Maize harvest", time: "08:30", description: "Harvest once cobs dry and kernel fill is complete." },
    ],
  },
  cotton: {
    label: "Cotton",
    timeline: [
      { daysAfter: 0, type: "sowing", title: "Cotton sowing", time: "08:00", description: "Use a clean seed line and avoid waterlogging at emergence." },
      { daysAfter: 28, type: "irrigation", title: "First irrigation", time: "06:30", description: "Maintain steady moisture during early crop establishment." },
      { daysAfter: 45, type: "spraying", title: "Sucking pest scouting", time: "07:00", description: "Look for whitefly, jassid, and early bollworm pressure." },
      { daysAfter: 90, type: "irrigation", title: "Flowering irrigation", time: "06:30", description: "Avoid stress while squares and flowers are forming." },
      { daysAfter: 180, type: "harvest", title: "Cotton picking window", time: "08:00", description: "Pick open bolls in dry weather and store separately." },
    ],
  },
  soybean: {
    label: "Soybean",
    timeline: [
      { daysAfter: 0, type: "sowing", title: "Soybean sowing", time: "08:00", description: "Plant in a moist seedbed and keep spacing even." },
      { daysAfter: 15, type: "irrigation", title: "Emergence irrigation check", time: "06:30", description: "Support germination if rainfall is delayed." },
      { daysAfter: 30, type: "spraying", title: "Aphid and pod borer scouting", time: "07:00", description: "Scout from the underside of leaves and act early." },
      { daysAfter: 45, type: "irrigation", title: "Flowering irrigation", time: "06:30", description: "Keep moisture balanced at flowering and pod set." },
      { daysAfter: 100, type: "harvest", title: "Soybean harvest", time: "08:30", description: "Harvest when pods turn mature and shattering risk is low." },
    ],
  },
  mustard: {
    label: "Mustard",
    timeline: [
      { daysAfter: 0, type: "sowing", title: "Mustard sowing", time: "08:00", description: "Sow in a fine seedbed with good moisture." },
      { daysAfter: 25, type: "irrigation", title: "Light irrigation check", time: "06:30", description: "Keep the crop moving through early establishment." },
      { daysAfter: 45, type: "spraying", title: "Aphid and blight scouting", time: "07:00", description: "Check buds and leaves for pest pressure." },
      { daysAfter: 70, type: "irrigation", title: "Flowering irrigation", time: "06:30", description: "Keep moisture steady at the flowering stage." },
      { daysAfter: 120, type: "harvest", title: "Mustard harvest", time: "08:00", description: "Harvest once siliquae dry and seeds are ready." },
    ],
  },
  chickpea: {
    label: "Chickpea",
    timeline: [
      { daysAfter: 0, type: "sowing", title: "Chickpea sowing", time: "08:00", description: "Plant with seed treatment and a clean seedbed." },
      { daysAfter: 30, type: "irrigation", title: "Flowering irrigation", time: "06:30", description: "Light irrigation only if soil moisture is low." },
      { daysAfter: 55, type: "spraying", title: "Pod borer scouting", time: "07:00", description: "Scout flowers and pods for early insect damage." },
      { daysAfter: 80, type: "irrigation", title: "Pod fill irrigation", time: "06:30", description: "Avoid excess water while pods are filling." },
      { daysAfter: 110, type: "harvest", title: "Chickpea harvest", time: "08:30", description: "Harvest when pods dry and plants start yellowing." },
    ],
  },
  watermelon: {
    label: "Watermelon",
    timeline: [
      { daysAfter: 0, type: "sowing", title: "Watermelon sowing", time: "08:00", description: "Use raised beds or mounds and keep soil warm." },
      { daysAfter: 12, type: "irrigation", title: "Vine establishment irrigation", time: "06:30", description: "Light frequent watering supports early vine growth." },
      { daysAfter: 25, type: "spraying", title: "Mildew and fruit fly scouting", time: "07:00", description: "Check leaves and fruit set for early issues." },
      { daysAfter: 40, type: "irrigation", title: "Fruit set irrigation", time: "06:30", description: "Maintain steady moisture while fruits are forming." },
      { daysAfter: 90, type: "harvest", title: "Watermelon harvest", time: "08:00", description: "Harvest at full size and field maturity." },
    ],
  },
};

const DEFAULT_CROP_TYPE = "wheat";

const createEmptyActivity = (selectedDate) => ({
  title: "",
  type: "sowing",
  date: format(selectedDate, "yyyy-MM-dd"),
  time: "09:00",
  description: "",
});

const createEmptyGenerator = (userData) => ({
  cropType: (userData?.cropType || DEFAULT_CROP_TYPE).toLowerCase(),
  plantingDate: format(new Date(), "yyyy-MM-dd"),
  phoneNumber: userData?.phoneNumber || "",
});

const sortActivities = (items) => [...items].sort((left, right) => {
  const leftDate = `${left.date || ""}T${left.time || "00:00"}`;
  const rightDate = `${right.date || ""}T${right.time || "00:00"}`;
  return leftDate.localeCompare(rightDate);
});

const getActivityStorageKey = (userId) => `agri:farming-calendar:${userId || "guest"}`;

const readLocalActivities = (userId) => {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(getActivityStorageKey(userId));
    return raw ? sortActivities(JSON.parse(raw)) : [];
  } catch (error) {
    console.warn("Unable to read local farming calendar cache:", error);
    return [];
  }
};

const saveLocalActivities = (userId, items) => {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(getActivityStorageKey(userId), JSON.stringify(sortActivities(items)));
  } catch (error) {
    console.warn("Unable to persist local farming calendar cache:", error);
  }
};

const createIcsTimestamp = (date, timeString) => {
  const [hours, minutes] = timeString.split(":").map(Number);
  const localDate = new Date(date);
  localDate.setHours(hours, minutes, 0, 0);
  return localDate.toISOString().replace(/[-:]/g, "").split(".")[0] + "Z";
};

const buildIcsContent = (items) => {
  const lines = [
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "PRODID:-//Fasal Saathi//Smart Crop Reminder Automation//EN",
    "CALSCALE:GREGORIAN",
  ];

  items.forEach((item, index) => {
    const start = createIcsTimestamp(parseISO(item.date), item.time || "09:00");
    const endDate = new Date(parseISO(item.date));
    const [hours, minutes] = (item.time || "09:00").split(":").map(Number);
    endDate.setHours(hours, minutes + 45, 0, 0);
    const end = endDate.toISOString().replace(/[-:]/g, "").split(".")[0] + "Z";

    lines.push(
      "BEGIN:VEVENT",
      `UID:${item.id || `agri-${index}`}@fasalsaathi`,
      `DTSTAMP:${createIcsTimestamp(new Date(), "00:00")}`,
      `DTSTART:${start}`,
      `DTEND:${end}`,
      `SUMMARY:${item.title}`,
      `DESCRIPTION:${(item.description || "").replace(/\n/g, " ")}`,
      `CATEGORIES:${item.type || "other"}`,
      "END:VEVENT",
    );
  });

  lines.push("END:VCALENDAR");
  return lines.join("\r\n");
};

const buildSmsDraft = (items, cropLabel, phoneNumber) => {
  const header = `Smart crop reminders for ${cropLabel}${phoneNumber ? ` (${phoneNumber})` : ""}`;
  const body = items.slice(0, 8).map((item) => {
    const dateLabel = format(parseISO(item.date), "dd MMM");
    return `${dateLabel} ${item.time || "09:00"} - ${item.title}`;
  }).join(" | ");

  return `${header}\n${body}`.trim();
};

const buildGeneratedActivities = (cropType, plantingDate, userId) => {
  const template = CROP_REMINDER_TEMPLATES[cropType] || CROP_REMINDER_TEMPLATES[DEFAULT_CROP_TYPE];
  const baseDate = new Date(`${plantingDate}T00:00:00`);

  return template.timeline.map((entry) => ({
    id: typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    userId,
    title: entry.title,
    type: entry.type,
    time: entry.time,
    description: entry.description,
    date: addDays(baseDate, entry.daysAfter).toISOString(),
    completed: false,
    createdAt: new Date().toISOString(),
    source: "auto",
    cropType: template.label,
  }));
};

const FarmingCalendar = ({ userData }) => {
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [activities, setActivities] = useState([]);
  const [showActivityModal, setShowActivityModal] = useState(false);
  const [editingActivity, setEditingActivity] = useState(null);
  const [activityForm, setActivityForm] = useState(() => createEmptyActivity(new Date()));
  const [generatorForm, setGeneratorForm] = useState(() => createEmptyGenerator(userData));
  const [statusMessage, setStatusMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const statusTimerRef = useRef(null);

  const effectiveUserId = userData?.uid || auth?.currentUser?.uid || userData?.id || "";
  const canUseFirestore = isFirebaseConfigured() && Boolean(effectiveUserId);

  useEffect(() => {
    setGeneratorForm((current) => ({
      ...current,
      cropType: (userData?.cropType || current.cropType || DEFAULT_CROP_TYPE).toLowerCase(),
      phoneNumber: userData?.phoneNumber || current.phoneNumber || "",
    }));
  }, [userData?.cropType, userData?.phoneNumber]);

  useEffect(() => {
    if (!canUseFirestore) {
      setActivities(readLocalActivities(effectiveUserId));
      setLoading(false);
      return undefined;
    }

    const q = query(collection(db, "activities"), where("userId", "==", effectiveUserId));
    const unsubscribe = onSnapshot(
      q,
      (snapshot) => {
        const docs = snapshot.docs.map((snap) => ({
          id: snap.id,
          ...snap.data(),
        }));
        const nextActivities = sortActivities(docs);
        setActivities(nextActivities);
        saveLocalActivities(effectiveUserId, nextActivities);
        setLoading(false);
      },
      (error) => {
        console.error("Error loading calendar activities:", error);
        setActivities(readLocalActivities(effectiveUserId));
        setLoading(false);
      }
    );

    return () => unsubscribe();
  }, [canUseFirestore, effectiveUserId]);

  useEffect(() => {
    return () => {
      if (statusTimerRef.current) {
        window.clearTimeout(statusTimerRef.current);
      }
    };
  }, []);

  const selectedDayActivities = useMemo(() => {
    return sortActivities(
      activities.filter((activity) => {
        if (!activity.date) {
          return false;
        }
        return isSameDay(parseISO(activity.date), selectedDate);
      })
    );
  }, [activities, selectedDate]);

  const reminderSummary = useMemo(() => {
    const nextTasks = sortActivities(activities)
      .filter((item) => !item.completed)
      .slice(0, 4);

    return {
      total: activities.length,
      nextTasks,
    };
  }, [activities]);

  const activityStore = canUseFirestore ? "cloud-synced" : "saved on this device";

  const notify = (message) => {
    setStatusMessage(message);
    if (statusTimerRef.current) {
      window.clearTimeout(statusTimerRef.current);
    }
    statusTimerRef.current = window.setTimeout(() => setStatusMessage(""), 3500);
  };

  const resetActivityForm = (date = selectedDate) => {
    setEditingActivity(null);
    setActivityForm(createEmptyActivity(date));
  };

  const openNewActivityModal = () => {
    resetActivityForm(selectedDate);
    setShowActivityModal(true);
  };

  const openEditActivityModal = (activity) => {
    setEditingActivity(activity);
    setActivityForm({
      title: activity.title || "",
      type: activity.type || "sowing",
      date: activity.date ? format(parseISO(activity.date), "yyyy-MM-dd") : format(selectedDate, "yyyy-MM-dd"),
      time: activity.time || "09:00",
      description: activity.description || "",
    });
    setShowActivityModal(true);
  };

  const persistActivities = (nextActivities) => {
    const sorted = sortActivities(nextActivities);
    setActivities(sorted);
    saveLocalActivities(effectiveUserId, sorted);
  };

  const renderHeader = () => {
    return (
      <div className="calendar-header">
        <div className="header-info">
          <h2>{format(currentMonth, "MMMM yyyy")}</h2>
          <p>Plan your agricultural activities and reminder exports</p>
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
        {days.map((day, index) => (
          <div key={index} className="day-name">{day}</div>
        ))}
      </div>
    );
  };

  const renderCells = () => {
    const monthStart = startOfMonth(currentMonth);
    const monthEnd = endOfMonth(monthStart);
    const startDate = startOfWeek(monthStart);
    const endDate = endOfWeek(monthEnd);

    const rows = [];
    let days = [];
    let day = startDate;

    while (day <= endDate) {
      for (let index = 0; index < 7; index += 1) {
        const formattedDate = format(day, "d");
        const cloneDay = day;
        const dayActivities = activities.filter((activity) => activity.date && isSameDay(parseISO(activity.date), cloneDay));

        days.push(
          <div
            className={`calendar-cell ${!isSameMonth(day, monthStart) ? "disabled" : ""} ${isSameDay(day, selectedDate) ? "selected" : ""} ${isToday(day) ? "today" : ""}`}
            key={day.toString()}
            onClick={() => setSelectedDate(cloneDay)}
          >
            <span className="cell-number">{formattedDate}</span>
            <div className="cell-indicators">
              {dayActivities.slice(0, 3).map((activity, activityIndex) => (
                <div
                  key={activityIndex}
                  className="activity-dot"
                  style={{ backgroundColor: ACTIVITY_TYPES.find((item) => item.id === activity.type)?.color }}
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

  const handleSaveActivity = async (event) => {
    event.preventDefault();

    const payload = {
      userId: effectiveUserId,
      title: activityForm.title.trim(),
      type: activityForm.type,
      time: activityForm.time,
      description: activityForm.description.trim(),
      date: new Date(`${activityForm.date}T00:00:00`).toISOString(),
      completed: editingActivity?.completed || false,
      createdAt: editingActivity?.createdAt || new Date().toISOString(),
      source: editingActivity?.source || "manual",
      cropType: editingActivity?.cropType || generatorForm.cropType || "",
    };

    try {
      if (canUseFirestore) {
        if (editingActivity?.id) {
          await updateDoc(doc(db, "activities", editingActivity.id), payload);
          persistActivities(activities.map((item) => (item.id === editingActivity.id ? { ...item, ...payload } : item)));
        } else {
          const docRef = await addDoc(collection(db, "activities"), payload);
          persistActivities([...activities, { id: docRef.id, ...payload }]);
        }
      } else if (editingActivity?.id) {
        persistActivities(activities.map((item) => (item.id === editingActivity.id ? { ...item, ...payload } : item)));
      } else {
        persistActivities([
          ...activities,
          {
            id: typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
            ...payload,
          },
        ]);
      }

      setShowActivityModal(false);
      setEditingActivity(null);
      setActivityForm(createEmptyActivity(parseISO(payload.date)));
      notify(editingActivity ? "Activity updated." : "Activity saved.");
    } catch (error) {
      console.error("Error saving activity:", error);
      notify("Unable to save the activity.");
    }
  };

  const handleDeleteActivity = async (activity) => {
    try {
      if (canUseFirestore && activity.id) {
        await deleteDoc(doc(db, "activities", activity.id));
      }

      persistActivities(activities.filter((item) => item.id !== activity.id));
      notify("Activity removed.");
    } catch (error) {
      console.error("Error deleting activity:", error);
      notify("Unable to delete the activity.");
    }
  };

  const toggleComplete = async (activity) => {
    try {
      const nextCompleted = !activity.completed;
      if (canUseFirestore && activity.id) {
        await updateDoc(doc(db, "activities", activity.id), {
          completed: nextCompleted,
        });
      }

      persistActivities(activities.map((item) => (item.id === activity.id ? { ...item, completed: nextCompleted } : item)));
    } catch (error) {
      console.error("Error toggling activity:", error);
      notify("Unable to update the activity status.");
    }
  };

  const handleGenerateSchedule = async () => {
    if (!generatorForm.cropType || !generatorForm.plantingDate) {
      notify("Choose a crop and planting date first.");
      return;
    }

    const template = CROP_REMINDER_TEMPLATES[generatorForm.cropType] || CROP_REMINDER_TEMPLATES[DEFAULT_CROP_TYPE];
    const generatedActivities = buildGeneratedActivities(generatorForm.cropType, generatorForm.plantingDate, effectiveUserId);

    try {
      if (canUseFirestore) {
        await Promise.all(generatedActivities.map((activity) => addDoc(collection(db, "activities"), activity)));
      }

      const merged = sortActivities([...activities, ...generatedActivities]);
      persistActivities(merged);
      setSelectedDate(new Date(`${generatorForm.plantingDate}T00:00:00`));
      setCurrentMonth(new Date(`${generatorForm.plantingDate}T00:00:00`));
      notify(`Generated ${generatedActivities.length} ${template.label} reminders.`);
    } catch (error) {
      console.error("Error generating crop schedule:", error);
      notify("Unable to generate the crop reminder schedule.");
    }
  };

  const handleExportCalendar = () => {
    if (activities.length === 0) {
      notify("Add or generate reminders before exporting.");
      return;
    }

    const blob = new Blob([buildIcsContent(sortActivities(activities))], { type: "text/calendar;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `fasal-saathi-reminders-${format(new Date(), "yyyy-MM-dd")}.ics`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    notify("Calendar export downloaded.");
  };

  const handleSmsSync = async () => {
    if (activities.length === 0) {
      notify("Add or generate reminders before preparing SMS.");
      return;
    }

    const cropLabel = CROP_REMINDER_TEMPLATES[generatorForm.cropType]?.label || generatorForm.cropType || "crop";
    const smsDraft = buildSmsDraft(sortActivities(activities), cropLabel, generatorForm.phoneNumber);

    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(smsDraft);
      notify("SMS draft copied. Paste it into your messaging app.");
      return;
    }

    const smsLink = `sms:${generatorForm.phoneNumber ? generatorForm.phoneNumber : ""}?body=${encodeURIComponent(smsDraft)}`;
    window.location.href = smsLink;
    notify("Opened the SMS composer.");
  };

  const templatePreview = CROP_REMINDER_TEMPLATES[generatorForm.cropType] || CROP_REMINDER_TEMPLATES[DEFAULT_CROP_TYPE];

  return (
    <div className="farming-calendar-container">
      {loading ? (
        <Loader message="Loading your farming schedule..." />
      ) : (
        <>
          <div className="calendar-main-glass">
            <div className="calendar-col">
              {renderHeader()}
              {renderDays()}
              {renderCells()}
            </div>

            <div className="details-col">
              <div className="details-header">
                <div>
                  <h3>{format(selectedDate, "do MMMM, yyyy")}</h3>
                  <p className="details-subtitle">{reminderSummary.total} reminders - {activityStore}</p>
                </div>
                <button className="add-activity-btn" onClick={openNewActivityModal} type="button">
                  <Plus size={18} /> Add Activity
                </button>
              </div>

              <div className="generator-panel">
                <div className="generator-heading">
                  <Sparkles size={18} /> Smart Crop Reminder Automation
                </div>
                <p className="generator-copy">
                  Auto-generate sowing, irrigation, spraying, and harvest tasks from a crop type, then export the plan to your calendar or SMS draft.
                </p>
                <div className="generator-grid">
                  <div className="form-group">
                    <label>Crop type</label>
                    <select
                      value={generatorForm.cropType}
                      onChange={(e) => setGeneratorForm({ ...generatorForm, cropType: e.target.value })}
                    >
                      {Object.entries(CROP_REMINDER_TEMPLATES).map(([key, item]) => (
                        <option key={key} value={key}>{item.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Planting date</label>
                    <input
                      type="date"
                      value={generatorForm.plantingDate}
                      onChange={(e) => setGeneratorForm({ ...generatorForm, plantingDate: e.target.value })}
                    />
                  </div>
                  <div className="form-group">
                    <label>SMS number</label>
                    <input
                      type="tel"
                      placeholder="Optional mobile number"
                      value={generatorForm.phoneNumber}
                      onChange={(e) => setGeneratorForm({ ...generatorForm, phoneNumber: e.target.value })}
                    />
                  </div>
                </div>
                <div className="generator-actions">
                  <button className="submit-btn" onClick={handleGenerateSchedule} type="button">
                    Generate Schedule
                  </button>
                  <button className="secondary-btn" onClick={handleExportCalendar} type="button">
                    <Download size={16} /> Export ICS
                  </button>
                  <button className="secondary-btn" onClick={handleSmsSync} type="button">
                    <MessageSquareText size={16} /> SMS Draft
                  </button>
                </div>
                <div className="generator-preview">
                  <strong>{templatePreview.label}</strong>
                  <span>{templatePreview.timeline.length} planned reminders per cycle</span>
                </div>
                {statusMessage && <p className="status-message">{statusMessage}</p>}
              </div>

              <div className="activities-list">
                {selectedDayActivities.length === 0 ? (
                  <div className="no-activities">
                    <CalendarIcon size={48} className="empty-icon" />
                    <p>No activities planned for this day.</p>
                    <span>Use the generator above to create a crop-aware reminder plan.</span>
                  </div>
                ) : (
                  selectedDayActivities.map((activity) => {
                    const activityType = ACTIVITY_TYPES.find((item) => item.id === activity.type) || ACTIVITY_TYPES[4];

                    return (
                      <div key={activity.id} className={`activity-item ${activity.completed ? "completed" : ""}`}>
                        <div className="activity-status" onClick={() => toggleComplete(activity)}>
                          {activity.completed ? <CheckCircle2 size={20} className="done" /> : <div className="pending-circle" />}
                        </div>
                        <div className="activity-info">
                          <div className="activity-type-badge" style={{ backgroundColor: `${activityType.color}20`, color: activityType.color }}>
                            {activityType.icon}
                            {activityType.label}
                          </div>
                          <h4>{activity.title}</h4>
                          {activity.cropType && <p className="activity-crop-tag">{activity.cropType}</p>}
                          <div className="activity-metadata">
                            <span><Clock size={14} /> {activity.time}</span>
                            {activity.description && <p>{activity.description}</p>}
                          </div>
                        </div>
                        <button className="edit-btn" onClick={() => openEditActivityModal(activity)} type="button" aria-label={`Edit ${activity.title}`}>
                          <PencilLine size={16} />
                        </button>
                        <button className="delete-btn" onClick={() => handleDeleteActivity(activity)} type="button">
                          <Trash2 size={16} />
                        </button>
                      </div>
                    );
                  })
                )}
              </div>

              {reminderSummary.nextTasks.length > 0 && (
                <div className="reminder-footer">
                  <p className="reminder-footer-title">Next reminders</p>
                  <ul>
                    {reminderSummary.nextTasks.map((task) => (
                      <li key={task.id || `${task.title}-${task.date}`}>
                        {format(parseISO(task.date), "dd MMM")} - {task.title}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {showActivityModal && (
            <div className="modal-overlay">
              <div className="modal-card">
                <h3>{editingActivity ? "Edit Activity" : "Add New Activity"}</h3>
                <form onSubmit={handleSaveActivity}>
                  <div className="form-group">
                    <label>Activity Title</label>
                    <input
                      type="text"
                      placeholder="e.g. Rice Sowing"
                      value={activityForm.title}
                      onChange={(e) => setActivityForm({ ...activityForm, title: e.target.value })}
                      required
                    />
                  </div>
                  <div className="form-row">
                    <div className="form-group">
                      <label>Type</label>
                      <select
                        value={activityForm.type}
                        onChange={(e) => setActivityForm({ ...activityForm, type: e.target.value })}
                      >
                        {ACTIVITY_TYPES.map((item) => (
                          <option key={item.id} value={item.id}>{item.label}</option>
                        ))}
                      </select>
                    </div>
                    <div className="form-group">
                      <label>Time</label>
                      <input
                        type="time"
                        value={activityForm.time}
                        onChange={(e) => setActivityForm({ ...activityForm, time: e.target.value })}
                      />
                    </div>
                  </div>
                  <div className="form-group">
                    <label>Date</label>
                    <input
                      type="date"
                      value={activityForm.date}
                      onChange={(e) => setActivityForm({ ...activityForm, date: e.target.value })}
                    />
                  </div>
                  <div className="form-group">
                    <label>Description (Optional)</label>
                    <textarea
                      rows="3"
                      placeholder="Details about the task..."
                      value={activityForm.description}
                      onChange={(e) => setActivityForm({ ...activityForm, description: e.target.value })}
                    />
                  </div>
                  <div className="modal-actions">
                    <button type="button" onClick={() => setShowActivityModal(false)} className="cancel-btn">Cancel</button>
                    <button type="submit" className="submit-btn">{editingActivity ? "Update Activity" : "Save Activity"}</button>
                  </div>
                </form>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default FarmingCalendar;