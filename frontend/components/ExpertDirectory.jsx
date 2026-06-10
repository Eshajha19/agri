import React, { useState } from "react";
import { useAdvisorStore } from "../stores/advisorStore";
import { db, auth } from "../lib/firebase";
import "./ExpertDirectory.css";
import { addDoc, collection } from "firebase/firestore";
import { toast } from "react-toastify";
import {
  X,
  Search,
  Filter,
  Calendar,
  Clock,
  User,
  MapPin,
  Phone,
  Video,
  PhoneCall,
  Star,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  XCircle,
  Building2,
} from "lucide-react";

const SPECIALIZATIONS = [
  { id: "crop_disease", label: "Crop Disease", icon: "🌾" },
  { id: "fertilizers", label: "Fertilizers", icon: "🧪" },
  { id: "irrigation", label: "Irrigation", icon: "💧" },
  { id: "pest_management", label: "Pest Management", icon: "🐛" },
  { id: "soil_health", label: "Soil Health", icon: "🪨" },
  { id: "market_advisory", label: "Market Advisory", icon: "📊" },
  { id: "kvk", label: "KVK Expert", icon: "🏢" },
  { id: "general", label: "General Agriculture", icon: "🌱" },
];

const MOCK_EXPERTS = [
  {
    id: "exp1",
    name: "Dr. Ramesh Kumar",
    specialization: "crop_disease",
    qualification: "Ph.D. in Plant Pathology",
    location: "Madhya Pradesh",
    phone: "+91 9876543210",
    rating: 4.8,
    experience: 15,
    isKvK: true,
    kvkName: "KVK Jabalpur",
    bio: "Specialist in crop disease diagnosis and organic treatment methods.",
    avatar: "https://randomuser.me/api/portraits/men/32.jpg",
  },
  {
    id: "exp2",
    name: "Dr. Priya Sharma",
    specialization: "fertilizers",
    qualification: "M.Sc. Agricultural Chemistry",
    location: "Maharashtra",
    phone: "+91 9876543211",
    rating: 4.9,
    experience: 12,
    isKvK: true,
    kvkName: "KVK Pune",
    bio: "Expert in nano-fertilizers and sustainable nutrient management.",
    avatar: "https://randomuser.me/api/portraits/women/44.jpg",
  },
  {
    id: "exp3",
    name: "Er. Suresh Patil",
    specialization: "irrigation",
    qualification: "B.Tech Agricultural Engineering",
    location: "Karnataka",
    phone: "+91 9876543212",
    rating: 4.7,
    experience: 10,
    isKvK: false,
    bio: "Drip irrigation and water management specialist.",
    avatar: "https://randomuser.me/api/portraits/men/45.jpg",
  },
  {
    id: "exp4",
    name: "Dr. Anjali Verma",
    specialization: "pest_management",
    qualification: "Ph.D. Entomology",
    location: "Uttar Pradesh",
    phone: "+91 9876543213",
    rating: 4.6,
    experience: 8,
    isKvK: true,
    kvkName: "KVK Lucknow",
    bio: "Integrated pest management and organic pest control expert.",
    avatar: "https://randomuser.me/api/portraits/women/65.jpg",
  },
  {
    id: "exp5",
    name: "Dr. Mahendra Singh",
    specialization: "soil_health",
    qualification: "Ph.D. Soil Science",
    location: "Rajasthan",
    phone: "+91 9876543214",
    rating: 4.9,
    experience: 20,
    isKvK: true,
    kvkName: "KVK Jaipur",
    bio: "Soil health assessment and reclamation specialist.",
    avatar: "https://randomuser.me/api/portraits/men/67.jpg",
  },
  {
    id: "exp6",
    name: "Dr. Kavita Desai",
    specialization: "market_advisory",
    qualification: "MBA Agriculture Business",
    location: "Gujarat",
    phone: "+91 9876543215",
    rating: 4.8,
    experience: 14,
    isKvK: false,
    bio: "Market intelligence and price forecasting expert.",
    avatar: "https://randomuser.me/api/portraits/women/28.jpg",
  },
];

function generateTimeSlots() {
  const slots = [];
  for (let hour = 9; hour < 18; hour++) {
    slots.push({
      time: `${hour.toString().padStart(2, "0")}:00`,
      display: `${hour}:00 ${hour < 12 ? "AM" : "PM"}`,
      available: Math.random() > 0.3,
    });
    slots.push({
      time: `${hour.toString().padStart(2, "0")}:30`,
      display: `${hour}:30 ${hour < 12 ? "AM" : "PM"}`,
      available: Math.random() > 0.3,
    });
  }
  return slots;
}

function ExpertDirectory({ userData, onClose, onBookConsultation }) {
  const { setSelectedExpert } = useAdvisorStore();
  const [experts] = useState(MOCK_EXPERTS);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSpecialization, setSelectedSpecialization] = useState(null);
  const [showOnlyKvK, setShowOnlyKvK] = useState(false);
  const [selectedExpert, setLocalSelectedExpert] = useState(null);
  const [showCalendar, setShowCalendar] = useState(false);
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [bookingNotes, setBookingNotes] = useState("");
  const [timeSlots, setTimeSlots] = useState([]);
  const [bookingLoading, setBookingLoading] = useState(false);

  const filteredExperts = experts.filter((expert) => {
    const matchesSearch =
      expert.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      expert.location.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSpecialization =
      !selectedSpecialization || expert.specialization === selectedSpecialization;
    const matchesKvK = !showOnlyKvK || expert.isKvK;
    return matchesSearch && matchesSpecialization && matchesKvK;
  });

  const getDaysInMonth = (date) => {
    const year = date.getFullYear();
    const month = date.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const days = [];

    for (let i = 0; i < firstDay.getDay(); i++) {
      days.push(null);
    }

    for (let i = 1; i <= lastDay.getDate(); i++) {
      const dayDate = new Date(year, month, i);
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      days.push({
        date: i,
        fullDate: dayDate,
        isPast: dayDate < today,
        isToday: dayDate.getTime() === today.getTime(),
      });
    }

    return days;
  };

  const days = getDaysInMonth(currentDate);

  const handlePrevMonth = () => {
    setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
  };

  const handleNextMonth = () => {
    setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));
  };

  const handleDateSelect = (day) => {
    if (!day || day.isPast) return;
    setTimeSlots(generateTimeSlots());
    setSelectedSlot(null);
  };

  const handleBookSlot = async () => {
    if (!selectedSlot || !selectedExpert) {
      toast.error("Please select a time slot");
      return;
    }

    // Require authentication before writing to Firestore.
    // The Firestore rule enforces userId == request.auth.uid, so writing
    // userId: "anonymous" would be rejected server-side anyway — but we
    // catch it here to give the user a clear, actionable error message
    // instead of a silent failure.
    const currentUser = auth?.currentUser;
    if (!currentUser) {
      toast.error("Please sign in to book a consultation.");
      return;
    }

    setBookingLoading(true);

    try {
      const consultationData = {
        userId: currentUser.uid,
        userName: currentUser.displayName || userData?.displayName || "Farmer",
        date: currentDate.toISOString().split("T")[0],
        time: selectedSlot.time,
        notes: bookingNotes,
        status: "scheduled",
        createdAt: new Date().toISOString(),
        type: "video",
      };

      await addDoc(collection(db, "consultations"), consultationData);

      toast.success("Consultation booked successfully!");
      setShowCalendar(false);
      setSelectedExpert(null);
      setSelectedSlot(null);
      setBookingNotes("");
      onBookConsultation?.(consultationData);
    } catch (error) {
      console.error("Booking error:", error);
      toast.error("Failed to book consultation");
    } finally {
      setBookingLoading(false);
    }
  };

  const handleExpertSelect = (expert) => {
    setLocalSelectedExpert(expert);
    setSelectedExpert(expert);
    setShowCalendar(true);
  };

  if (showCalendar && selectedExpert) {
    return (
      <div className="expert-directory-overlay" onClick={() => setShowCalendar(false)}>
        <div className="expert-booking-calendar" onClick={(e) => e.stopPropagation()}>
          <button className="close-btn" onClick={() => setShowCalendar(false)}>
            <X size={20} />
          </button>

          <div className="booking-header">
            <div className="expert-info-mini">
              <img src={selectedExpert.avatar} alt={selectedExpert.name} className="expert-avatar-small" />
              <div>
                <h3>{selectedExpert.name}</h3>
                <p>{SPECIALIZATIONS.find((s) => s.id === selectedExpert.specialization)?.label}</p>
              </div>
            </div>
          </div>

          <div className="calendar-navigation">
            <button onClick={handlePrevMonth} className="nav-btn">
              <ChevronLeft size={20} />
            </button>
            <h3>
              {currentDate.toLocaleDateString("en-US", { month: "long", year: "numeric" })}
            </h3>
            <button onClick={handleNextMonth} className="nav-btn">
              <ChevronRight size={20} />
            </button>
          </div>

          <div className="calendar-grid">
            {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((day) => (
              <div key={day} className="calendar-day-name">
                {day}
              </div>
            ))}
            {days.map((day, index) => (
              <div
                key={index}
                className={`calendar-day ${day ? (day.isPast ? "past" : day.isToday ? "today" : "") : "empty"}`}
                onClick={() => handleDateSelect(day)}
              >
                {day?.date}
              </div>
            ))}
          </div>

          {timeSlots.length > 0 && (
            <div className="time-slots-section">
              <h4>Available Time Slots</h4>
              <div className="time-slots-grid">
                {timeSlots.map((slot, index) => (
                  <button
                    key={index}
                    className={`time-slot ${selectedSlot?.time === slot.time ? "selected" : ""} ${!slot.available ? "unavailable" : ""}`}
                    onClick={() => slot.available && setSelectedSlot(slot)}
                    disabled={!slot.available}
                  >
                    {slot.display}
                  </button>
                ))}
              </div>
            </div>
          )}

          {selectedSlot && (
            <div className="booking-form-section">
              <h4>Booking Details</h4>
              <textarea
                placeholder="Describe your issue or question..."
                value={bookingNotes}
                onChange={(e) => setBookingNotes(e.target.value)}
                rows={3}
              />
              <div className="consultation-type">
                <label>Consultation Type:</label>
                <div className="type-options">
                  <button className="type-btn active">
                    <Video size={16} /> Video Call
                  </button>
                  <button className="type-btn">
                    <PhoneCall size={16} /> Audio Call
                  </button>
                </div>
              </div>
              <button
                className="book-btn"
                onClick={handleBookSlot}
                disabled={bookingLoading}
              >
                {bookingLoading ? "Booking..." : "Confirm Booking"}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="expert-directory">
        <div className="directory-header">
          <h2>
            <User className="header-icon" /> Expert/KVK Directory
          </h2>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className="directory-filters">
          <div className="search-box">
            <Search size={18} />
            <input
              type="text"
              placeholder="Search by name or location..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <Filter size={18} />
            <select
              value={selectedSpecialization || ""}
              onChange={(e) => setSelectedSpecialization(e.target.value || null)}
            >
              <option value="">All Specializations</option>
              {SPECIALIZATIONS.map((spec) => (
                <option key={spec.id} value={spec.id}>
                  {spec.icon} {spec.label}
                </option>
              ))}
            </select>
          </div>

          <label className="kvk-toggle">
            <input
              type="checkbox"
              checked={showOnlyKvK}
              onChange={(e) => setShowOnlyKvK(e.target.checked)}
            />
            <span>KVK Experts Only</span>
          </label>
        </div>

        <div className="experts-list">
          {filteredExperts.length === 0 ? (
            <div className="no-results">
              <User size={48} />
              <p>No experts found matching your criteria</p>
            </div>
          ) : (
            filteredExperts.map((expert) => (
              <div key={expert.id} className="expert-card">
                <div className="expert-avatar-section">
                  <img src={expert.avatar} alt={expert.name} className="expert-avatar" />
                  {expert.isKvK && <span className="kvk-badge">KVK</span>}
                </div>

                <div className="expert-details">
                  <h3>{expert.name}</h3>
                  <p className="qualification">{expert.qualification}</p>
                  <div className="expert-meta">
                    <span className="meta-item">
                      <MapPin size={14} /> {expert.location}
                    </span>
                    {expert.kvkName && (
                      <span className="meta-item">
                        <Building2 size={14} /> {expert.kvkName}
                      </span>
                    )}
                    <span className="meta-item">
                      <Star size={14} /> {expert.rating} ({expert.experience} yrs)
                    </span>
                  </div>
                  <p className="bio">{expert.bio}</p>
                  <div className="specialization-tag">
                    {SPECIALIZATIONS.find((s) => s.id === expert.specialization)?.icon}{" "}
                    {SPECIALIZATIONS.find((s) => s.id === expert.specialization)?.label}
                  </div>
                </div>

                <div className="expert-actions">
                  <button
                    className="book-consultation-btn"
                    onClick={() => handleExpertSelect(expert)}
                  >
                    <Calendar size={16} /> Book Consultation
                  </button>
                  {/* Phone numbers are not exposed directly — contact is
                      handled through the consultation booking flow only.
                      Direct tel: links would bypass the booking system and
                      expose expert contact details to all authenticated users. */}
                  <button
                    className="call-btn"
                    onClick={() => handleExpertSelect(expert)}
                    title="Schedule a call via the booking calendar"
                  >
                    <Phone size={16} /> Schedule Call
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
    </div>
  );
}

export default ExpertDirectory;