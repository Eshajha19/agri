import React, { useState, useMemo, useEffect } from "react";
import "./AgriMarketplace.css";
import { Search, MapPin, Plus, Calendar, Clock, Phone, Info, X, Heart } from "lucide-react";

// ---------------- ICONS ----------------
const TYPE_ICONS = {
  Tractor: "🚜",
  Harvester: "🌾",
  Drone: "🚁",
  Tillage: "⚙️",
  Sowing: "🌱",
};

const TYPES = ["All", "Tractor", "Harvester", "Drone", "Tillage", "Sowing"];

// ---------------- DATA ----------------
const INITIAL_EQUIPMENT = [
  { id: 1, name: "John Deere Tractor 5050D", type: "Tractor", price: 800, priceUnit: "hr", location: "Karnal", distance: 5, rating: 4.8, owner: "Suresh Kumar", phone: "9999999991", available: true },
  { id: 2, name: "Mahindra Rice Harvester", type: "Harvester", price: 2500, priceUnit: "day", location: "Ludhiana", distance: 12, rating: 4.5, owner: "Hardeep Singh", phone: "9999999992", available: true },
  { id: 3, name: "DJI Agras T40 Drone", type: "Drone", price: 1500, priceUnit: "hr", location: "Bhopal", distance: 8, rating: 4.9, owner: "TechAgri", phone: "9999999993", available: false },
  { id: 4, name: "Sonalika Rotavator 200", type: "Tillage", price: 400, priceUnit: "hr", location: "Nagpur", distance: 3, rating: 4.2, owner: "Ramesh Patil", phone: "9999999994", available: true },
  { id: 5, name: "Kubota MU5501 Tractor", type: "Tractor", price: 900, priceUnit: "hr", location: "Pune", distance: 4, rating: 4.7, owner: "Santosh Shinde", phone: "9999999995", available: true },
];

// ---------------- HELPER ----------------
const calculateTotal = (price, duration) => price * Number(duration || 0);

// ---------------- COMPONENT ----------------
export default function AgriMarketplace() {

  // ---------------- STATES ----------------
  const [equipment, setEquipment] = useState(INITIAL_EQUIPMENT);
  const [searchQuery, setSearchQuery] = useState("");
  const [locationQuery, setLocationQuery] = useState("");
  const [selectedType, setSelectedType] = useState("All");
  const [favorites, setFavorites] = useState([]);

  const [showListModal, setShowListModal] = useState(false);
  const [showBookingModal, setShowBookingModal] = useState(null);

  const [bookingDate, setBookingDate] = useState("");
  const [bookingTime, setBookingTime] = useState("");
  const [bookingDuration, setBookingDuration] = useState("");
  const [bookingError, setBookingError] = useState("");

  const [newListing, setNewListing] = useState({
    name: "",
    type: "Tractor",
    price: "",
    priceUnit: "hr",
    location: ""
  });

  const [loadingLocation, setLoadingLocation] = useState(true);

  // ---------------- GEO LOCATION ----------------
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(() => {
      setLoadingLocation(false);
    }, () => setLoadingLocation(false));
  }, []);

  // ---------------- FILTER ----------------
  const filteredEquipment = useMemo(() => {
    return equipment
      .filter(item => {
        return (
          (selectedType === "All" || item.type === selectedType) &&
          item.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
          item.location.toLowerCase().includes(locationQuery.toLowerCase())
        );
      })
      .sort((a, b) => a.distance - b.distance);
  }, [equipment, searchQuery, locationQuery, selectedType]);

  const selectedItem = useMemo(() => equipment.find(e => e.id === showBookingModal), [showBookingModal, equipment]);

  const minPrice = useMemo(() => Math.min(...equipment.map(e => e.price)), [equipment]);

  // ---------------- FAVORITES ----------------
  const toggleFavorite = (id) => {
    setFavorites(prev => prev.includes(id) ? prev.filter(f => f !== id) : [...prev, id]);
  };

  // ---------------- LIST ----------------
  const handleListEquipment = (e) => {
    e.preventDefault();

    const newItem = {
      ...newListing,
      id: Date.now(),
      distance: Math.floor(Math.random() * 10),
      rating: 5,
      owner: "You",
      available: true
    };

    setEquipment(prev => [newItem, ...prev]);
    setShowListModal(false);
  };

  // ---------------- BOOKING ----------------
  const handleBooking = () => {
    if (!bookingDate) return setBookingError("Select date");
    if (!bookingTime) return setBookingError("Select time");
    if (!bookingDuration || bookingDuration < 1) return setBookingError("Invalid duration");

    alert(`Booked ${selectedItem.name} for ₹${calculateTotal(selectedItem.price, bookingDuration)}`);

    setShowBookingModal(null);
    setBookingDate("");
    setBookingTime("");
    setBookingDuration("");
    setBookingError("");
  };

  // ---------------- CALL ----------------
  const handleCall = (phone) => {
    alert(`Calling ${phone}`);
  };

  // ---------------- UI ----------------
  return (
    <div className="marketplace-container">

      {/* HEADER */}
      <div className="header">
        <h1>🚜 Smart Agri Marketplace</h1>
        {loadingLocation && <p>Detecting location...</p>}

        <button className="list-btn" onClick={() => setShowListModal(true)}>
          <Plus size={16}/> List Equipment
        </button>
      </div>

      {/* SEARCH */}
      <div className="search-bar">
        <Search size={16}/>
        <input placeholder="Search equipment..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />

        <MapPin size={16}/>
        <input placeholder="Location..." value={locationQuery} onChange={(e) => setLocationQuery(e.target.value)} />
      </div>

      {/* FILTERS */}
      <div className="filters">
        {TYPES.map(type => (
          <button key={type} onClick={() => setSelectedType(type)} className={selectedType === type ? "active" : ""}>{type}</button>
        ))}
      </div>

      {/* GRID */}
      <div className="grid">
        {filteredEquipment.map(item => (
          <div key={item.id} className={`card ${!item.available ? "disabled" : ""}`}>

            <div className="card-top">
              <div className="icon">{TYPE_ICONS[item.type]}</div>

              <button onClick={() => toggleFavorite(item.id)}>
                <Heart color={favorites.includes(item.id) ? "red" : "gray"} />
              </button>
            </div>

            <h3>{item.name}</h3>
            <p>{item.location} ({item.distance} km)</p>
            <p>⭐ {item.rating}</p>
            <p>Owner: {item.owner}</p>

            {item.price === minPrice && <div className="best">🔥 Best Deal</div>}

            <div className="price">₹{item.price}/{item.priceUnit}</div>

            <button disabled={!item.available} onClick={() => setShowBookingModal(item.id)}>
              {item.available ? "Book Now" : "Unavailable"}
            </button>

            <button onClick={() => handleCall(item.phone)} className="call">
              <Phone size={14}/> Call
            </button>
          </div>
        ))}
      </div>

      {/* LIST MODAL */}
      {showListModal && (
        <div className="modal" onClick={() => setShowListModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setShowListModal(false)}><X/></button>
            <h2>Add Equipment</h2>

            <form onSubmit={handleListEquipment}>
              <input placeholder="Name" required onChange={(e)=>setNewListing({...newListing,name:e.target.value})}/>
              <select onChange={(e)=>setNewListing({...newListing,type:e.target.value})}>
                {TYPES.slice(1).map(t => <option key={t}>{t}</option>)}
              </select>
              <input type="number" placeholder="Price" required onChange={(e)=>setNewListing({...newListing,price:e.target.value})}/>
              <input placeholder="Location" required onChange={(e)=>setNewListing({...newListing,location:e.target.value})}/>
              <button type="submit">Submit</button>
            </form>
          </div>
        </div>
      )}

      {/* BOOKING MODAL */}
      {showBookingModal && selectedItem && (
        <div className="modal" onClick={() => setShowBookingModal(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>

            <button onClick={() => setShowBookingModal(null)}><X/></button>

            <h2>{selectedItem.name}</h2>

            <input type="date" onChange={(e)=>setBookingDate(e.target.value)} />
            <input type="time" onChange={(e)=>setBookingTime(e.target.value)} />
            <input type="number" placeholder="Duration" onChange={(e)=>setBookingDuration(e.target.value)} />

            {bookingError && <p className="error">{bookingError}</p>}

            <p>Total: ₹{calculateTotal(selectedItem.price, bookingDuration)}</p>

            <button onClick={handleBooking}>Confirm Booking</button>

          </div>
        </div>
      )}

    </div>
  );
}
