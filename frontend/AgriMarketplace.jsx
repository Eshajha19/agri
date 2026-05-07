import React, { useState } from "react";
import "./AgriMarketplace.css";
import { Search, MapPin, Plus, Calendar, Clock, Phone, Info, X } from "lucide-react";



const TYPE_ICONS = {
  Tractor:   "🚜",
  Harvester: "🌾",
  Drone:     "🚁",
  Tillage:   "⚙️",
  Sowing:    "🌱",
};

const INITIAL_EQUIPMENT = [
  { id: 1,  name: "John Deere Tractor 5050D",    type: "Tractor",   price: 800,  priceUnit: "hr",  location: "Karnal, Haryana",      distance: "5 km",  rating: 4.8, owner: "Suresh Kumar",        available: true  },
  { id: 2,  name: "Mahindra Rice Harvester",      type: "Harvester", price: 2500, priceUnit: "day", location: "Ludhiana, Punjab",      distance: "12 km", rating: 4.5, owner: "Hardeep Singh",       available: true  },
  { id: 3,  name: "DJI Agras T40 Drone",          type: "Drone",     price: 1500, priceUnit: "hr",  location: "Bhopal, MP",           distance: "8 km",  rating: 4.9, owner: "TechAgri Solutions",   available: false },
  { id: 4,  name: "Sonalika Rotavator 200",        type: "Tillage",   price: 400,  priceUnit: "hr",  location: "Nagpur, Maharashtra",   distance: "3 km",  rating: 4.2, owner: "Ramesh Patil",        available: true  },
  { id: 5,  name: "Kubota MU5501 Tractor",         type: "Tractor",   price: 900,  priceUnit: "hr",  location: "Pune, Maharashtra",     distance: "4 km",  rating: 4.7, owner: "Santosh Shinde",      available: true  },
  { id: 6,  name: "Mahindra JIVO 245 DI",          type: "Tractor",   price: 750,  priceUnit: "hr",  location: "Nashik, Maharashtra",   distance: "6 km",  rating: 4.6, owner: "Ganesh Deshmukh",     available: true  },
  { id: 7,  name: "Preet 987 Combine Harvester",   type: "Harvester", price: 3200, priceUnit: "day", location: "Amravati, Maharashtra", distance: "9 km",  rating: 4.3, owner: "Vijay Bhalerao",      available: true  },
  { id: 8,  name: "AgriDrone Sprayer X8",          type: "Drone",     price: 1200, priceUnit: "hr",  location: "Pune, Maharashtra",     distance: "2 km",  rating: 4.8, owner: "DroneAgri Pvt Ltd",   available: true  },
  { id: 9,  name: "New Holland Excel 4710",        type: "Tractor",   price: 1100, priceUnit: "hr",  location: "Mumbai, Maharashtra",   distance: "15 km", rating: 4.4, owner: "Pramod Jadhav",      available: false },
  { id: 10, name: "Swaraj 744 FE Tractor",         type: "Tractor",   price: 700,  priceUnit: "hr",  location: "Solapur, Maharashtra",  distance: "7 km",  rating: 4.1, owner: "Arjun Kulkarni",     available: true  },
  { id: 11, name: "Fieldking Offset Disc Harrow",  type: "Tillage",   price: 350,  priceUnit: "hr",  location: "Aurangabad, Maharashtra",distance: "11 km",rating: 4.0, owner: "Sunil Mane",         available: true  },
  { id: 12, name: "VST Shakti Tractor 270 DI",     type: "Tractor",   price: 600,  priceUnit: "hr",  location: "Bengaluru, Karnataka",  distance: "10 km", rating: 4.3, owner: "Ravi Naik",          available: true  },
  { id: 13, name: "CLAAS Crop Tiger 30 Terra",      type: "Harvester", price: 4000, priceUnit: "day", location: "Hyderabad, Telangana",  distance: "18 km", rating: 4.7, owner: "Krishnamurthy Agro",  available: true  },
  { id: 14, name: "Ecorobotix ARA Sprayer",        type: "Drone",     price: 1800, priceUnit: "hr",  location: "Jaipur, Rajasthan",     distance: "6 km",  rating: 4.9, owner: "SmartFarm Solutions", available: true  },
  { id: 15, name: "Landforce Potato Planter",      type: "Sowing",    price: 500,  priceUnit: "hr",  location: "Agra, UP",              distance: "5 km",  rating: 4.2, owner: "Dinesh Agarwal",     available: true  },
  { id: 16, name: "Massey Ferguson 9500",          type: "Tractor",   price: 950,  priceUnit: "hr",  location: "Patna, Bihar",          distance: "14 km", rating: 4.5, owner: "Manoj Singh",        available: false },
];

export default function AgriMarketplace({ onClose }) {
  const [equipment, setEquipment] = useState(INITIAL_EQUIPMENT);
  const [searchQuery, setSearchQuery] = useState("");
  const [locationQuery, setLocationQuery] = useState("");
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

  const filteredEquipment = equipment.filter(item => 
    item.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
    item.location.toLowerCase().includes(locationQuery.toLowerCase())
  );

  const handleListEquipment = (e) => {
    e.preventDefault();
    const listing = {
      ...newListing,
      id: equipment.length + 1,
      distance: "0 km",
      rating: 5.0,
      owner: "You",
      available: true
    };
    setEquipment([listing, ...equipment]);
    setShowListModal(false);
    setNewListing({ name: "", type: "Tractor", price: "", priceUnit: "hr", location: "" });
  };

  const handleBooking = (id) => {
    if (!bookingDate) {
      setBookingError("Please select a date.");
      return;
    }
    if (!bookingTime) {
      setBookingError("Please select a start time.");
      return;
    }
    if (!bookingDuration || Number(bookingDuration) < 1) {
      setBookingError("Please enter a valid duration (minimum 1).");
      return;
    }
    setBookingError("");
    alert(`Booking request sent for ${equipment.find(e => e.id === id).name}! The owner will contact you shortly.`);
    setShowBookingModal(null);
    setBookingDate("");
    setBookingTime("");
    setBookingDuration("");
  };

  return (
    <div className="marketplace-container">
      <div className="marketplace-header">
        <div className="header-top">
          <h1>🚜 P2P Agri-Equipment Marketplace</h1>
          <button className="list-btn" onClick={() => setShowListModal(true)}>
            <Plus size={20} /> List Equipment
          </button>
        </div>
        
        <div className="search-bar-container">
          <div className="search-input">
            <Search size={20} />
            <input 
              type="text" 
              placeholder="Search tractors, harvesters..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="location-input">
            <MapPin size={20} />
            <input 
              type="text" 
              placeholder="Enter locality..." 
              value={locationQuery}
              onChange={(e) => setLocationQuery(e.target.value)}
            />
          </div>
        </div>
      </div>

      <div className="equipment-grid">
        {filteredEquipment.length > 0 ? (
          filteredEquipment.map(item => (
            <div key={item.id} className={`equipment-card ${!item.available ? 'unavailable' : ''}`}>
              <div className="card-icon-header">
                <span className="type-icon">{TYPE_ICONS[item.type] || "🔧"}</span>
                <div className="badge">{item.type}</div>
                {!item.available && <div className="unavailable-tag">Currently Rented</div>}
              </div>
              <div className="card-content">
                <div className="card-header">
                  <h3>{item.name}</h3>
                  <div className="rating">⭐ {item.rating}</div>
                </div>
                <p className="owner">Owner: {item.owner}</p>
                <div className="details">
                  <div className="detail-item">
                    <MapPin size={16} /> {item.location} ({item.distance})
                  </div>
                  <div className="price">
                    ₹{item.price}<span>/{item.priceUnit}</span>
                  </div>
                </div>
                <button 
                  className="book-btn" 
                  disabled={!item.available}
                  onClick={() => setShowBookingModal(item.id)}
                >
                  {item.available ? "Book Now" : "Unavailable"}
                </button>
              </div>
            </div>
          ))
        ) : (
          <div className="no-results">
            <p>No equipment found matching your criteria. Try adjusting your search.</p>
          </div>
        )}
      </div>

      {/* List Equipment Modal */}
      {showListModal && (
        <div className="modal-overlay" onClick={() => setShowListModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-modal" onClick={() => setShowListModal(false)}><X size={24} /></button>
            <h2>📢 List Your Equipment</h2>
            <form onSubmit={handleListEquipment}>
              <div className="form-group">
                <label>Equipment Name</label>
                <input 
                  type="text" 
                  required 
                  placeholder="e.g. Sonalika Tractor" 
                  value={newListing.name}
                  onChange={(e) => setNewListing({...newListing, name: e.target.value})}
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Type</label>
                  <select 
                    value={newListing.type}
                    onChange={(e) => setNewListing({...newListing, type: e.target.value})}
                  >
                    <option>Tractor</option>
                    <option>Harvester</option>
                    <option>Drone</option>
                    <option>Tillage</option>
                    <option>Sowing</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Price</label>
                  <div className="price-input">
                    <input 
                      type="number" 
                      required 
                      placeholder="Amount" 
                      value={newListing.price}
                      onChange={(e) => setNewListing({...newListing, price: e.target.value})}
                    />
                    <select 
                      value={newListing.priceUnit}
                      onChange={(e) => setNewListing({...newListing, priceUnit: e.target.value})}
                    >
                      <option value="hr">/hr</option>
                      <option value="day">/day</option>
                    </select>
                  </div>
                </div>
              </div>
              <div className="form-group">
                <label>Location</label>
                <input 
                  type="text" 
                  required 
                  placeholder="Your locality" 
                  value={newListing.location}
                  onChange={(e) => setNewListing({...newListing, location: e.target.value})}
                />
              </div>

              <button type="submit" className="submit-btn">Post Listing</button>
            </form>
          </div>
        </div>
      )}

      {/* Booking Modal */}
      {showBookingModal && (
        <div className="modal-overlay" onClick={() => setShowBookingModal(null)}>
          <div className="modal-content booking-modal" onClick={(e) => e.stopPropagation()}>
            <button className="close-modal" onClick={() => setShowBookingModal(null)}><X size={24} /></button>
            <h2>📅 Schedule Booking</h2>
            <div className="item-info">
              <span className="item-info-icon">{TYPE_ICONS[equipment.find(e => e.id === showBookingModal).type] || "🔧"}</span>
              <div>
                <h3>{equipment.find(e => e.id === showBookingModal).name}</h3>
                <p>₹{equipment.find(e => e.id === showBookingModal).price}/{equipment.find(e => e.id === showBookingModal).priceUnit}</p>
              </div>
            </div>
            
            <div className="booking-form">
              <div className="form-group">
                <label><Calendar size={16} /> Select Date</label>
                <input
                  type="date"
                  value={bookingDate}
                  min={new Date().toISOString().split('T')[0]}
                  onChange={(e) => { setBookingDate(e.target.value); setBookingError(""); }}
                />
                {bookingError && !bookingTime && !bookingDuration && (
                  <div style={{ color: '#f87171', fontSize: '0.82rem', marginTop: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    ⚠️ {bookingError}
                  </div>
                )}
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label><Clock size={16} /> Start Time</label>
                  <input
                    type="time"
                    value={bookingTime}
                    onChange={(e) => { setBookingTime(e.target.value); setBookingError(""); }}
                  />
                  {bookingDate && bookingError && !bookingDuration && (
                    <div style={{ color: '#f87171', fontSize: '0.82rem', marginTop: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      ⚠️ {bookingError}
                    </div>
                  )}
                </div>
                <div className="form-group">
                  <label><Clock size={16} /> Duration ({equipment.find(e => e.id === showBookingModal).priceUnit === 'hr' ? 'Hours' : 'Days'})</label>
                  <input
                    type="number"
                    min="1"
                    placeholder="e.g. 5"
                    value={bookingDuration}
                    onChange={(e) => { setBookingDuration(e.target.value); setBookingError(""); }}
                  />
                  {bookingDate && bookingTime && bookingError && (
                    <div style={{ color: '#f87171', fontSize: '0.82rem', marginTop: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      ⚠️ {bookingError}
                    </div>
                  )}
                </div>
              </div>
              <div className="total-cost">
                <span>Estimated Total:</span>
                <strong>{bookingDuration && Number(bookingDuration) > 0
                  ? `₹${equipment.find(e => e.id === showBookingModal).price * Number(bookingDuration)}`
                  : '₹ --'}
                </strong>
              </div>
              <button className="confirm-btn" onClick={() => handleBooking(showBookingModal)}>
                Confirm Booking
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}