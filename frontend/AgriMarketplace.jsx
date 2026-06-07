/**
 * AgriMarketplace.jsx — P2P Agri-Equipment Marketplace
 *
 * Security / correctness fix: all equipment listings and booking requests
 * are now persisted server-side via the /api/marketplace/* endpoints.
 *
 * Previously:
 *  - 16 listings were hardcoded in INITIAL_EQUIPMENT (client-only).
 *  - "List Equipment" added entries to React state only — gone on refresh.
 *  - "Confirm Booking" showed a browser alert() with no API call, no
 *    Firestore write, and no notification to the equipment owner.
 *
 * Now:
 *  - GET  /api/marketplace/listings  — fetches server-side listings on mount.
 *  - POST /api/marketplace/listings  — persists new listings (auth required).
 *  - POST /api/marketplace/bookings  — persists bookings (auth required);
 *    the owner's listing is marked unavailable server-side.
 *  - GET  /api/marketplace/bookings  — shows the farmer's own bookings.
 */
import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
} from "react";
import "./AgriMarketplace.css";
import {
  Search, MapPin, Plus, Calendar, Clock, X, AlertCircle, CheckCircle,
} from "lucide-react";
import apiClient from "./lib/apiClient";

const TYPE_ICONS = {
  Tractor:   "🚜",
  Harvester: "🌾",
  Drone:     "🚁",
  Tillage:   "⚙️",
  Sowing:    "🌱",
  Other:     "🔧",
};

const EMPTY_LISTING = {
  name: "", type: "Tractor", price: "", priceUnit: "hr", location: "",
};

export default function AgriMarketplace({ onClose }) {
  const [listings, setListings]           = useState([]);
  const [loadingListings, setLoadingListings] = useState(true);
  const [listingsError, setListingsError] = useState("");

  const [searchQuery, setSearchQuery]     = useState("");
  const [locationQuery, setLocationQuery] = useState("");

  const [showListModal, setShowListModal] = useState(false);
  const [newListing, setNewListing]       = useState(EMPTY_LISTING);
  const [listSubmitting, setListSubmitting] = useState(false);
  const [listError, setListError]         = useState("");
  const [listSuccess, setListSuccess]     = useState("");

  const [showBookingModal, setShowBookingModal] = useState(null); // equipmentId | null
  const [bookingDate, setBookingDate]     = useState("");
  const [bookingTime, setBookingTime]     = useState("");
  const [bookingDuration, setBookingDuration] = useState("");
  const [bookingSubmitting, setBookingSubmitting] = useState(false);
  const [bookingError, setBookingError]   = useState("");
  const [bookingSuccess, setBookingSuccess] = useState("");
  const mountedRef = useRef(true);
  const filterRequestRef = useRef(0);
  const debounceRef = useRef(null);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;

      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  // ── Fetch listings from server ────────────────────────────────────────────
  const fetchListings = useCallback(async () => {
    const requestId = ++filterRequestRef.current;

    setLoadingListings(true);
    setListingsError("");
    try {
      const params = new URLSearchParams();
      if (searchQuery)  params.set("search",   searchQuery);
      if (locationQuery) params.set("location", locationQuery);
      const res = await apiClient.get(`/api/marketplace/listings?${params}`);

      if (
        !mountedRef.current ||
        requestId !== filterRequestRef.current
      ) {
        return;
      }

      setListings(res.data?.data || []);
    } catch {
      if (
        !mountedRef.current ||
        requestId !== filterRequestRef.current
      ) {
        return;
      }

      setListingsError(
        "Failed to load equipment listings. Please try again."
      );
    }
  }, [searchQuery, locationQuery]);

  useEffect(() => {
    const requestId = ++filterRequestRef.current;

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      if (
        mountedRef.current &&
        requestId === filterRequestRef.current
      ) {
        fetchListings();
      }
    }, 300);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [fetchListings]);

  // ── List new equipment ────────────────────────────────────────────────────
  const handleListEquipment = async (e) => {
    e.preventDefault();
    setListError("");
    setListSuccess("");
    setListSubmitting(true);
    try {
      const res = await apiClient.post("/api/marketplace/listings", {
        name:      newListing.name,
        type:      newListing.type,
        price:     Number(newListing.price),
        priceUnit: newListing.priceUnit,
        location:  newListing.location,
      });
      const created = res.data?.listing;
      if (created) {
        setListings(prev => [created, ...prev]);
        setListSuccess(`"${created.name}" listed successfully.`);
        setNewListing(EMPTY_LISTING);
        setTimeout(() => { setShowListModal(false); setListSuccess(""); }, 1800);
      }
    } catch (err) {
      const status = err?.response?.status;
      if (status === 401) {
        setListError("You must be logged in to list equipment.");
      } else {
        setListError("Failed to post listing. Please try again.");
      }
    } finally {
      setListSubmitting(false);
    }
  };

  // ── Book equipment ────────────────────────────────────────────────────────
  const handleBooking = async (equipmentId) => {
    if (!bookingDate)                          { setBookingError("Please select a date."); return; }
    if (!bookingTime)                          { setBookingError("Please select a start time."); return; }
    if (!bookingDuration || Number(bookingDuration) < 1) { setBookingError("Please enter a valid duration (minimum 1)."); return; }

    setBookingError("");
    setBookingSuccess("");
    setBookingSubmitting(true);

    try {
      const res = await apiClient.post("/api/marketplace/bookings", {
        equipmentId,
        date:     bookingDate,
        time:     bookingTime,
        duration: Number(bookingDuration),
      });
      const booking = res.data?.booking;
      if (booking) {
        // Mark the listing as unavailable in local state immediately.
        setListings(prev =>
          prev.map(l => l.id === equipmentId ? { ...l, available: false } : l)
        );
        setBookingSuccess(
          `Booking confirmed! Booking ID: ${booking.id.slice(0, 8).toUpperCase()}. ` +
          `The owner has been notified.`
        );
        // Close modal after a short delay so the farmer can read the confirmation.
        setTimeout(() => {
          setShowBookingModal(null);
          setBookingDate(""); setBookingTime(""); setBookingDuration("");
          setBookingSuccess("");
        }, 2500);
      }
    } catch (err) {
      const status = err?.response?.status;
      if (status === 401) {
        setBookingError("You must be logged in to book equipment.");
      } else if (status === 409) {
        setBookingError("This equipment is no longer available. Please choose another.");
        setListings(prev =>
          prev.map(l => l.id === equipmentId ? { ...l, available: false } : l)
        );
      } else if (status === 404) {
        setBookingError("Equipment listing not found.");
      } else {
        setBookingError("Booking failed. Please try again.");
      }
    } finally {
      setBookingSubmitting(false);
    }
  };

  const selectedItem = useMemo(() => {
    return showBookingModal
      ? listings.find(
          l => l.id === showBookingModal
        )
      : null;
  }, [showBookingModal, listings]);

  const estimatedCost = useMemo(() => {
    return selectedItem &&
      bookingDuration &&
      Number(bookingDuration) > 0
        ? selectedItem.price *
            Number(bookingDuration)
        : null;
  }, [selectedItem, bookingDuration]);

  const handleSearchChange = useCallback((e) => {
    setSearchQuery(e.target.value);
  }, []);

  const handleLocationChange = useCallback((e) => {
    setLocationQuery(e.target.value);
  }, []);

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
              onChange={handleSearchChange}
            />
          </div>
          <div className="location-input">
            <MapPin size={20} />
            <input
              type="text"
              placeholder="Enter locality..."
              value={locationQuery}
              onChange={handleLocationChange}
            />
          </div>
        </div>
      </div>

      {/* Listings grid */}
      {loadingListings ? (
        <div className="loading-msg">Loading equipment listings…</div>
      ) : listingsError ? (
        <div className="error-msg" role="alert">
          <AlertCircle size={16} style={{ marginRight: 6 }} />{listingsError}
        </div>
      ) : (
        <div className="equipment-grid">
          {listings.length > 0 ? listings.map(item => (
            <div key={item.id} className={`equipment-card ${!item.available ? "unavailable" : ""}`}>
              <div className="card-icon-header">
                <span className="type-icon">{TYPE_ICONS[item.type] || "🔧"}</span>
                <div className="badge">{item.type}</div>
                {!item.available && <div className="unavailable-tag">Currently Rented</div>}
              </div>
              <div className="card-content">
                <div className="card-header">
                  <h3>{item.name}</h3>
                  <div className="rating">⭐ {item.rating?.toFixed(1) ?? "—"}</div>
                </div>
                <p className="owner">Owner: {item.owner}</p>
                <div className="details">
                  <div className="detail-item">
                    <MapPin size={16} /> {item.location}
                  </div>
                  <div className="price">
                    ₹{item.price}<span>/{item.priceUnit}</span>
                  </div>
                </div>
                <button
                  className="book-btn"
                  disabled={!item.available}
                  onClick={() => {
                    setShowBookingModal(item.id);
                    setBookingError(""); setBookingSuccess("");
                    setBookingDate(""); setBookingTime(""); setBookingDuration("");
                  }}
                >
                  {item.available ? "Book Now" : "Unavailable"}
                </button>
              </div>
            </div>
          )) : (
            <div className="no-results">
              <p>No equipment found matching your criteria. Try adjusting your search.</p>
            </div>
          )}
        </div>
      )}

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
                  type="text" required placeholder="e.g. Sonalika Tractor"
                  value={newListing.name}
                  onChange={(e) => setNewListing({ ...newListing, name: e.target.value })}
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Type</label>
                  <select value={newListing.type} onChange={(e) => setNewListing({ ...newListing, type: e.target.value })}>
                    <option>Tractor</option>
                    <option>Harvester</option>
                    <option>Drone</option>
                    <option>Tillage</option>
                    <option>Sowing</option>
                    <option>Other</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Price</label>
                  <div className="price-input">
                    <input
                      type="number" required placeholder="Amount" min="1"
                      value={newListing.price}
                      onChange={(e) => setNewListing({ ...newListing, price: e.target.value })}
                    />
                    <select value={newListing.priceUnit} onChange={(e) => setNewListing({ ...newListing, priceUnit: e.target.value })}>
                      <option value="hr">/hr</option>
                      <option value="day">/day</option>
                    </select>
                  </div>
                </div>
              </div>
              <div className="form-group">
                <label>Location</label>
                <input
                  type="text" required placeholder="Your locality"
                  value={newListing.location}
                  onChange={(e) => setNewListing({ ...newListing, location: e.target.value })}
                />
              </div>
              {listError && (
                <div className="error-msg" role="alert">
                  <AlertCircle size={14} style={{ marginRight: 6 }} />{listError}
                </div>
              )}
              {listSuccess && (
                <div className="success-msg" role="status">
                  <CheckCircle size={14} style={{ marginRight: 6 }} />{listSuccess}
                </div>
              )}
              <button type="submit" className="submit-btn" disabled={listSubmitting}>
                {listSubmitting ? "Posting…" : "Post Listing"}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Booking Modal */}
      {showBookingModal && selectedItem && (
        <div className="modal-overlay" onClick={() => setShowBookingModal(null)}>
          <div className="modal-content booking-modal" onClick={(e) => e.stopPropagation()}>
            <button className="close-modal" onClick={() => setShowBookingModal(null)}><X size={24} /></button>
            <h2>📅 Schedule Booking</h2>
            <div className="item-info">
              <span className="item-info-icon">{TYPE_ICONS[selectedItem.type] || "🔧"}</span>
              <div>
                <h3>{selectedItem.name}</h3>
                <p>₹{selectedItem.price}/{selectedItem.priceUnit}</p>
              </div>
            </div>

            <div className="booking-form">
              <div className="form-group">
                <label><Calendar size={16} /> Select Date</label>
                <input
                  type="date"
                  value={bookingDate}
                  min={new Date().toISOString().split("T")[0]}
                  onChange={(e) => { setBookingDate(e.target.value); setBookingError(""); }}
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label><Clock size={16} /> Start Time</label>
                  <input
                    type="time"
                    value={bookingTime}
                    onChange={(e) => { setBookingTime(e.target.value); setBookingError(""); }}
                  />
                </div>
                <div className="form-group">
                  <label>
                    <Clock size={16} /> Duration ({selectedItem.priceUnit === "hr" ? "Hours" : "Days"})
                  </label>
                  <input
                    type="number" min="1" placeholder="e.g. 5"
                    value={bookingDuration}
                    onChange={(e) => { setBookingDuration(e.target.value); setBookingError(""); }}
                  />
                </div>
              </div>

              <div className="total-cost">
                <span>Estimated Total:</span>
                <strong>{estimatedCost !== null ? `₹${estimatedCost}` : "₹ --"}</strong>
              </div>

              {bookingError && (
                <div className="error-msg" role="alert">
                  <AlertCircle size={14} style={{ marginRight: 6 }} />{bookingError}
                </div>
              )}
              {bookingSuccess && (
                <div className="success-msg" role="status">
                  <CheckCircle size={14} style={{ marginRight: 6 }} />{bookingSuccess}
                </div>
              )}

              <button
                className="confirm-btn"
                onClick={() => handleBooking(showBookingModal)}
                disabled={bookingSubmitting || !!bookingSuccess}
              >
                {bookingSubmitting ? "Confirming…" : "Confirm Booking"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
