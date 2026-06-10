import React, { useEffect, useState } from "react";
import { 
  FaPhoneAlt, 
  FaCloudShowersHeavy, 
  FaShieldAlt, 
  FaFlask, 
  FaHeadset, 
  FaInfoCircle,
  FaTimes,
  FaCheckCircle,
  FaExternalLinkAlt
} from "react-icons/fa";
import "./Helpline.css";

const helplineData = [
  {
    id: 1,
    category: "Agriculture Support",
    title: "Kisan Call Center (KCC)",
    number: "1800-180-1551",
    description: "24/7 support for agricultural queries, pest control, and farming techniques in multiple languages.",
    details: {
      availability: "Available 24/7 in multiple Indian languages.",
      whoToCall: "Call when you need crop advice, input guidance, or pest-management support.",
      actions: ["Keep your crop name, district, and issue details ready.", "Ask for the local agri officer escalation path if the issue is urgent."],
    },
    icon: <FaHeadset />,
    color: "#2e7d32"
  },
  {
    id: 2,
    category: "Weather Emergencies",
    title: "IMD Weather Helpline",
    number: "1800-180-1551",
    description: "Real-time weather alerts and emergency forecasting for farmers during storms or floods.",
    details: {
      availability: "Use during severe weather, heat waves, storms, floods, or frost alerts.",
      whoToCall: "Call before irrigation, spraying, or harvest work when weather conditions are unstable.",
      actions: ["Check nearby shelter, drainage, and crop protection steps.", "Avoid field activity until the forecast is confirmed safe."],
    },
    icon: <FaCloudShowersHeavy />,
    color: "#1976d2"
  },
  {
    id: 3,
    category: "Crop Insurance",
    title: "PMFBY Support",
    number: "1800-180-1551",
    description: "Guidance on Pradhan Mantri Fasal Bima Yojana, claim procedures, and policy status.",
    details: {
      availability: "Best for crop loss, claim status, and policy support questions.",
      whoToCall: "Call after weather damage, pest damage, or when you need claim guidance.",
      actions: ["Keep your policy number and crop survey details ready.", "Ask for the nearest office if you need claim assistance in person."],
    },
    icon: <FaShieldAlt />,
    color: "#f57c00"
  },
  {
    id: 4,
    category: "Soil Testing",
    title: "Soil Health Portal",
    number: "011-24305530",
    description: "Locate nearest soil testing centers and get help with Soil Health Card (SHC) applications.",
    details: {
      availability: "Use for soil-testing appointments and Soil Health Card support.",
      whoToCall: "Call before fertilizer application or when yield drops unexpectedly.",
      actions: ["Ask for sample collection instructions.", "Request the nearest lab or center details for your district."],
    },
    icon: <FaFlask />,
    color: "#7b1fa2"
  },
  {
    id: 5,
    category: "Disaster Management",
    title: "National Emergency Hub",
    number: "1070",
    description: "Immediate assistance during natural disasters, cyclones, and large-scale farming crises.",
    details: {
      availability: "Use during life-threatening weather, floods, fire, or large-scale damage.",
      whoToCall: "Call immediately when people, livestock, or property are at risk.",
      actions: ["Move to a safe location first.", "Share your exact location and the kind of emergency."],
    },
    icon: <FaInfoCircle />,
    color: "#d32f2f"
  }
];

const Helpline = () => {
  const [activeCard, setActiveCard] = useState(null);

  useEffect(() => {
    if (!activeCard) return undefined;

    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setActiveCard(null);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeCard]);

  return (
    <div className="helpline-container">
      <header className="helpline-header">
        <div className="header-badge">EMERGENCY ASSISTANCE</div>
        <h1>Farming Helplines & Support Hub</h1>
        <p>Access critical support services, emergency numbers, and expert advice for your farming needs.</p>
      </header>

      <div className="helpline-grid">
        {helplineData.map((item) => (
          <div key={item.id} className="helpline-card" style={{"--accent-color": item.color}}>
            <div className="card-top">
              <div className="icon-wrapper">
                {item.icon}
              </div>
              <div className="category-tag">{item.category}</div>
            </div>
            
            <div className="card-body">
              <h3>{item.title}</h3>
              <p>{item.description}</p>
              
              <div className="number-display">
                <FaPhoneAlt className="phone-icon" />
                <span>{item.number}</span>
              </div>
            </div>

            <div className="card-footer">
              <a href={`tel:${item.number.replace(/-/g, '')}`} className="call-btn">
                <FaPhoneAlt /> Call Now
              </a>
              <button
                type="button"
                className="info-btn"
                title={`View more details for ${item.title}`}
                onClick={() => setActiveCard(item)}
              >
                Details
              </button>
            </div>
          </div>
        ))}
      </div>

      {activeCard && (
        <div className="helpline-modal-backdrop" onClick={() => setActiveCard(null)}>
          <div className="helpline-modal" role="dialog" aria-modal="true" aria-labelledby="helpline-modal-title" onClick={(event) => event.stopPropagation()}>
            <button type="button" className="modal-close-btn" onClick={() => setActiveCard(null)} aria-label="Close details">
              <FaTimes />
            </button>

            <div className="modal-header" style={{ "--accent-color": activeCard.color }}>
              <div className="modal-icon">{activeCard.icon}</div>
              <div>
                <p className="modal-category">{activeCard.category}</p>
                <h3 id="helpline-modal-title">{activeCard.title}</h3>
              </div>
            </div>

            <div className="modal-number-row">
              <FaPhoneAlt />
              <a href={`tel:${activeCard.number.replace(/-/g, '')}`}>{activeCard.number}</a>
            </div>

            <p className="modal-description">{activeCard.description}</p>

            <div className="modal-detail-block">
              <h4>When to use it</h4>
              <p>{activeCard.details.availability}</p>
            </div>

            <div className="modal-detail-block">
              <h4>What to say</h4>
              <p>{activeCard.details.whoToCall}</p>
            </div>

            <div className="modal-detail-block">
              <h4>Quick steps</h4>
              <ul className="modal-action-list">
                {activeCard.details.actions.map((action) => (
                  <li key={action}>
                    <FaCheckCircle />
                    <span>{action}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="modal-footer">
              <button type="button" className="modal-primary-btn" onClick={() => setActiveCard(null)}>
                Close details
              </button>
              <a href={`tel:${activeCard.number.replace(/-/g, '')}`} className="modal-call-btn">
                <FaPhoneAlt /> Call now
              </a>
            </div>
          </div>
        </div>
      )}

      <section className="portal-links">
        <h2>Important Government Portals</h2>
        <div className="portal-grid">
          <a href="https://pmkisan.gov.in/" target="_blank" rel="noopener noreferrer" className="portal-card">
            <span>PM-Kisan Portal</span>
            <FaExternalLinkAlt />
          </a>
          <a href="https://pmfby.gov.in/" target="_blank" rel="noopener noreferrer" className="portal-card">
            <span>Crop Insurance (PMFBY)</span>
            <FaExternalLinkAlt />
          </a>
          <a href="https://soilhealth.dac.gov.in/" target="_blank" rel="noopener noreferrer" className="portal-card">
            <span>Soil Health Card Portal</span>
            <FaExternalLinkAlt />
          </a>
        </div>
      </section>

      <div className="disclaimer-box">
        <FaInfoCircle />
        <p>
          <strong>Note:</strong> Most of these numbers are toll-free and available in regional languages. 
          For immediate life-threatening emergencies, please contact your local police or medical services first.
        </p>
      </div>
    </div>
  );
};

export default Helpline;
