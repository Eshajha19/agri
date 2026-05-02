import React from "react";
import { 
  FaPhoneAlt, 
  FaCloudShowersHeavy, 
  FaShieldAlt, 
  FaFlask, 
  FaHeadset, 
  FaInfoCircle,
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
    icon: <FaHeadset />,
    color: "#2e7d32"
  },
  {
    id: 2,
    category: "Weather Emergencies",
    title: "IMD Weather Helpline",
    number: "1800-180-1551",
    description: "Real-time weather alerts and emergency forecasting for farmers during storms or floods.",
    icon: <FaCloudShowersHeavy />,
    color: "#1976d2"
  },
  {
    id: 3,
    category: "Crop Insurance",
    title: "PMFBY Support",
    number: "1800-180-1551",
    description: "Guidance on Pradhan Mantri Fasal Bima Yojana, claim procedures, and policy status.",
    icon: <FaShieldAlt />,
    color: "#f57c00"
  },
  {
    id: 4,
    category: "Soil Testing",
    title: "Soil Health Portal",
    number: "011-24305530",
    description: "Locate nearest soil testing centers and get help with Soil Health Card (SHC) applications.",
    icon: <FaFlask />,
    color: "#7b1fa2"
  },
  {
    id: 5,
    category: "Disaster Management",
    title: "National Emergency Hub",
    number: "1070",
    description: "Immediate assistance during natural disasters, cyclones, and large-scale farming crises.",
    icon: <FaInfoCircle />,
    color: "#d32f2f"
  }
];

const Helpline = () => {
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
              <button className="info-btn" title="View More Details">
                Details
              </button>
            </div>
          </div>
        ))}
      </div>

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
