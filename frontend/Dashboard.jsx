import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  FaUser,
  FaSeedling,
  FaCloudSun,
  FaChartLine,
  FaTractor,
  FaCalendarAlt,
  FaMapMarkerAlt,
  FaArrowRight,
  FaLeaf,
  FaBell,
  FaWater,
  FaBug,
  FaComments,
  FaWhatsapp,
  FaCheckCircle,
} from "react-icons/fa";
import "./Dashboard.css";

export default function Dashboard() {
  const name = localStorage.getItem("farmerName") || "Farmer";
  const preferredLang = localStorage.getItem("preferredLanguage") || "en";

  const [currentTime, setCurrentTime] = useState(new Date());

  const [phoneNumber, setPhoneNumber] = useState(localStorage.getItem("farmerPhone") || "");
  const [whatsappAlerts, setWhatsappAlerts] = useState(localStorage.getItem("whatsappAlerts") === "true");
  const [isUpdating, setIsUpdating] = useState(false);
  const [updateMsg, setUpdateMsg] = useState("");

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  const handleUpdateWhatsApp = async () => {
    setIsUpdating(true);
    setUpdateMsg("");
    try {
      const response = await fetch("http://localhost:8000/api/whatsapp/subscribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: localStorage.getItem("userId") || "user_" + name,
          phone_number: phoneNumber,
          name: name,
          enabled: whatsappAlerts
        }),
      });
      const data = await response.json();
      if (data.success) {
        localStorage.setItem("farmerPhone", phoneNumber);
        localStorage.setItem("whatsappAlerts", whatsappAlerts.toString());
        setUpdateMsg("Settings saved successfully!");
        setTimeout(() => setUpdateMsg(""), 3000);
      }
    } catch (err) {
      setUpdateMsg("Error saving settings.");
    } finally {
      setIsUpdating(false);
    }
  };

  const getGreeting = () => {
    const hour = currentTime.getHours();
    if (hour < 12) return "Good Morning";
    if (hour < 17) return "Good Afternoon";
    return "Good Evening";
  };

  const getFormattedDate = () => {
    return currentTime.toLocaleDateString("en-IN", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const quickStats = [
    { label: "Active Crops", value: "4", icon: <FaSeedling />, trend: "+1 this month" },
    { label: "Farm Area", value: "12 Acres", icon: <FaMapMarkerAlt />, trend: "Paddy, Cotton" },
    { label: "Yield Score", value: "87%", icon: <FaChartLine />, trend: "+5% vs last season" },
    { label: "Next Harvest", value: "45 Days", icon: <FaCalendarAlt />, trend: "Kharif Season" },
  ];

  const recentActivity = [
    {
      icon: <FaCloudSun />,
      title: "Weather Alert Received",
      description: "Heavy rainfall expected in your region for the next 3 days",
      time: "2 hours ago",
      type: "warning",
    },
    {
      icon: <FaSeedling />,
      title: "Crop Health Check Completed",
      description: "Paddy field section A shows healthy growth patterns",
      time: "5 hours ago",
      type: "success",
    },
    {
      icon: <FaChartLine />,
      title: "Yield Prediction Updated",
      description: "Expected yield increased to 42 quintals/acre for current season",
      time: "1 day ago",
      type: "info",
    },
    {
      icon: <FaWater />,
      title: "Irrigation Schedule Set",
      description: "Drip irrigation scheduled for tomorrow at 6:00 AM",
      time: "1 day ago",
      type: "default",
    },
    {
      icon: <FaBug />,
      title: "Pest Alert Dismissed",
      description: "Brown planthopper risk level returned to normal",
      time: "2 days ago",
      type: "success",
    },
    {
      icon: <FaTractor />,
      title: "Soil Test Report Ready",
      description: "Nitrogen levels optimal, phosphorus slightly low",
      time: "3 days ago",
      type: "info",
    },
  ];

  const recommendations = [
    {
      icon: <FaLeaf />,
      title: "Switch to Drip Irrigation",
      description: "Based on your soil type and crop selection, drip irrigation can save up to 40% water and increase yield by 15%.",
      tag: "Water Management",
    },
    {
      icon: <FaSeedling />,
      title: "Plant Cover Crops",
      description: "Adding leguminous cover crops between seasons improves soil nitrogen and reduces fertilizer costs by 25%.",
      tag: "Soil Health",
    },
    {
      icon: <FaBell />,
      title: "Optimal Sowing Window",
      description: "Weather data suggests the best sowing window for Rabi wheat is in the next 10-15 days for your region.",
      tag: "Planning",
    },
    {
      icon: <FaChartLine />,
      title: "Market Price Trending Up",
      description: "Cotton prices have risen 12% this month. Consider timing your harvest for maximum returns.",
      tag: "Market",
    },
  ];

  const quickActions = [
    { label: "AI Advisor", icon: <FaSeedling />, link: "/advisor" },
    { label: "Community", icon: <FaComments />, link: "/community" },
    { label: "How It Works", icon: <FaChartLine />, link: "/how-it-works" },
  ];

  return (
    <div className="dashboard">
      <section className="dashboard-hero">
        <div className="dashboard-hero-bg"></div>
        <div className="dashboard-hero-content">
          <div className="welcome-block">
            <div className="user-avatar">
              <FaUser />
            </div>
            <div className="welcome-text">
              <h1>{getGreeting()}, {name}</h1>
              <p className="welcome-date">{getFormattedDate()}</p>
              <p className="welcome-sub">Here is an overview of your farm activity and insights</p>
            </div>
          </div>
          <div className="quick-actions-row">
            {quickActions.map((action, idx) => (
              <Link to={action.link} key={idx} className="quick-action-btn">
                {action.icon}
                <span>{action.label}</span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="dashboard-stats">
        {quickStats.map((stat, idx) => (
          <div className="stat-card" key={idx}>
            <div className="stat-card-icon">{stat.icon}</div>
            <div className="stat-card-info">
              <span className="stat-card-value">{stat.value}</span>
              <span className="stat-card-label">{stat.label}</span>
              <span className="stat-card-trend">{stat.trend}</span>
            </div>
          </div>
        ))}
      </section>

      <section className="dashboard-grid">
        <div className="dashboard-column">
          <div className="dashboard-section-card">
            <div className="section-card-header">
              <h2>Recent Activity</h2>
              <span className="section-badge">{recentActivity.length} updates</span>
            </div>
            <div className="activity-list">
              {recentActivity.map((item, idx) => (
                <div className="activity-item" key={idx}>
                  <div className={`activity-icon activity-${item.type}`}>
                    {item.icon}
                  </div>
                  <div className="activity-content">
                    <span className="activity-title">{item.title}</span>
                    <span className="activity-desc">{item.description}</span>
                    <span className="activity-time">{item.time}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="dashboard-column">
          <div className="dashboard-section-card">
            <div className="section-card-header">
              <h2>Recommendations</h2>
              <span className="section-badge">AI Powered</span>
            </div>
            <div className="recommendations-list">
              {recommendations.map((rec, idx) => (
                <div className="recommendation-card" key={idx}>
                  <div className="rec-icon">{rec.icon}</div>
                  <div className="rec-content">
                    <div className="rec-header-row">
                      <span className="rec-title">{rec.title}</span>
                      <span className="rec-tag">{rec.tag}</span>
                    </div>
                    <p className="rec-desc">{rec.description}</p>
                  </div>
                  <FaArrowRight className="rec-arrow" />
                </div>
              ))}
            </div>
          </div>

          <div className="dashboard-section-card farm-summary-card">
            <div className="section-card-header">
              <h2>Farm Overview</h2>
            </div>
            <div className="farm-summary-grid">
              <div className="farm-summary-item">
                <span className="farm-summary-label">Primary Crop</span>
                <span className="farm-summary-value">Paddy</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Season</span>
                <span className="farm-summary-value">Kharif</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Soil Type</span>
                <span className="farm-summary-value">Alluvial</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Irrigation</span>
                <span className="farm-summary-value">Drip</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Region</span>
                <span className="farm-summary-value">Maharashtra</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Language</span>
                <span className="farm-summary-value">{preferredLang.toUpperCase()}</span>
              </div>
            </div>
            <Link to="/advisor" className="farm-cta-btn">
              Get AI Advice <FaArrowRight />
            </Link>
          </div>

          <div className="dashboard-section-card whatsapp-settings-card">
            <div className="section-card-header">
              <h2><FaWhatsapp /> WhatsApp Alerts</h2>
              <span className={`status-dot ${whatsappAlerts ? "status-active" : ""}`}></span>
            </div>
            <div className="whatsapp-settings-body">
              <p className="settings-intro">Receive real-time weather and pest alerts on your phone.</p>
              <div className="input-group">
                <label>Phone Number (with code)</label>
                <input 
                  type="text" 
                  placeholder="+91 9876543210" 
                  value={phoneNumber} 
                  onChange={(e) => setPhoneNumber(e.target.value)}
                />
              </div>
              <div className="checkbox-group">
                <input 
                  type="checkbox" 
                  id="wa-toggle" 
                  checked={whatsappAlerts} 
                  onChange={(e) => setWhatsappAlerts(e.target.checked)}
                />
                <label htmlFor="wa-toggle">Enable Real-time Alerts</label>
              </div>
              <button 
                className={`save-wa-btn ${isUpdating ? "loading" : ""}`} 
                onClick={handleUpdateWhatsApp}
                disabled={isUpdating}
              >
                {isUpdating ? "Saving..." : "Save Settings"}
              </button>
              {updateMsg && (
                <p className={`update-msg ${updateMsg.includes("Error") ? "error" : "success"}`}>
                  {updateMsg.includes("success") && <FaCheckCircle />} {updateMsg}
                </p>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
