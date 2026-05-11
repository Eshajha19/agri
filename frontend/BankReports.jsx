import React, { useState } from "react";
import { FaFileInvoiceDollar, FaDownload, FaShieldAlt, FaCheckCircle, FaSpinner, FaHistory } from "react-icons/fa";
import "./BankReports.css";
import apiClient from "./lib/apiClient";

const BankReports = ({ userData }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [reports, setReports] = useState(JSON.parse(localStorage.getItem("farm_reports") || "[]"));

  const [formData, setFormData] = useState({
    name: userData?.displayName || "",
    crop: userData?.cropType || "Wheat",
    area: userData?.area || "5 Acres",
    profit: "50,000",
    season: "Kharif 2026"
  });

  const handleGenerate = async () => {
    setLoading(true);
    setError("");
    try {
      // Use apiClient instead of raw fetch() so the Firebase auth token is
      // automatically injected via the Axios request interceptor in
      // services/api.js.  The raw fetch() call had no Authorization header,
      // so every request was rejected with HTTP 401 by verify_role() on the
      // backend — the feature was silently broken for all users.
      //
      // responseType: 'blob' tells Axios to treat the response body as binary
      // data (the PDF file) rather than trying to parse it as JSON.
      const response = await apiClient.post(
        "/api/reports/generate",
        formData,
        { responseType: "blob" }
      );

      const blob = response.data;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `FasalSaathi_BankReport_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      // Save to history
      const newReport = {
        id: Math.random().toString(36).substr(2, 9),
        date: new Date().toLocaleDateString(),
        crop: formData.crop,
        profit: formData.profit,
        sigId: "CERT-" + Math.random().toString(36).substr(2, 5).toUpperCase()
      };
      const updatedReports = [newReport, ...reports];
      setReports(updatedReports);
      localStorage.setItem("farm_reports", JSON.stringify(updatedReports));

    } catch (err) {
      console.error(err);
      const status = err?.response?.status;
      if (status === 401) {
        setError("You must be logged in to generate a report. Please sign in and try again.");
      } else if (status === 403) {
        setError("Report generation requires Expert or Admin role. Contact your administrator.");
      } else if (status === 429) {
        setError("Too many requests. Please wait a moment before trying again.");
      } else {
        setError("Failed to generate report. Please try again later.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="reports-container">
      <div className="reports-header">
        <FaFileInvoiceDollar className="header-icon" />
        <h1>Bank-Ready Financial Reports</h1>
        <p>Generate cryptographically signed reports for loan and subsidy applications.</p>
      </div>

      <div className="reports-grid">
        <div className="report-form-card">
          <div className="card-header">
            <FaShieldAlt />
            <h2>Certified Report Details</h2>
          </div>
          
          <div className="form-group">
            <label>Farmer Name (As per Bank A/C)</label>
            <input 
              type="text" 
              value={formData.name} 
              onChange={(e) => setFormData({...formData, name: e.target.value})}
              placeholder="e.g. Rajesh Kumar"
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Crop Type</label>
              <select value={formData.crop} onChange={(e) => setFormData({...formData, crop: e.target.value})}>
                <option>Wheat</option>
                <option>Rice</option>
                <option>Maize</option>
                <option>Sugarcane</option>
              </select>
            </div>
            <div className="form-group">
              <label>Farm Area</label>
              <input 
                type="text" 
                value={formData.area} 
                onChange={(e) => setFormData({...formData, area: e.target.value})}
                placeholder="e.g. 5 Acres"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Estimated Season Profit (₹)</label>
              <input 
                type="text" 
                value={formData.profit} 
                onChange={(e) => setFormData({...formData, profit: e.target.value})}
              />
            </div>
            <div className="form-group">
              <label>Current Season</label>
              <input 
                type="text" 
                value={formData.season} 
                onChange={(e) => setFormData({...formData, season: e.target.value})}
              />
            </div>
          </div>

          {error && <div className="error-msg">{error}</div>}

          <button 
            className={`generate-btn ${loading ? 'loading' : ''}`} 
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? <FaSpinner className="spin" /> : <FaDownload />}
            {loading ? "Generating Signature..." : "Download Certified Report"}
          </button>
          
          <p className="security-note">
            <FaCheckCircle /> This report will be cryptographically signed and cannot be edited after generation.
          </p>
        </div>

        <div className="report-history-card">
          <div className="card-header">
            <FaHistory />
            <h2>Recent Reports</h2>
          </div>
          <div className="history-list">
            {reports.length === 0 ? (
              <div className="empty-history">No reports generated yet.</div>
            ) : (
              reports.map(report => (
                <div key={report.id} className="history-item">
                  <div className="item-main">
                    <span className="item-crop">{report.crop} Report</span>
                    <span className="item-date">{report.date}</span>
                  </div>
                  <div className="item-details">
                    <span className="item-profit">₹{report.profit}</span>
                    <span className="item-sig">ID: {report.sigId}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BankReports;
