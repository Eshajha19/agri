import React, { useState, useRef, useEffect } from "react";
import { FaFileInvoiceDollar, FaDownload, FaShieldAlt, FaCheckCircle, FaSpinner, FaHistory, FaExclamationTriangle } from "react-icons/fa";
import "./BankReports.css";
import apiClient from "./lib/apiClient";
import { loadVersionedArray, saveVersionedArray } from "./utils/versionedStorage";

const REPORTS_STORAGE_KEY = "farm_reports";
const REPORTS_STORAGE_VERSION = 1;
const MAX_REPORTS = 50;

// Validation bounds — these are intentionally generous to accommodate
// large commercial farms while still preventing obviously fabricated figures.
const PROFIT_MIN_INR = 0;
const PROFIT_MAX_INR = 50_000_000;   // ₹5 crore — upper bound for a single season
const AREA_MIN_ACRES = 0.1;
const AREA_MAX_ACRES = 10_000;       // 10,000 acres — large but plausible

const BankReports = ({ userData }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [reports, setReports] = useState(() =>
    loadVersionedArray(REPORTS_STORAGE_KEY, {
      version: REPORTS_STORAGE_VERSION,
      fallback: [],
      maxItems: MAX_REPORTS,
    })
  );

  const mountedRef = useRef(true);
  const activeRequestRef = useRef(0);
  const downloadUrlRef = useRef(null);

  const [formData, setFormData] = useState({
    name: userData?.displayName || "",
    crop: userData?.cropType || "Wheat",
    areaAcres: "",          // numeric acres — replaces free-text "area"
    profitInr: "",          // numeric INR — replaces free-text "profit"
    season: "Kharif 2026"
  });
  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;

      activeRequestRef.current += 1;

      if (downloadUrlRef.current) {
        window.URL.revokeObjectURL(downloadUrlRef.current);
        downloadUrlRef.current = null;
      }
    };
  }, []);

  // Validate all fields and return true only if everything is within bounds.
  const validateForm = () => {
    const errors = {};

    if (!formData.name.trim()) {
      errors.name = "Farmer name is required.";
    }

    const profit = parseFloat(formData.profitInr);
    if (formData.profitInr === "" || isNaN(profit)) {
      errors.profitInr = "Please enter a numeric profit amount.";
    } else if (profit < PROFIT_MIN_INR) {
      errors.profitInr = "Profit cannot be negative.";
    } else if (profit > PROFIT_MAX_INR) {
      errors.profitInr = `Profit cannot exceed ₹${PROFIT_MAX_INR.toLocaleString("en-IN")} per season.`;
    }

    const area = parseFloat(formData.areaAcres);
    if (formData.areaAcres === "" || isNaN(area)) {
      errors.areaAcres = "Please enter a numeric farm area.";
    } else if (area < AREA_MIN_ACRES) {
      errors.areaAcres = `Farm area must be at least ${AREA_MIN_ACRES} acres.`;
    } else if (area > AREA_MAX_ACRES) {
      errors.areaAcres = `Farm area cannot exceed ${AREA_MAX_ACRES.toLocaleString()} acres.`;
    }

    if (!formData.season.trim()) {
      errors.season = "Season is required.";
    }

    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleGenerate = async () => {
    setError("");
    if (!validateForm()) return;

    setLoading(true);
    const requestId = Date.now();
    activeRequestRef.current = requestId;
    try {
      // Send validated, typed values to the backend.
      // profit and area are sent as formatted strings matching the backend
      // ReportRequest schema, but derived from validated numeric inputs.
      const profit = parseFloat(formData.profitInr);
      const area = parseFloat(formData.areaAcres);

      const payload = {
        name: formData.name.trim(),
        crop: formData.crop,
        area: `${area.toLocaleString("en-IN")} Acres`,
        profit: profit.toLocaleString("en-IN"),
        season: formData.season.trim(),
      };

      const response = await apiClient.post(
        "/api/reports/generate",
        payload,
        { responseType: "blob" }
      );

      if (
        !mountedRef.current ||
        activeRequestRef.current !== requestId
      ) {
        return;
      }

      const blob = response.data;

      if (!blob || blob.size === 0) {
        throw new Error("Empty report received from server");
      }

      const MAX_REPORT_SIZE_MB = 25;

      if (blob.size > MAX_REPORT_SIZE_MB * 1024 * 1024) {
        throw new Error(
          `Generated report exceeds ${MAX_REPORT_SIZE_MB}MB size limit`
        );
      }

      if (downloadUrlRef.current) {
        window.URL.revokeObjectURL(downloadUrlRef.current);
      }

      const url = window.URL.createObjectURL(blob);
      downloadUrlRef.current = url;
      const a = document.createElement("a");
      a.href = url;
      a.download = `FasalSaathi_BankReport_${Date.now()}.pdf`;
      document.body.appendChild(a);
      try {
        a.click();
      } catch {
        throw new Error("Failed to start report download");
      }
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      if (downloadUrlRef.current === url) {
        downloadUrlRef.current = null;
      }

      const newReport = {
        id: Math.random().toString(36).substr(2, 9),
        date: new Date().toLocaleDateString(),
        crop: formData.crop,
        profit: profit.toLocaleString("en-IN"),
        sigId: "CERT-" + Math.random().toString(36).substr(2, 5).toUpperCase()
      };
      const updatedReports = [newReport, ...reports].slice(0, MAX_REPORTS);
      if (
        !mountedRef.current ||
        activeRequestRef.current !== requestId
      ) {
        return;
      }
      setReports(updatedReports);
      const saved = saveVersionedArray(REPORTS_STORAGE_KEY, updatedReports, {
        version: REPORTS_STORAGE_VERSION,
        maxItems: MAX_REPORTS,
      });
      if (!saved) {
        setError("Report history is full. Older reports were kept in memory only.");
      }

    } catch (err) {
      console.error(err);

      if (
        !mountedRef.current ||
        activeRequestRef.current !== requestId
      ) {
        return;
      }

      if (err.code === "ECONNABORTED") {
        setError(
          "Report generation timed out. Please try again with a smaller dataset."
        );
        return;
      }

      const status = err?.response?.status;

      if (status === 401) {
        setError(
          "You must be logged in to generate a report. Please sign in and try again."
        );
      } else if (status === 403) {
        setError(
          "Report generation requires Expert or Admin role. Contact your administrator."
        );
      } else if (status === 422) {
        setError(
          "The submitted values were rejected by the server. Please check your inputs."
        );
      } else if (status === 429) {
        setError(
          "Too many requests. Please wait a moment before trying again."
        );
      } else {
        setError(
          "Failed to generate report. Please try again later."
        );
      }
    } finally {
      if (
        mountedRef.current &&
        activeRequestRef.current === requestId
      ) {
        setLoading(false);
      }
    }
    };
    
    const field = (key, value, onChange) => ({
    value,
    onChange: (e) => {
      onChange(e);
      if (fieldErrors[key]) setFieldErrors(prev => ({ ...prev, [key]: undefined }));
    },
  });

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

          {/* Disclaimer — shown before the form so users understand the limitation */}
          <div className="report-disclaimer" style={{
            padding: "10px 14px", marginBottom: "16px",
            background: "#fffbeb", border: "1px solid #fde68a",
            borderRadius: "8px", fontSize: "0.82rem", color: "#92400e",
            display: "flex", gap: "8px", alignItems: "flex-start"
          }}>
            <FaExclamationTriangle style={{ marginTop: "2px", flexShrink: 0 }} />
            <span>
              <strong>Important:</strong> The cryptographic signature on this report confirms
              the document has not been altered after generation. It does <strong>not</strong> independently
              verify the accuracy of the figures you enter. Enter only accurate, verifiable data.
            </span>
          </div>

          <div className="form-group">
            <label>Farmer Name (As per Bank A/C)</label>
            <input
              type="text"
              {...field("name", formData.name, (e) => setFormData({...formData, name: e.target.value}))}
              placeholder="e.g. Rajesh Kumar"
              maxLength={100}
            />
            {fieldErrors.name && <p className="field-error">{fieldErrors.name}</p>}
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
              <label>Farm Area (Acres)</label>
              <input
                type="number"
                {...field("areaAcres", formData.areaAcres, (e) => setFormData({...formData, areaAcres: e.target.value}))}
                placeholder={`e.g. 5  (max ${AREA_MAX_ACRES.toLocaleString()})`}
                min={AREA_MIN_ACRES}
                max={AREA_MAX_ACRES}
                step="0.1"
              />
              {fieldErrors.areaAcres && <p className="field-error">{fieldErrors.areaAcres}</p>}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Estimated Season Profit (₹)</label>
              <input
                type="number"
                {...field("profitInr", formData.profitInr, (e) => setFormData({...formData, profitInr: e.target.value}))}
                placeholder={`e.g. 50000  (max ₹${PROFIT_MAX_INR.toLocaleString("en-IN")})`}
                min={PROFIT_MIN_INR}
                max={PROFIT_MAX_INR}
                step="1"
              />
              {fieldErrors.profitInr && <p className="field-error">{fieldErrors.profitInr}</p>}
            </div>
            <div className="form-group">
              <label>Current Season</label>
              <input
                type="text"
                {...field("season", formData.season, (e) => setFormData({...formData, season: e.target.value}))}
                placeholder="e.g. Kharif 2026"
                maxLength={50}
              />
              {fieldErrors.season && <p className="field-error">{fieldErrors.season}</p>}
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
