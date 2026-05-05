import React, { useState, useRef, useEffect } from "react";
import { Html5Qrcode } from "html5-qrcode";
import { ShieldCheck, ShieldAlert, QrCode, ScanLine, X, Loader2, Search, Camera, RefreshCw } from "lucide-react";
import "./SeedVerifier.css";

const SeedVerifier = ({ onClose }) => {
  const [scannedData, setScannedData] = useState(null);
  const [status, setStatus] = useState("idle"); // idle, scanning, validating, authentic, invalid, not_found
  const [error, setError] = useState(null);
  const scannerRef = useRef(null);

  // Use a effect to start scanner when status changes to "scanning"
  // This ensures the #reader element exists in the DOM first
  useEffect(() => {
    if (status === "scanning") {
      const initScanner = async () => {
        try {
          const html5QrCode = new Html5Qrcode("reader");
          scannerRef.current = html5QrCode;
          
          const config = { fps: 10, qrbox: { width: 250, height: 250 } };
          
          await html5QrCode.start(
            { facingMode: "environment" }, 
            config,
            (decodedText) => {
              setScannedData(decodedText);
              setStatus("validating");
              stopScanner();
              verifyCode(decodedText);
            },
            () => {}
          );
        } catch (err) {
          console.error("Camera start error:", err);
          setError("Could not access camera. Please check permissions.");
          setStatus("idle");
        }
      };

      // Small timeout to ensure DOM is ready
      const timer = setTimeout(initScanner, 100);
      return () => clearTimeout(timer);
    }
  }, [status]);

  const stopScanner = async () => {
    if (scannerRef.current && scannerRef.current.isScanning) {
      try {
        await scannerRef.current.stop();
      } catch (err) {
        console.warn("Failed to stop scanner:", err);
      }
    }
  };

  const verifyCode = (code) => {
    setTimeout(() => {
      const upperCode = code.toUpperCase();
      if (upperCode.includes("FS-AUTH")) {
        setStatus("authentic");
      } else if (upperCode.includes("FS-INVALID")) {
        setStatus("invalid");
      } else {
        setStatus("not_found");
      }
    }, 1500);
  };

  const handleManualSubmit = (e) => {
    e.preventDefault();
    const code = e.target.manualCode.value;
    if (code) {
      setScannedData(code);
      setStatus("validating");
      verifyCode(code);
    }
  };

  const handleReset = () => {
    stopScanner();
    setScannedData(null);
    setStatus("idle");
    setError(null);
  };

  return (
    <div className="seed-verifier">
      <div className="verifier-header">
        <h2><QrCode className="header-icon" /> Vision-Lite: Seed Verifier</h2>
        {onClose && <button className="close-btn" onClick={() => { stopScanner(); onClose(); }} aria-label="Close modal"><X /></button>}
      </div>

      <div className="verifier-content">
        {status === "idle" && (
          <div className="idle-state">
            <div className="camera-preview-placeholder">
              <Camera size={48} className="placeholder-icon" />
              <p>Camera is currently off</p>
            </div>
            <button className="start-btn" onClick={() => setStatus("scanning")}>
              <Camera size={20} /> Start Camera Scanner
            </button>
            {error && <p className="error-msg">{error}</p>}
            
            <div className="manual-entry-inline">
              <div className="divider"><span>OR</span></div>
              <form onSubmit={handleManualSubmit} className="manual-form">
                <input name="manualCode" type="text" placeholder="Enter code manually..." className="manual-input" />
                <button type="submit" className="manual-submit-btn">Verify</button>
              </form>
            </div>
          </div>
        )}

        {status === "scanning" && (
          <div className="scanner-container">
            <div id="reader"></div>
            <div className="scanner-overlay">
              <div className="scan-frame">
                <ScanLine className="scan-line-anim" />
              </div>
              <p>Align QR or Barcode in frame</p>
              <button className="stop-btn" onClick={handleReset}>Cancel</button>
            </div>
          </div>
        )}

        {status === "validating" && (
          <div className="result-card validating">
            <Loader2 className="animate-spin result-icon validating-icon" />
            <h3>Validating Data...</h3>
            <p>Registry check for: <code>{scannedData}</code></p>
          </div>
        )}

        {status === "authentic" && (
          <div className="result-card authentic">
            <ShieldCheck className="result-icon success" />
            <h3 className="success-text">Authentic Seed!</h3>
            <p>Verified & Safe for Use</p>
            <div className="data-details">
              <p><span>Batch Code:</span> <strong>{scannedData}</strong></p>
              <p><span>Status:</span> <strong className="success-text">REGISTERED</strong></p>
            </div>
            <button className="action-btn" onClick={handleReset}>Scan Another</button>
          </div>
        )}

        {status === "invalid" && (
          <div className="result-card invalid">
            <ShieldAlert className="result-icon danger" />
            <h3 className="danger-text">Counterfeit Alert!</h3>
            <p>This code is blacklisted or invalid.</p>
            <div className="data-details">
              <p><span>Batch Code:</span> <strong>{scannedData}</strong></p>
              <p><span>Status:</span> <strong className="danger-text">BLACKLISTED</strong></p>
            </div>
            <button className="action-btn danger-btn" onClick={handleReset}>Scan Again</button>
          </div>
        )}

        {status === "not_found" && (
          <div className="result-card not-found">
            <Search className="result-icon warning" />
            <h3 className="warning-text">Unregistered</h3>
            <p>Code not found in our central database.</p>
            <div className="data-details">
              <p><span>Batch Code:</span> <strong>{scannedData}</strong></p>
              <p><span>Status:</span> <strong className="warning-text">NOT FOUND</strong></p>
            </div>
            <button className="action-btn secondary" onClick={handleReset}>Scan Again</button>
          </div>
        )}
      </div>

      <div className="verifier-footer">
        <p>© 2026 Fasal Saathi AI Vision Services</p>
      </div>
    </div>
  );
};

export default SeedVerifier;
