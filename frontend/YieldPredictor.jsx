/**
 * YieldPredictor — standalone page accessible at /yield-predictor.
 *
 * Renders the shared YieldPredictorForm component inside a full-page layout.
 * The Advisor modal flow is completely unaffected.
 */
import React from "react";
import { useNavigate } from "react-router-dom";
import { X } from "lucide-react";
import YieldPredictorForm from "./YieldPredictorForm";
import "./YieldPredictor.css";

export default function YieldPredictor() {
  const navigate = useNavigate();

  return (
    <div className="yield-predictor-page">
      <div className="yield-predictor-container">
        <button
          className="yield-predictor-back-btn"
          onClick={() => navigate(-1)}
          aria-label="Go back"
        >
          <X size={20} aria-hidden="true" />
        </button>

        {/* Reusable form — no onClose prop means no Cancel button */}
        <YieldPredictorForm />
      </div>
    </div>
  );
}
