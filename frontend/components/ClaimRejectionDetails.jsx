import React from "react";
import { FaExclamationCircle, FaArrowRight, FaRedo, FaPhoneAlt } from "react-icons/fa";
import "./ClaimRejectionDetails.css";

/**
 * ClaimRejectionDetails Component
 * Displays detailed explanation and corrective actions when a crop insurance claim is rejected.
 *
 * Props:
 * - claimId: string (ID of the claim)
 * - rejectionDetails: Array of { reason: string, recommended_action: string }
 * - onResubmitClick: function (callback when farmer clicks to try again)
 */
export default function ClaimRejectionDetails({
  claimId = "",
  rejectionDetails = [],
  onResubmitClick = null,
}) {
  // Default fallback rejection reasons if none are provided
  const reasons = rejectionDetails.length > 0 ? rejectionDetails : [
    {
      reason: "Missing land ownership proof",
      recommended_action: "Upload Khasra document (Land record details)",
    }
  ];

  return (
    <div className="rejection-detail-card animate-fade-in" id={`rejection-details-${claimId}`}>
      {/* Header section with high-quality styling */}
      <div className="rejection-header">
        <div className="rejection-badge">
          <span className="badge-icon">❌</span>
          <span className="badge-text">Claim Rejected</span>
        </div>
        <h2 className="rejection-title">Claim Rejection Explanation</h2>
        <p className="rejection-subtitle">
          Your claim <span className="claim-id-text">#{claimId}</span> could not be processed. 
          Please review the reasons and recommended actions below to correct and resubmit your claim.
        </p>
      </div>

      {/* Rejection Reasons & Actions Grid */}
      <div className="rejection-list">
        {reasons.map((item, idx) => (
          <div key={idx} className="rejection-item" id={`rejection-item-${idx}`}>
            <div className="rejection-item-icon">
              <FaExclamationCircle />
            </div>
            <div className="rejection-item-content">
              <div className="rejection-reason-block">
                <span className="rejection-label">Reason:</span>
                <p className="rejection-text">{item.reason}</p>
              </div>
              <div className="rejection-action-block">
                <span className="action-label">Recommended Action:</span>
                <p className="action-text">
                  <FaArrowRight className="action-arrow" /> {item.recommended_action}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Helpful alert block */}
      <div className="rejection-help-alert">
        <p>
          💡 <strong>Need Help?</strong> You can contact our agricultural desk at 
          <strong> 1800-XXX-XXXX</strong> or update your documents in the profile settings 
          before trying again.
        </p>
      </div>

      {/* Button Actions */}
      <div className="rejection-actions">
        {onResubmitClick && (
          <button
            id="btn-rejection-resubmit"
            className="btn-rejection btn-rejection-primary"
            onClick={onResubmitClick}
            aria-label="Edit and resubmit claim details"
          >
            <FaRedo className="btn-icon" /> Resubmit / Correct Claim
          </button>
        )}
        <a
          href="tel:1800000000"
          className="btn-rejection btn-rejection-secondary"
          aria-label="Call support helpline"
        >
          <FaPhoneAlt className="btn-icon" /> Call Helpline
        </a>
      </div>
    </div>
  );
}
