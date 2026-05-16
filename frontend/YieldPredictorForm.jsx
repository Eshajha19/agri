/**
 * YieldPredictorForm — reusable yield prediction form.
 *
 * Used by:
 *  - YieldPredictor.jsx  (standalone /yield-predictor page)
 *  - Advisor.jsx         (modal popup — existing flow, unchanged)
 *
 * Props:
 *  - onClose  (optional) — called when the user clicks Cancel; omit on the
 *                          standalone page where there is no modal to close.
 */
import React from "react";
import { BarChart3, X } from "lucide-react";
import LastUpdated from "./LastUpdated";
import { useYieldPrediction } from "./hooks/useYieldPrediction";

export default function YieldPredictorForm({ onClose }) {
  const {
    yieldForm,
    updateYieldFormField,
    yieldPrediction,
    yieldLastUpdated,
    yieldError,
    yieldLoading,
    fetchYield,
    closeYieldPopup,
  } = useYieldPrediction();

  // When used inside the Advisor modal, closeYieldPopup resets state AND
  // closes the popup via the store. On the standalone page we only need to
  // reset state (no popup to close), so we call closeYieldPopup which is safe
  // in both contexts.
  const handleCancel = () => {
    closeYieldPopup();
    if (onClose) onClose();
  };

  return (
    <div className="yield-predictor-form-root">
      <h2 className="yield-form-title">
        <BarChart3 className="inline-icon" aria-hidden="true" />
        <span className="notranslate"> Yield Prediction</span>
      </h2>

      {yieldError && (
        <div className="yield-error-box" role="alert">
          Error: {yieldError}
        </div>
      )}

      {yieldPrediction === null ? (
        <form onSubmit={fetchYield} className="yield-form">
          <div className="form-group">
            <label htmlFor="yf-crop">Crop</label>
            <select
              id="yf-crop"
              value={yieldForm.Crop}
              onChange={(e) => updateYieldFormField("Crop", e.target.value)}
            >
              <option value="Paddy">Paddy</option>
              <option value="Cotton">Cotton</option>
              <option value="Maize">Maize</option>
              <option value="Bengal Gram">Bengal Gram</option>
              <option value="Groundnut">Groundnut</option>
              <option value="Chillies">Chillies</option>
              <option value="Red Gram">Red Gram</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-season">Season</label>
            <select
              id="yf-season"
              value={yieldForm.Season}
              onChange={(e) => updateYieldFormField("Season", e.target.value)}
            >
              <option value="Rabi">Rabi</option>
              <option value="Kharif">Kharif</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-area">Covered Area (acres)</label>
            <input
              id="yf-area"
              type="number"
              min="0"
              step="0.1"
              value={yieldForm.CropCoveredArea}
              onChange={(e) =>
                updateYieldFormField("CropCoveredArea", parseFloat(e.target.value))
              }
            />
          </div>

          <div className="form-group">
            <label htmlFor="yf-height">Crop Height (cm)</label>
            <input
              id="yf-height"
              type="number"
              min="0"
              value={yieldForm.CHeight}
              onChange={(e) =>
                updateYieldFormField("CHeight", parseInt(e.target.value))
              }
            />
          </div>

          <div className="form-group">
            <label htmlFor="yf-next">Next Crop</label>
            <select
              id="yf-next"
              value={yieldForm.CNext}
              onChange={(e) => updateYieldFormField("CNext", e.target.value)}
            >
              <option value="Pea">Pea</option>
              <option value="Lentil">Lentil</option>
              <option value="Maize">Maize</option>
              <option value="Sorghum">Sorghum</option>
              <option value="Wheat">Wheat</option>
              <option value="Soybean">Soybean</option>
              <option value="Mustard">Mustard</option>
              <option value="Rice">Rice</option>
              <option value="Tomato">Tomato</option>
              <option value="Onion">Onion</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-last">Last Crop</label>
            <select
              id="yf-last"
              value={yieldForm.CLast}
              onChange={(e) => updateYieldFormField("CLast", e.target.value)}
            >
              <option value="Lentil">Lentil</option>
              <option value="Pea">Pea</option>
              <option value="Maize">Maize</option>
              <option value="Sorghum">Sorghum</option>
              <option value="Soybean">Soybean</option>
              <option value="Wheat">Wheat</option>
              <option value="Mustard">Mustard</option>
              <option value="Rice">Rice</option>
              <option value="Tomato">Tomato</option>
              <option value="Onion">Onion</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-transp">Transplanting Method</label>
            <select
              id="yf-transp"
              value={yieldForm.CTransp}
              onChange={(e) => updateYieldFormField("CTransp", e.target.value)}
            >
              <option value="Transplanting">Transplanting</option>
              <option value="Drilling">Drilling</option>
              <option value="Broadcasting">Broadcasting</option>
              <option value="Seed Drilling">Seed Drilling</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-irri-type">Irrigation Type</label>
            <select
              id="yf-irri-type"
              value={yieldForm.IrriType}
              onChange={(e) => updateYieldFormField("IrriType", e.target.value)}
            >
              <option value="Flood">Flood</option>
              <option value="Sprinkler">Sprinkler</option>
              <option value="Drip">Drip</option>
              <option value="Surface">Surface</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-irri-src">Irrigation Source</label>
            <select
              id="yf-irri-src"
              value={yieldForm.IrriSource}
              onChange={(e) => updateYieldFormField("IrriSource", e.target.value)}
            >
              <option value="Groundwater">Groundwater</option>
              <option value="Canal">Canal</option>
              <option value="Rainfed">Rainfed</option>
              <option value="Well">Well</option>
              <option value="Tubewell">Tubewell</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-irri-count">Irrigation Count</label>
            <input
              id="yf-irri-count"
              type="number"
              min="0"
              value={yieldForm.IrriCount}
              onChange={(e) =>
                updateYieldFormField("IrriCount", parseInt(e.target.value))
              }
            />
          </div>

          <div className="form-group">
            <label htmlFor="yf-water-cov">Water Coverage (%)</label>
            <input
              id="yf-water-cov"
              type="number"
              min="0"
              max="100"
              value={yieldForm.WaterCov}
              onChange={(e) =>
                updateYieldFormField("WaterCov", parseInt(e.target.value))
              }
            />
          </div>

          <div className="form-group full-width form-actions">
            <button type="submit" className="action-btn" disabled={yieldLoading}>
              {yieldLoading ? "Predicting…" : "Predict Yield"}
            </button>
            {onClose && (
              <button
                type="button"
                className="action-btn secondary"
                onClick={handleCancel}
              >
                Cancel
              </button>
            )}
          </div>
        </form>
      ) : (
        <div className="yield-result-block">
          <p className="yield-result">
            Predicted Yield:{" "}
            <strong>{yieldPrediction.toFixed(2)}</strong> quintals/acre
          </p>
          {yieldLastUpdated && (
            <div className="yield-updated">
              <LastUpdated timestamp={yieldLastUpdated} />
            </div>
          )}
          <button className="action-btn" onClick={closeYieldPopup}>
            Predict Another
          </button>
        </div>
      )}
    </div>
  );
}
