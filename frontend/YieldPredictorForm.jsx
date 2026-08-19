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
import { BarChart3, Loader2, X } from "lucide-react";
import LastUpdated from "./LastUpdated";
import { useYieldPrediction } from "./hooks/useYieldPrediction";
import { useTranslation } from "react-i18next";

export default function YieldPredictorForm({ onClose }) {
  const { t } = useTranslation();
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
        <span className="notranslate"> {t("yieldPredictor.title")}</span>
      </h2>

      {yieldError && (
        <div className="yield-error-box" role="alert">
          {t("yieldPredictor.error")} {yieldError}
        </div>
      )}

      {yieldPrediction === null ? (
        <form onSubmit={fetchYield} className="yield-form">
          <div className="form-group">
            <label htmlFor="yf-crop">{t("yieldPredictor.crop")}</label>
            <select
              id="yf-crop"
              value={yieldForm.Crop}
              onChange={(e) => updateYieldFormField("Crop", e.target.value)}
            >
              <option value="Paddy">{t("yieldPredictor.crops.paddy")}</option>
              <option value="Cotton">{t("yieldPredictor.crops.cotton")}</option>
              <option value="Maize">{t("yieldPredictor.crops.maize")}</option>
              <option value="Bengal Gram">{t("yieldPredictor.crops.bengalGram")}</option>
              <option value="Groundnut">{t("yieldPredictor.crops.groundnut")}</option>
              <option value="Chillies">{t("yieldPredictor.crops.chillies")}</option>
              <option value="Red Gram">{t("yieldPredictor.crops.redGram")}</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-season">{t("yieldPredictor.season")}</label>
            <select
              id="yf-season"
              value={yieldForm.Season}
              onChange={(e) => updateYieldFormField("Season", e.target.value)}
            >
              <option value="Rabi">{t("yieldPredictor.seasons.rabi")}</option>
              <option value="Kharif">{t("yieldPredictor.seasons.kharif")}</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-area">{t("yieldPredictor.coveredArea")}</label>
            <input
              id="yf-area"
              type="number"
              min="0"
              step="0.1"
              value={yieldForm.CropCoveredArea}
              onChange={(e) =>
                updateYieldFormField("CropCoveredArea", e.target.value === "" ? 0 : parseFloat(e.target.value))
              }
            />
          </div>

          <div className="form-group">
            <label htmlFor="yf-height">{t("yieldPredictor.cropHeight")}</label>
            <input
              id="yf-height"
              type="number"
              min="0"
              value={yieldForm.CHeight}
              onChange={(e) =>
                updateYieldFormField("CHeight", e.target.value === "" ? 0 : parseInt(e.target.value))
              }
            />
          </div>

          <div className="form-group">
            <label htmlFor="yf-next">{t("yieldPredictor.nextCrop")}</label>
            <select
              id="yf-next"
              value={yieldForm.CNext}
              onChange={(e) => updateYieldFormField("CNext", e.target.value)}
            >
              <option value="Pea">{t("yieldPredictor.crops.pea")}</option>
              <option value="Lentil">{t("yieldPredictor.crops.lentil")}</option>
              <option value="Maize">{t("yieldPredictor.crops.maize")}</option>
              <option value="Sorghum">{t("yieldPredictor.crops.sorghum")}</option>
              <option value="Wheat">{t("yieldPredictor.crops.wheat")}</option>
              <option value="Soybean">{t("yieldPredictor.crops.soybean")}</option>
              <option value="Mustard">{t("yieldPredictor.crops.mustard")}</option>
              <option value="Rice">{t("yieldPredictor.crops.rice")}</option>
              <option value="Tomato">{t("yieldPredictor.crops.tomato")}</option>
              <option value="Onion">{t("yieldPredictor.crops.onion")}</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-last">{t("yieldPredictor.lastCrop")}</label>
            <select
              id="yf-last"
              value={yieldForm.CLast}
              onChange={(e) => updateYieldFormField("CLast", e.target.value)}
            >
              <option value="Lentil">{t("yieldPredictor.crops.lentil")}</option>
              <option value="Pea">{t("yieldPredictor.crops.pea")}</option>
              <option value="Maize">{t("yieldPredictor.crops.maize")}</option>
              <option value="Sorghum">{t("yieldPredictor.crops.sorghum")}</option>
              <option value="Soybean">{t("yieldPredictor.crops.soybean")}</option>
              <option value="Wheat">{t("yieldPredictor.crops.wheat")}</option>
              <option value="Mustard">{t("yieldPredictor.crops.mustard")}</option>
              <option value="Rice">{t("yieldPredictor.crops.rice")}</option>
              <option value="Tomato">{t("yieldPredictor.crops.tomato")}</option>
              <option value="Onion">{t("yieldPredictor.crops.onion")}</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-transp">{t("yieldPredictor.transplantingMethod")}</label>
            <select
              id="yf-transp"
              value={yieldForm.CTransp}
              onChange={(e) => updateYieldFormField("CTransp", e.target.value)}
            >
              <option value="Transplanting">{t("yieldPredictor.transplantingMethods.transplanting")}</option>
              <option value="Drilling">{t("yieldPredictor.transplantingMethods.drilling")}</option>
              <option value="Broadcasting">{t("yieldPredictor.transplantingMethods.broadcasting")}</option>
              <option value="Seed Drilling">{t("yieldPredictor.transplantingMethods.seedDrilling")}</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-irri-type">{t("yieldPredictor.irrigationType")}</label>
            <select
              id="yf-irri-type"
              value={yieldForm.IrriType}
              onChange={(e) => updateYieldFormField("IrriType", e.target.value)}
            >
              <option value="Flood">{t("yieldPredictor.irrigationTypes.flood")}</option>
              <option value="Sprinkler">{t("yieldPredictor.irrigationTypes.sprinkler")}</option>
              <option value="Drip">{t("yieldPredictor.irrigationTypes.drip")}</option>
              <option value="Surface">{t("yieldPredictor.irrigationTypes.surface")}</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-irri-src">{t("yieldPredictor.irrigationSource")}</label>
            <select
              id="yf-irri-src"
              value={yieldForm.IrriSource}
              onChange={(e) => updateYieldFormField("IrriSource", e.target.value)}
            >
              <option value="Groundwater">{t("yieldPredictor.irrigationSources.groundwater")}</option>
              <option value="Canal">{t("yieldPredictor.irrigationSources.canal")}</option>
              <option value="Rainfed">{t("yieldPredictor.irrigationSources.rainfed")}</option>
              <option value="Well">{t("yieldPredictor.irrigationSources.well")}</option>
              <option value="Tubewell">{t("yieldPredictor.irrigationSources.tubewell")}</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="yf-irri-count">{t("yieldPredictor.irrigationCount")}</label>
            <input
              id="yf-irri-count"
              type="number"
              min="0"
              value={yieldForm.IrriCount}
              onChange={(e) =>
                updateYieldFormField("IrriCount", e.target.value === "" ? 0 : parseInt(e.target.value))
              }
            />
          </div>

          <div className="form-group">
            <label htmlFor="yf-water-cov">{t("yieldPredictor.waterCoverage")}</label>
            <input
              id="yf-water-cov"
              type="number"
              min="0"
              max="100"
              value={yieldForm.WaterCov}
              onChange={(e) =>
                updateYieldFormField("WaterCov", e.target.value === "" ? 0 : parseInt(e.target.value))
              }
            />
          </div>

          <div className="form-group full-width form-actions">
            <button type="submit" className="action-btn" disabled={yieldLoading}>
              {yieldLoading ? (
                <>
                  <Loader2 className="spinner" size={18} aria-hidden="true" />
                  {t("yieldPredictor.predicting")}
                </>
              ) : (
                t("yieldPredictor.predictYield")
              )}
            </button>
            {onClose && (
              <button
                type="button"
                className="action-btn secondary"
                onClick={handleCancel}
              >
                {t("yieldPredictor.cancel")}
              </button>
            )}
          </div>
        </form>
      ) : (
        <div className="yield-result-block">
          <p className="yield-result">
            {t("yieldPredictor.predictedYield")}{" "}
            <strong>{yieldPrediction.toFixed(2)}</strong> {t("yieldPredictor.quintalsPerAcre")}
          </p>
          {yieldLastUpdated && (
            <div className="yield-updated">
              <LastUpdated timestamp={yieldLastUpdated} />
            </div>
          )}
          <button className="action-btn" onClick={closeYieldPopup}>
            {t("yieldPredictor.predictAnother")}
          </button>
        </div>
      )}
    </div>
  );
}
