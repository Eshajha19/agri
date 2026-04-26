import React, { useState } from "react";
import { IndianRupee, TrendingUp, AlertCircle } from "lucide-react";
import "./CropProfitCalculator.css";

export default function CropProfitCalculator() {
  const [formData, setFormData] = useState({
    farmingCost: "",
    expectedYield: "",
    marketPrice: "",
  });

  const [results, setResults] = useState(null);
  const [errors, setErrors] = useState({});
  const [hasCalculated, setHasCalculated] = useState(false);

  const validateInputs = () => {
    const newErrors = {};

    if (!formData.farmingCost || formData.farmingCost <= 0) {
      newErrors.farmingCost = "Please enter a valid farming cost";
    }

    if (!formData.expectedYield || formData.expectedYield <= 0) {
      newErrors.expectedYield = "Please enter a valid expected yield";
    }

    if (!formData.marketPrice || formData.marketPrice <= 0) {
      newErrors.marketPrice = "Please enter a valid market price";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    // Clear error for this field when user starts typing
    if (errors[name]) {
      setErrors((prev) => ({
        ...prev,
        [name]: "",
      }));
    }
  };

  const calculateProfit = (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      return;
    }

    const cost = parseFloat(formData.farmingCost);
    const yield_ = parseFloat(formData.expectedYield);
    const price = parseFloat(formData.marketPrice);

    const revenue = yield_ * price;
    const profit = revenue - cost;

    setResults({
      cost,
      yield: yield_,
      price,
      revenue,
      profit,
      profitPercentage: ((profit / cost) * 100).toFixed(2),
    });

    setHasCalculated(true);
  };

  const handleReset = () => {
    setFormData({
      farmingCost: "",
      expectedYield: "",
      marketPrice: "",
    });
    setResults(null);
    setErrors({});
    setHasCalculated(false);
  };

  return (
    <div className="profit-calculator-page">
      <div className="calculator-container">
        {/* Header */}
        <div className="calculator-header">
          <div className="header-content">
            <TrendingUp className="header-icon" />
            <div>
              <h1>Crop Profit Calculator</h1>
              <p>Estimate your potential profit before choosing crops</p>
            </div>
          </div>
        </div>

        <div className="calculator-content">
          {/* Form Section */}
          <div className="calculator-form-section">
            <h2>Enter Your Details</h2>
            <form onSubmit={calculateProfit} className="calculator-form">
              {/* Farming Cost Input */}
              <div className="form-group">
                <label htmlFor="farmingCost">
                  <IndianRupee size={18} />
                  Cost of Farming
                </label>
                <div className="input-wrapper">
                  <span className="currency-prefix">₹</span>
                  <input
                    type="number"
                    id="farmingCost"
                    name="farmingCost"
                    placeholder="Enter total farming cost"
                    value={formData.farmingCost}
                    onChange={handleInputChange}
                    min="0"
                    step="0.01"
                    className={errors.farmingCost ? "input-error" : ""}
                  />
                </div>
                {errors.farmingCost && (
                  <div className="error-message">
                    <AlertCircle size={16} />
                    {errors.farmingCost}
                  </div>
                )}
              </div>

              {/* Expected Yield Input */}
              <div className="form-group">
                <label htmlFor="expectedYield">
                  <TrendingUp size={18} />
                  Expected Yield (in quintals)
                </label>
                <div className="input-wrapper">
                  <input
                    type="number"
                    id="expectedYield"
                    name="expectedYield"
                    placeholder="Enter expected yield"
                    value={formData.expectedYield}
                    onChange={handleInputChange}
                    min="0"
                    step="0.01"
                    className={errors.expectedYield ? "input-error" : ""}
                  />
                  <span className="unit-suffix">q</span>
                </div>
                {errors.expectedYield && (
                  <div className="error-message">
                    <AlertCircle size={16} />
                    {errors.expectedYield}
                  </div>
                )}
              </div>

              {/* Market Price Input */}
              <div className="form-group">
                <label htmlFor="marketPrice">
                  <IndianRupee size={18} />
                  Market Price (per quintal)
                </label>
                <div className="input-wrapper">
                  <span className="currency-prefix">₹</span>
                  <input
                    type="number"
                    id="marketPrice"
                    name="marketPrice"
                    placeholder="Enter market price per quintal"
                    value={formData.marketPrice}
                    onChange={handleInputChange}
                    min="0"
                    step="0.01"
                    className={errors.marketPrice ? "input-error" : ""}
                  />
                  <span className="unit-suffix">/q</span>
                </div>
                {errors.marketPrice && (
                  <div className="error-message">
                    <AlertCircle size={16} />
                    {errors.marketPrice}
                  </div>
                )}
              </div>

              {/* Buttons */}
              <div className="form-buttons">
                <button type="submit" className="btn-calculate">
                  Calculate Profit
                </button>
                {hasCalculated && (
                  <button type="button" onClick={handleReset} className="btn-reset">
                    Reset
                  </button>
                )}
              </div>
            </form>
          </div>

          {/* Results Section */}
          {hasCalculated && results && (
            <div className="calculator-results-section">
              <h2>Profit Analysis</h2>

              {/* Summary Cards */}
              <div className="results-grid">
                {/* Revenue Card */}
                <div className="result-card revenue-card">
                  <div className="card-label">Total Revenue</div>
                  <div className="card-value">₹{results.revenue.toFixed(2)}</div>
                  <div className="card-formula">
                    {results.yield.toFixed(2)}q × ₹{results.price.toFixed(2)}/q
                  </div>
                </div>

                {/* Cost Card */}
                <div className="result-card cost-card">
                  <div className="card-label">Total Cost</div>
                  <div className="card-value">₹{results.cost.toFixed(2)}</div>
                  <div className="card-formula">Farming Expenses</div>
                </div>

                {/* Profit/Loss Card */}
                <div
                  className={`result-card profit-card ${
                    results.profit >= 0 ? "profit-positive" : "profit-negative"
                  }`}
                >
                  <div className="card-label">
                    {results.profit >= 0 ? "Profit" : "Loss"}
                  </div>
                  <div className="card-value">
                    {results.profit >= 0 ? "+" : ""}₹{results.profit.toFixed(2)}
                  </div>
                  <div className="card-formula">
                    {results.profitPercentage}% return on investment
                  </div>
                </div>
              </div>

              {/* Breakdown Section */}
              <div className="breakdown-section">
                <h3>Detailed Breakdown</h3>
                <div className="breakdown-item">
                  <span className="breakdown-label">Farming Cost:</span>
                  <span className="breakdown-value">₹{results.cost.toFixed(2)}</span>
                </div>
                <div className="breakdown-item">
                  <span className="breakdown-label">Expected Yield:</span>
                  <span className="breakdown-value">{results.yield.toFixed(2)} quintals</span>
                </div>
                <div className="breakdown-item">
                  <span className="breakdown-label">Market Price:</span>
                  <span className="breakdown-value">₹{results.price.toFixed(2)}/quintal</span>
                </div>
                <div className="breakdown-divider"></div>
                <div className="breakdown-item total">
                  <span className="breakdown-label">Revenue:</span>
                  <span className="breakdown-value">₹{results.revenue.toFixed(2)}</span>
                </div>
                <div className="breakdown-item total">
                  <span className="breakdown-label">
                    {results.profit >= 0 ? "Total Profit" : "Total Loss"}:
                  </span>
                  <span className={`breakdown-value ${results.profit >= 0 ? "positive" : "negative"}`}>
                    {results.profit >= 0 ? "+" : ""}₹{results.profit.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Formula Info */}
              <div className="formula-info">
                <div className="formula-title">Formula Used:</div>
                <div className="formula-text">
                  <strong>Profit = (Yield × Market Price) - Cost</strong>
                </div>
                <div className="formula-calculation">
                  Profit = ({results.yield.toFixed(2)} × ₹{results.price.toFixed(2)}) - ₹{results.cost.toFixed(2)}
                </div>
                <div className="formula-result">
                  = ₹{results.revenue.toFixed(2)} - ₹{results.cost.toFixed(2)} = ₹{results.profit.toFixed(2)}
                </div>
              </div>

              {/* Recommendation */}
              <div
                className={`recommendation ${
                  results.profit >= 0 ? "positive-recommendation" : "negative-recommendation"
                }`}
              >
                {results.profit >= 0 ? (
                  <>
                    <h3>✅ This is a Profitable Crop Choice!</h3>
                    <p>
                      With a profit of ₹{results.profit.toFixed(2)}, this crop can generate a good return on your investment. 
                      This represents a {results.profitPercentage}% return on your farming cost.
                    </p>
                  </>
                ) : (
                  <>
                    <h3>⚠️ This Crop May Result in Loss</h3>
                    <p>
                      With a potential loss of ₹{Math.abs(results.profit).toFixed(2)}, you may want to reconsider this crop choice
                      or look for ways to increase yield or reduce farming costs.
                    </p>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
