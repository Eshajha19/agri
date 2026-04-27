import React, { useState } from "react";
import { IndianRupee, TrendingUp, AlertCircle, Plus, Trash2, Trophy } from "lucide-react";
import "./CropProfitCalculator.css";

export default function CropProfitCalculator() {
  const [isCompareMode, setIsCompareMode] = useState(false);

  const [formData, setFormData] = useState({
    farmingCost: "",
    expectedYield: "",
    marketPrice: "",
  });

  const [compareData, setCompareData] = useState([
    { id: 1, cropName: "", farmingCost: "", expectedYield: "", marketPrice: "" },
    { id: 2, cropName: "", farmingCost: "", expectedYield: "", marketPrice: "" },
  ]);

  const [results, setResults] = useState(null);
  const [compareResults, setCompareResults] = useState(null);
  const [errors, setErrors] = useState({});
  const [hasCalculated, setHasCalculated] = useState(false);

  const validateInputs = () => {
    const newErrors = {};

    if (isCompareMode) {
      compareData.forEach((crop, index) => {
        if (!crop.cropName.trim()) newErrors[`cropName-${index}`] = "Required";
        if (!crop.farmingCost || crop.farmingCost <= 0) newErrors[`farmingCost-${index}`] = "Required";
        if (!crop.expectedYield || crop.expectedYield <= 0) newErrors[`expectedYield-${index}`] = "Required";
        if (!crop.marketPrice || crop.marketPrice <= 0) newErrors[`marketPrice-${index}`] = "Required";
      });
    } else {
      if (!formData.farmingCost || formData.farmingCost <= 0) {
        newErrors.farmingCost = "Please enter a valid farming cost";
      }

      if (!formData.expectedYield || formData.expectedYield <= 0) {
        newErrors.expectedYield = "Please enter a valid expected yield";
      }

      if (!formData.marketPrice || formData.marketPrice <= 0) {
        newErrors.marketPrice = "Please enter a valid market price";
      }
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
    if (errors[name]) {
      setErrors((prev) => ({
        ...prev,
        [name]: "",
      }));
    }
  };

  const handleCompareInputChange = (index, e) => {
    const { name, value } = e.target;
    const newData = [...compareData];
    newData[index][name] = value;
    setCompareData(newData);
    
    if (errors[`${name}-${index}`]) {
      setErrors((prev) => ({ ...prev, [`${name}-${index}`]: "" }));
    }
  };

  const addCrop = () => {
    if (compareData.length < 5) {
      setCompareData([...compareData, { id: Date.now(), cropName: "", farmingCost: "", expectedYield: "", marketPrice: "" }]);
    }
  };

  const removeCrop = (index) => {
    if (compareData.length > 2) {
      const newData = compareData.filter((_, i) => i !== index);
      setCompareData(newData);
    }
  };

  const toggleMode = (mode) => {
    if (mode === 'compare' && !isCompareMode) {
      setIsCompareMode(true);
      setHasCalculated(false);
      setErrors({});
    } else if (mode === 'single' && isCompareMode) {
      setIsCompareMode(false);
      setHasCalculated(false);
      setErrors({});
    }
  };

  const calculateProfit = (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      return;
    }

    if (isCompareMode) {
      const resultsArray = compareData.map((crop) => {
        const cost = parseFloat(crop.farmingCost);
        const yield_ = parseFloat(crop.expectedYield);
        const price = parseFloat(crop.marketPrice);
        const revenue = yield_ * price;
        const profit = revenue - cost;
        const profitPercentage = ((profit / cost) * 100).toFixed(2);
        
        return {
          id: crop.id,
          name: crop.cropName,
          cost,
          yield: yield_,
          price,
          revenue,
          profit,
          profitPercentage,
        };
      });

      const maxProfit = Math.max(...resultsArray.map((r) => r.profit));
      const bestCropIds = resultsArray.filter(r => r.profit === maxProfit && r.profit > 0).map(r => r.id);
      
      setCompareResults({ data: resultsArray, bestCropIds });
    } else {
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
    }

    setHasCalculated(true);
  };

  const handleReset = () => {
    if (isCompareMode) {
      setCompareData([
        { id: Date.now(), cropName: "", farmingCost: "", expectedYield: "", marketPrice: "" },
        { id: Date.now() + 1, cropName: "", farmingCost: "", expectedYield: "", marketPrice: "" },
      ]);
      setCompareResults(null);
    } else {
      setFormData({
        farmingCost: "",
        expectedYield: "",
        marketPrice: "",
      });
      setResults(null);
    }
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

        {/* Mode Toggle */}
        <div className="mode-toggle-container">
          <div className="mode-toggle">
            <button 
              type="button"
              className={`toggle-btn ${!isCompareMode ? 'active' : ''}`} 
              onClick={() => toggleMode('single')}
            >
              Single Crop
            </button>
            <button 
              type="button"
              className={`toggle-btn ${isCompareMode ? 'active' : ''}`} 
              onClick={() => toggleMode('compare')}
            >
              Compare Crops
            </button>
          </div>
        </div>

        <div className={`calculator-content ${isCompareMode ? 'compare-mode-active' : ''}`}>
          {/* Form Section */}
          <div className="calculator-form-section">
            <h2>{isCompareMode ? "Enter Crops to Compare" : "Enter Your Details"}</h2>
            <form onSubmit={calculateProfit} className="calculator-form">
              
              {!isCompareMode ? (
                <>
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
                </>
              ) : (
                <div className="compare-form-container">
                  {compareData.map((crop, index) => (
                    <div key={crop.id} className="compare-crop-row">
                      <div className="crop-row-header">
                        <h3>Crop {index + 1}</h3>
                        {compareData.length > 2 && (
                          <button type="button" className="btn-remove-crop" onClick={() => removeCrop(index)} title="Remove Crop">
                            <Trash2 size={18} />
                          </button>
                        )}
                      </div>
                      <div className="compare-inputs-grid">
                        <div className="form-group">
                          <label>Crop Name</label>
                          <input
                            type="text"
                            name="cropName"
                            placeholder="e.g., Wheat"
                            value={crop.cropName}
                            onChange={(e) => handleCompareInputChange(index, e)}
                            className={`compare-input ${errors[`cropName-${index}`] ? "input-error" : ""}`}
                          />
                        </div>
                        <div className="form-group">
                          <label>Cost (₹)</label>
                          <input
                            type="number"
                            name="farmingCost"
                            placeholder="Total Cost"
                            value={crop.farmingCost}
                            onChange={(e) => handleCompareInputChange(index, e)}
                            min="0" step="0.01"
                            className={`compare-input ${errors[`farmingCost-${index}`] ? "input-error" : ""}`}
                          />
                        </div>
                        <div className="form-group">
                          <label>Yield (q)</label>
                          <input
                            type="number"
                            name="expectedYield"
                            placeholder="Yield"
                            value={crop.expectedYield}
                            onChange={(e) => handleCompareInputChange(index, e)}
                            min="0" step="0.01"
                            className={`compare-input ${errors[`expectedYield-${index}`] ? "input-error" : ""}`}
                          />
                        </div>
                        <div className="form-group">
                          <label>Price (₹/q)</label>
                          <input
                            type="number"
                            name="marketPrice"
                            placeholder="Market Price"
                            value={crop.marketPrice}
                            onChange={(e) => handleCompareInputChange(index, e)}
                            min="0" step="0.01"
                            className={`compare-input ${errors[`marketPrice-${index}`] ? "input-error" : ""}`}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {compareData.length < 5 && (
                    <button type="button" className="btn-add-crop" onClick={addCrop}>
                      <Plus size={18} /> Add Another Crop
                    </button>
                  )}
                </div>
              )}

              {/* Buttons */}
              <div className="form-buttons">
                <button type="submit" className="btn-calculate">
                  {isCompareMode ? "Compare Profits" : "Calculate Profit"}
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
          {hasCalculated && (
            <div className="calculator-results-section">
              <h2>{isCompareMode ? "Comparison Results" : "Profit Analysis"}</h2>

              {!isCompareMode && results && (
                <>
                  {/* Summary Cards */}
                  <div className="results-grid">
                    <div className="result-card revenue-card">
                      <div className="card-label">Total Revenue</div>
                      <div className="card-value">₹{results.revenue.toFixed(2)}</div>
                      <div className="card-formula">
                        {results.yield.toFixed(2)}q × ₹{results.price.toFixed(2)}/q
                      </div>
                    </div>
                    <div className="result-card cost-card">
                      <div className="card-label">Total Cost</div>
                      <div className="card-value">₹{results.cost.toFixed(2)}</div>
                      <div className="card-formula">Farming Expenses</div>
                    </div>
                    <div className={`result-card profit-card ${results.profit >= 0 ? "profit-positive" : "profit-negative"}`}>
                      <div className="card-label">{results.profit >= 0 ? "Profit" : "Loss"}</div>
                      <div className="card-value">
                        {results.profit >= 0 ? "+" : ""}₹{results.profit.toFixed(2)}
                      </div>
                      <div className="card-formula">{results.profitPercentage}% return on investment</div>
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

                  {/* Recommendation */}
                  <div className={`recommendation ${results.profit >= 0 ? "positive-recommendation" : "negative-recommendation"}`}>
                    {results.profit >= 0 ? (
                      <>
                        <h3>✅ This is a Profitable Crop Choice!</h3>
                        <p>With a profit of ₹{results.profit.toFixed(2)}, this crop can generate a good return on your investment. This represents a {results.profitPercentage}% return on your farming cost.</p>
                      </>
                    ) : (
                      <>
                        <h3>⚠️ This Crop May Result in Loss</h3>
                        <p>With a potential loss of ₹{Math.abs(results.profit).toFixed(2)}, you may want to reconsider this crop choice or look for ways to increase yield or reduce farming costs.</p>
                      </>
                    )}
                  </div>
                </>
              )}

              {isCompareMode && compareResults && (
                <div className="compare-results-container">
                  <div className="table-responsive">
                    <table className="comparison-table">
                      <thead>
                        <tr>
                          <th>Crop</th>
                          <th>Cost</th>
                          <th>Revenue</th>
                          <th>Profit/Loss</th>
                          <th>ROI</th>
                        </tr>
                      </thead>
                      <tbody>
                        {compareResults.data.map((res) => {
                          const isWinner = compareResults.bestCropIds.includes(res.id);
                          return (
                            <tr key={res.id} className={isWinner ? "winner-row" : ""}>
                              <td>
                                <div className="crop-name-cell">
                                  {res.name}
                                  {isWinner && <Trophy className="winner-icon" size={16} title="Most Profitable" />}
                                </div>
                              </td>
                              <td>₹{res.cost.toFixed(2)}</td>
                              <td>₹{res.revenue.toFixed(2)}</td>
                              <td className={res.profit >= 0 ? "text-positive" : "text-negative"}>
                                {res.profit >= 0 ? "+" : ""}₹{res.profit.toFixed(2)}
                              </td>
                              <td className={res.profit >= 0 ? "text-positive" : "text-negative"}>
                                {res.profitPercentage}%
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  
                  {compareResults.bestCropIds.length > 0 && (
                    <div className="recommendation positive-recommendation mt-4">
                      <h3><Trophy size={18} className="inline-icon" style={{ display: 'inline', marginRight: '8px', verticalAlign: 'text-bottom' }} /> Best Choice</h3>
                      <p>
                        Based on the comparison, <strong>{compareResults.data.filter(r => compareResults.bestCropIds.includes(r.id)).map(r => r.name).join(", ")}</strong> {compareResults.bestCropIds.length > 1 ? "are the most profitable options" : "is the most profitable option"}.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
