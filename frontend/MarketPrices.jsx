import React, { useState, useEffect, useMemo } from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, ReferenceLine
} from "recharts";
import { 
  TrendingUp, TrendingDown, Search, Filter, MapPin, Calendar, Info, Bell,
  ArrowRight, ArrowLeft, RefreshCw, BarChart3, Sparkles
} from "lucide-react";
import { fetchMarketPrices, fetchPriceTrends, getUniqueStates, getUniqueCommodities } from "./lib/marketApi";
import { fetchPriceForecast, FORECASTABLE_COMMODITIES } from "./services/marketForecastApi";
import "./MarketPrices.css";
import Loader from "./Loader";
import LastUpdated from "./LastUpdated";

const MarketPrices = () => {
  const [prices, setPrices] = useState([]);
  const [trends, setTrends] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({ state: "All", commodity: "All", search: "" });
  const [selectedCommodity, setSelectedCommodity] = useState("Wheat");
  const [activeTab, setActiveTab] = useState("list");
  const [lastUpdated, setLastUpdated] = useState(null);

  // Forecast state
  const [forecast, setForecast] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState(null);
  const [forecastCommodity, setForecastCommodity] = useState("Wheat");

  // Derive unique values from the current prices list
  const states = useMemo(() => getUniqueStates(), []);
  const commodities = useMemo(() => getUniqueCommodities(), []);

  const loadData = async () => {
    setLoading(true);
    try {
      const priceData = await fetchMarketPrices(filters);
      const trendData = await fetchPriceTrends(selectedCommodity);
      setPrices(priceData || []);
      setTrends(trendData || []);
      setLastUpdated(Date.now());
    } catch (err) {
      console.error("Failed to load market data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [filters, selectedCommodity]);

  // Load forecast when forecast tab is active or commodity changes
  const loadForecast = async (commodity) => {
    setForecastLoading(true);
    setForecast(null);
    setForecastError(null);
    try {
      const data = await fetchPriceForecast(commodity, 14);
      if (data) {
        setForecast(data);
      } else {
        setForecastError("Failed to generate forecast. Please try again.");
      }
    } catch (err) {
      console.error("Failed to load forecast:", err);
      setForecastError("An unexpected error occurred while generating the forecast.");
    } finally {
      setForecastLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === "forecast") {
      loadForecast(forecastCommodity);
    }
  }, [activeTab, forecastCommodity]);

  const handleSearchChange = (e) => {
    setFilters(prev => ({ ...prev, search: e.target.value }));
  };

  const handleStateChange = (e) => {
    setFilters(prev => ({ ...prev, state: e.target.value }));
  };

  const handleCommodityChange = (e) => {
    const val = e.target.value;
    setFilters(prev => ({ ...prev, commodity: val }));
    if (val !== "All") setSelectedCommodity(val);
  };

  // Analytics Calculations
  const analytics = useMemo(() => {
    if (!prices || prices.length === 0) {
      return { avg: 0, highest: 0, highestMandi: "N/A", trend: "Stable" };
    }
    
    const validPrices = prices.filter(p => !isNaN(p.modalPrice) && p.modalPrice > 0);
    const avg = validPrices.length > 0 
      ? (validPrices.reduce((acc, curr) => acc + curr.modalPrice, 0) / validPrices.length).toFixed(0)
      : 0;
    
    const maxRecord = prices.reduce((prev, curr) => 
      (Number(prev.maxPrice) || 0) > (Number(curr.maxPrice) || 0) ? prev : curr, 
      prices[0]
    );
    
    return {
      avg,
      highest: maxRecord.maxPrice || 0,
      highestMandi: maxRecord.mandi || "N/A",
      trend: "Stable"
    };
  }, [prices]);

  return (
    <div className="market-prices-container">
       <header className="market-header">
         <div className="header-info">
           <h1><TrendingUp className="header-icon-svg" size={32} /> <span className="notranslate">Market Price Guidance</span></h1>
           <p>Real-time mandi prices and analytics for informed selling</p>
         </div>
        <div className="header-actions">
          <button className="alert-btn">
            <Bell size={18} /> <span className="notranslate">Set Price Alert</span>
          </button>
          <button className="refresh-btn-market" onClick={loadData}>
            <RefreshCw size={18} className={loading ? "spin" : ""} /> <span className="notranslate">Refresh Data</span>
          </button>
        </div>
      </header>
      {lastUpdated && (
        <div style={{ padding: '0 2rem' }}>
          <LastUpdated timestamp={lastUpdated} />
        </div>
      )}

      <section className="market-filters">
        <div className="search-wrapper">
          <Search className="search-icon" size={20} aria-hidden="true" />
          <input 
            type="text" 
            placeholder="Search crop or mandi..." 
            value={filters.search}
            onChange={handleSearchChange}
            aria-label="Search by crop or mandi name"
          />
        </div>
        <div className="filter-group">
          <div className="select-wrapper">
            <Filter size={16} className="select-icon" aria-hidden="true" />
            <select 
              value={filters.state} 
              onChange={handleStateChange}
              aria-label="Filter market by state"
            >
              {states.map(s => <option key={s} value={s}>{s === "All" ? "All States" : s}</option>)}
            </select>
          </div>
          <div className="select-wrapper">
            <BarChart3 size={16} className="select-icon" aria-hidden="true" />
            <select 
              value={filters.commodity} 
              onChange={handleCommodityChange}
              aria-label="Filter market by crop"
            >
              {commodities.map(c => <option key={c} value={c}>{c === "All" ? "All Crops" : c}</option>)}
            </select>
          </div>
        </div>
      </section>

      <div className="market-stats-grid" role="status" aria-live="polite">
        <div className="stat-card price-up">
          <div className="stat-icon-wrap" aria-hidden="true">
            <TrendingUp size={24} />
          </div>
          <div className="stat-content">
            <label>Avg. Market Price</label>
            <h3>₹{loading ? "..." : analytics.avg}</h3>
            <span className="trend-pos">+2.4% from last week</span>
          </div>
        </div>
        <div className="stat-card highest">
          <div className="stat-icon-wrap" aria-hidden="true">
            <Info size={24} />
          </div>
          <div className="stat-content">
            <label>Highest Price</label>
            <h3>₹{loading ? "..." : analytics.highest}</h3>
            <span className="sub-info">{loading ? "Fetching..." : analytics.highestMandi} Mandi</span>
          </div>
        </div>
        <div className="stat-card recommendation">
          <div className="stat-icon-wrap" aria-hidden="true">
            <MapPin size={24} />
          </div>
          <div className="stat-content">
            <label>Suggested Action</label>
            <h3>Hold / Sell</h3>
            <span className="sub-info">Price trend is stable</span>
          </div>
        </div>
      </div>

      <div className="market-main-content">
        <div className="content-tabs" role="tablist">
          <button 
            className={`tab-btn ${activeTab === 'list' ? 'active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'list'}
            onClick={() => setActiveTab('list')}
          >
            Mandi Wise Prices
          </button>
          <button 
            className={`tab-btn ${activeTab === 'trends' ? 'active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'trends'}
            onClick={() => setActiveTab('trends')}
          >
            Price Trends
          </button>
          <button
            className={`tab-btn ${activeTab === 'forecast' ? 'active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'forecast'}
            onClick={() => setActiveTab('forecast')}
          >
            <Sparkles size={15} style={{ marginRight: 4 }} aria-hidden="true" />
            14-Day Forecast
          </button>
        </div>

        {loading ? (
          <Loader message="Fetching latest market data from official sources..." />
        ) : activeTab === 'list' ? (
          <div className="mandi-table-container">
            <table className="mandi-table">
              <thead>
                <tr>
                  <th>Commodity</th>
                  <th>Mandi (District)</th>
                  <th>State</th>
                  <th>Min Price</th>
                  <th>Max Price</th>
                  <th>Modal Price</th>
                  <th>Arrival Date</th>
                </tr>
              </thead>
              <tbody>
                {prices.map((price, idx) => (
                  <tr key={price.id || idx}>
                    <td><span className="commodity-badge">{price.commodity}</span></td>
                    <td>
                      <div className="mandi-name">{price.mandi}</div>
                      <div className="district-name">{price.district}</div>
                    </td>
                    <td>{price.state}</td>
                    <td className="price-val">₹{price.minPrice}</td>
                    <td className="price-val">₹{price.maxPrice}</td>
                    <td className="price-val modal-price">₹{price.modalPrice}</td>
                    <td>{price.arrivalDate}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {prices.length === 0 && (
              <div className="no-data">
                <Search size={48} />
                <p>No mandi data found for selected filters.</p>
              </div>
            )}
          </div>
        ) : activeTab === 'trends' ? (
          <div className="trends-container">
            <div className="trend-header">
              <h3>7-Day Price Trend for {selectedCommodity}</h3>
              <div className="trend-info">
                <span className="price-current">Current: ₹{trends[trends.length-1]?.price}</span>
              </div>
            </div>
            <div 
              className="chart-wrapper"
              role="img"
              aria-label={`Area chart showing 7-day price trend for ${selectedCommodity}.`}
            >
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={trends}>
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#166534" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#166534" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} tickFormatter={(val) => `₹${val}`} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                    formatter={(val) => [`₹${val}`, 'Price']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#166534" 
                    strokeWidth={3}
                    fillOpacity={1} 
                    fill="url(#colorPrice)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          /* ── 14-Day LSTM Forecast Tab ── */
          <div className="forecast-container">
            <div className="forecast-controls">
              <div className="forecast-commodity-selector">
                <label htmlFor="forecast-commodity-select">
                  <Sparkles size={16} aria-hidden="true" /> Select Commodity
                </label>
                <select
                  id="forecast-commodity-select"
                  value={forecastCommodity}
                  onChange={(e) => setForecastCommodity(e.target.value)}
                  aria-label="Select commodity for price forecast"
                >
                  {FORECASTABLE_COMMODITIES.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
              <button
                className="refresh-btn-market"
                onClick={() => loadForecast(forecastCommodity)}
                disabled={forecastLoading}
                aria-label="Refresh price forecast"
              >
                <RefreshCw size={16} className={forecastLoading ? "spin" : ""} />
                Refresh Forecast
              </button>
            </div>

            {forecastLoading ? (
              <Loader message="Running LSTM price forecast model..." />
            ) : forecast ? (
              <>
                {/* Best Time to Sell recommendation card */}
                <div className={`sell-recommendation-card ${
                  forecast.recommendation.includes('rise') ? 'rec-hold' :
                  forecast.recommendation.includes('fall') ? 'rec-sell' : 'rec-stable'
                }`}>
                  <div className="rec-icon-wrap" aria-hidden="true">
                    {forecast.recommendation.includes('rise') ? <TrendingUp size={28} /> :
                     forecast.recommendation.includes('fall') ? <TrendingDown size={28} /> :
                     <Info size={28} />}
                  </div>
                  <div className="rec-content">
                    <h4>
                      Best Time to Sell: <strong>{new Date(forecast.best_sell_date).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })}</strong>
                      {' '}at <strong>₹{forecast.best_sell_price.toLocaleString('en-IN')}/quintal</strong>
                    </h4>
                    <p>{forecast.recommendation}</p>
                    <span className="model-badge">
                      Powered by {forecast.model_type} · Generated {new Date(forecast.generated_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </div>

                {/* 14-day forecast chart with confidence bands */}
                <div className="forecast-chart-header">
                  <h3>14-Day Price Forecast — {forecastCommodity}</h3>
                  <span className="confidence-legend">
                    <span className="ci-swatch" aria-hidden="true" /> Confidence interval (±1σ)
                  </span>
                </div>
                <div
                  className="chart-wrapper"
                  role="img"
                  aria-label={`Area chart showing 14-day LSTM price forecast for ${forecastCommodity} with confidence bands.`}
                >
                  <ResponsiveContainer width="100%" height={380}>
                    <AreaChart
                      data={forecast.forecast.map(d => ({
                        date: new Date(d.date).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }),
                        price: d.price,
                        lower: d.lower_bound,
                        upper: d.upper_bound,
                        // Recharts area needs [lower, upper] as a range
                        band: [d.lower_bound, d.upper_bound],
                      }))}
                      margin={{ top: 10, right: 20, left: 10, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor="#166534" stopOpacity={0.15} />
                          <stop offset="95%" stopColor="#166534" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.12} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.04} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                      <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                      <YAxis axisLine={false} tickLine={false} tickFormatter={v => `₹${v}`} tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                        formatter={(val, name) => {
                          if (name === 'upper') return [`₹${val}`, 'Upper bound'];
                          if (name === 'lower') return [`₹${val}`, 'Lower bound'];
                          return [`₹${val}`, 'Forecast price'];
                        }}
                      />
                      {/* Confidence band — upper */}
                      <Area
                        type="monotone"
                        dataKey="upper"
                        stroke="none"
                        fill="url(#ciGrad)"
                        fillOpacity={1}
                        legendType="none"
                      />
                      {/* Confidence band — lower (fills back to baseline) */}
                      <Area
                        type="monotone"
                        dataKey="lower"
                        stroke="none"
                        fill="#ffffff"
                        fillOpacity={1}
                        legendType="none"
                      />
                      {/* Forecast price line */}
                      <Area
                        type="monotone"
                        dataKey="price"
                        stroke="#166534"
                        strokeWidth={2.5}
                        fill="url(#forecastGrad)"
                        fillOpacity={1}
                        dot={false}
                        activeDot={{ r: 5, fill: '#166534' }}
                      />
                      {/* Mark the best sell date */}
                      <ReferenceLine
                        x={new Date(forecast.best_sell_date).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })}
                        stroke="#f59e0b"
                        strokeDasharray="4 3"
                        strokeWidth={2}
                        label={{ value: 'Best Sell', position: 'top', fill: '#f59e0b', fontSize: 11 }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* Forecast data table */}
                <details className="forecast-table-details">
                  <summary>View full forecast data table</summary>
                  <div className="forecast-table-wrap">
                    <table className="mandi-table forecast-table">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Forecast Price (₹/qtl)</th>
                          <th>Lower Bound</th>
                          <th>Upper Bound</th>
                        </tr>
                      </thead>
                      <tbody>
                        {forecast.forecast.map((d, i) => (
                          <tr key={i} className={d.date === forecast.best_sell_date ? 'best-sell-row' : ''}>
                            <td>{new Date(d.date).toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short' })}</td>
                            <td className="price-val modal-price">₹{d.price.toLocaleString('en-IN')}</td>
                            <td className="price-val">₹{d.lower_bound.toLocaleString('en-IN')}</td>
                            <td className="price-val">₹{d.upper_bound.toLocaleString('en-IN')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              </>
            ) : forecastError ? (
              <div className="no-data error-state">
                <Info size={48} className="text-red-500" />
                <p className="error-msg">{forecastError}</p>
                <button 
                  className="refresh-btn-market mt-4" 
                  onClick={() => loadForecast(forecastCommodity)}
                >
                  <RefreshCw size={16} style={{ marginRight: 8 }} /> Retry Forecast
                </button>
              </div>
            ) : (
              <div className="no-data">
                <Sparkles size={48} />
                <p>Select a commodity and click Refresh Forecast to generate predictions.</p>
              </div>
            )}
          </div>
        )}
      </div>

      <footer className="market-footer-info">
        <div className="info-item">
          <Info size={16} />
          <span>Prices are per quintal (100kg). Data sourced from Agmarknet API.</span>
        </div>
        <div className="info-links">
          <a href="https://enam.gov.in" target="_blank" rel="noreferrer">Official eNAM Portal</a>
          <a href="https://agmarknet.gov.in" target="_blank" rel="noreferrer">Agmarknet</a>
        </div>
      </footer >
    </div>
  );
};

export default MarketPrices;
