import React, { useState, useEffect, useMemo } from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area 
} from "recharts";
import { 
  TrendingUp, TrendingDown, Search, Filter, MapPin, Calendar, Info, Bell, ArrowRight, ArrowLeft, RefreshCw, BarChart3
} from "lucide-react";
import { fetchMarketPrices, fetchPriceTrends, getUniqueStates, getUniqueCommodities } from "./lib/marketApi";
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
        ) : (
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
