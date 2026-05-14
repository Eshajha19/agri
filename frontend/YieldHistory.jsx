import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  FaHistory, FaChartLine, FaExclamationTriangle, FaCheckCircle, 
  FaCalendarAlt, FaLeaf, FaSave, FaTachometerAlt 
} from 'react-icons/fa';
import { fetchYieldAnalytics, recordActualYield } from './services/yieldApi';
import { toast } from 'react-hot-toast';
import './YieldHistory.css';

const YieldHistory = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actualYieldInputs, setActualYieldInputs] = useState({});

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      const result = await fetchYieldAnalytics();
      setData(result);
    } catch (err) {
      setError('Failed to load yield history. Please try again later.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleRecordActual = async (recordId) => {
    const actual = parseFloat(actualYieldInputs[recordId]);
    if (isNaN(actual) || actual < 0) {
      toast.error('Please enter a valid actual yield.');
      return;
    }

    try {
      await recordActualYield({ record_id: recordId, actual_yield: actual });
      toast.success('Actual yield recorded successfully!');
      loadAnalytics(); // Refresh
      setActualYieldInputs(prev => {
        const next = { ...prev };
        delete next[recordId];
        return next;
      });
    } catch (err) {
      toast.error('Failed to save actual yield.');
      console.error(err);
    }
  };

  if (loading) return <div className="yield-history-loading">Loading analytics...</div>;
  if (error) return <div className="yield-history-error">{error}</div>;

  const chartData = [...(data?.history || [])]
    .reverse()
    .filter(item => item.actual_yield !== null)
    .map(item => ({
      name: `${item.crop} (${item.season})`,
      predicted: item.predicted_yield,
      actual: item.actual_yield,
      accuracy: item.accuracy
    }));

  return (
    <div className="yield-history-container">
      <header className="yh-header">
        <h1><FaHistory /> Yield History & ML Analytics</h1>
        <p>Track your harvest performance and monitor model accuracy.</p>
      </header>

      {data?.drift_alert && (
        <div className="yh-drift-alert">
          <FaExclamationTriangle />
          <div className="alert-content">
            <strong>Model Drift Detected!</strong>
            <p>{data.drift_alert}</p>
          </div>
        </div>
      )}

      <div className="yh-stats-grid">
        <div className="yh-stat-card">
          <div className="yh-stat-icon"><FaChartLine /></div>
          <div className="yh-stat-info">
            <label>Avg. Accuracy</label>
            <span>{data?.metrics?.avg_accuracy ? `${data.metrics.avg_accuracy}%` : 'N/A'}</span>
          </div>
        </div>
        <div className="yh-stat-card">
          <div className="yh-stat-icon"><FaTachometerAlt /></div>
          <div className="yh-stat-info">
            <label>Drift Status</label>
            <span className={data?.metrics?.drift_detected ? 'drift-bad' : 'drift-good'}>
              {data?.metrics?.drift_detected ? 'Unstable' : 'Healthy'}
            </span>
          </div>
        </div>
        <div className="yh-stat-card">
          <div className="yh-stat-icon"><FaCheckCircle /></div>
          <div className="yh-stat-info">
            <label>Completed Seasons</label>
            <span>{data?.metrics?.completed_records || 0}</span>
          </div>
        </div>
      </div>

      <div className="yh-charts-section">
        <div className="yh-chart-container">
          <h3>Predicted vs Actual Yield (kg/acre)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="predicted" fill="#4caf50" name="Predicted" />
              <Bar dataKey="actual" fill="#ff9800" name="Actual" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="yh-chart-container">
          <h3>Accuracy Trend (%)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="accuracy" stroke="#2196f3" strokeWidth={3} name="Accuracy %" dot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="yh-list-section">
        <h3>Seasonal Prediction Logs</h3>
        <div className="yh-table-wrapper">
          <table className="yh-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Crop & Season</th>
                <th>Area</th>
                <th>Prediction</th>
                <th>Actual Yield</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {data?.history?.map(item => (
                <tr key={item.id}>
                  <td>{new Date(item.timestamp).toLocaleDateString()}</td>
                  <td>
                    <div className="yh-crop-cell">
                      <FaLeaf /> <span>{item.crop}</span>
                      <small><FaCalendarAlt /> {item.season}</small>
                    </div>
                  </td>
                  <td>{item.area} acres</td>
                  <td className="pred-val">{item.predicted_yield} kg</td>
                  <td>
                    {item.actual_yield !== null ? (
                      <span className="actual-val">{item.actual_yield} kg</span>
                    ) : (
                      <div className="yh-record-form">
                        <input 
                          type="number" 
                          placeholder="Enter yield"
                          value={actualYieldInputs[item.id] || ''}
                          onChange={(e) => setActualYieldInputs(prev => ({ ...prev, [item.id]: e.target.value }))}
                        />
                        <button onClick={() => handleRecordActual(item.id)} title="Save harvest data">
                          <FaSave />
                        </button>
                      </div>
                    )}
                  </td>
                  <td>
                    {item.accuracy !== null ? (
                      <span className={`yh-acc-badge ${item.accuracy > 85 ? 'high' : item.accuracy > 70 ? 'mid' : 'low'}`}>
                        {item.accuracy}%
                      </span>
                    ) : (
                      <span className="yh-pending">Pending Harvest</span>
                    )}
                  </td>
                </tr>
              ))}
              {(!data?.history || data.history.length === 0) && (
                <tr>
                  <td colSpan="6" className="yh-empty">No predictions recorded yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default YieldHistory;
