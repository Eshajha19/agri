// frontend/components/FeatureFlagDashboard/FeatureFlagDashboard.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './FeatureFlagDashboard.css';

const FeatureFlagDashboard = () => {
  const [flags, setFlags] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [flagsRes, expsRes] = await Promise.all([
        axios.get('/api/flags'),
        axios.get('/api/experiments')
      ]);
      setFlags(flagsRes.data.flags || []);
      setExperiments(expsRes.data.experiments || []);
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
      setError('Failed to load dashboard. Make sure the backend is running.');
      setLoading(false);
    }
  };

  const handleToggleFlag = async (flagId, currentEnabled) => {
    try {
      const flag = flags.find(f => f.id === flagId);
      await axios.post(`/api/flags/${flagId}`, {
        ...flag,
        enabled: !currentEnabled
      });
      setFlags(prev => prev.map(f => 
        f.id === flagId ? { ...f, enabled: !currentEnabled } : f
      ));
    } catch (err) {
      alert('Failed to update flag');
    }
  };

  const handleRolloutChange = async (flagId, newPct) => {
    try {
      const flag = flags.find(f => f.id === flagId);
      await axios.post(`/api/flags/${flagId}`, {
        ...flag,
        rollout_pct: parseInt(newPct)
      });
      setFlags(prev => prev.map(f => 
        f.id === flagId ? { ...f, rollout_pct: parseInt(newPct) } : f
      ));
    } catch (err) {
      console.error('Failed to update rollout:', err);
    }
  };

  const handleRollback = async (flagId) => {
    if (!window.confirm(`Are you sure you want to ROLLBACK ${flagId}? This will disable it for everyone.`)) return;
    
    try {
      const response = await axios.post(`/api/flags/${flagId}/rollback`);
      const updatedFlag = response.data.flag;
      setFlags(prev => prev.map(f => f.id === flagId ? updatedFlag : f));
    } catch (err) {
      alert('Rollback failed');
    }
  };

  if (loading) return <div className="ff-dashboard">Loading framework...</div>;
  if (error) return <div className="ff-dashboard"><div className="error-msg">{error}</div></div>;

  return (
    <div className="ff-dashboard">
      <header className="ff-header">
        <h1>Feature Flag Framework</h1>
        <button className="btn-refresh" onClick={fetchData}>Refresh Data</button>
      </header>

      <section className="ff-section">
        <h2>🚩 Feature Flags</h2>
        <div className="ff-table-container">
          <table className="ff-table">
            <thead>
              <tr>
                <th>Flag ID</th>
                <th>Enabled</th>
                <th>Rollout %</th>
                <th>Description</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {flags.map(flag => (
                <tr key={flag.id}>
                  <td><span className="ff-flag-id">{flag.id}</span></td>
                  <td>
                    <label className="switch">
                      <input 
                        type="checkbox" 
                        checked={flag.enabled} 
                        onChange={() => handleToggleFlag(flag.id, flag.enabled)}
                      />
                      <span className="slider"></span>
                    </label>
                  </td>
                  <td>
                    <div className="rollout-cell">
                      <input 
                        type="range" 
                        min="0" 
                        max="100" 
                        value={flag.rollout_pct} 
                        className="rollout-input"
                        onChange={(e) => handleRolloutChange(flag.id, e.target.value)}
                      />
                      <span className="rollout-val">{flag.rollout_pct}%</span>
                    </div>
                  </td>
                  <td><span className="ff-description">{flag.description}</span></td>
                  <td>
                    <button className="btn-rollback" onClick={() => handleRollback(flag.id)}>Rollback</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="ff-section">
        <h2>🧪 A/B Experiments</h2>
        <div className="exp-grid">
          {experiments.map(exp => (
            <div className="exp-card" key={exp.id}>
              <div className={`exp-status status-${exp.status}`}>{exp.status}</div>
              <h3>{exp.name}</h3>
              <p className="ff-description">{exp.description}</p>
              
              <div className="exp-variants">
                {exp.variants.map(variant => (
                  <div key={variant.id}>
                    <div className="variant-row">
                      <span>{variant.name}</span>
                      <span>{variant.weight}%</span>
                    </div>
                    <div className="variant-bar-bg">
                      <div className="variant-bar-fill" style={{ width: `${variant.weight}%` }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default FeatureFlagDashboard;
