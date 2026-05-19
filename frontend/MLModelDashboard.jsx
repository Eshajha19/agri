import React, { useState, useEffect } from "react";
import { TrendingUp, TrendingDown, Activity, Zap, AlertTriangle } from "lucide-react";
import "./MLModelDashboard.css";

const MLModelDashboard = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [abTests, setAbTests] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [deploymentHistory, setDeploymentHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels();
    fetchABTests();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/ml/models");
      const data = await response.json();
      setModels(data.models || []);
      if (data.models && data.models.length > 0) {
        setSelectedModel(data.models[0]);
        fetchDeploymentHistory(data.models[0].model_name);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch("/api/ml/metrics");
      const data = await response.json();
      setMetrics(data.models || {});
    } catch (err) {
      console.error("Failed to fetch metrics:", err);
    }
  };

  const fetchABTests = async () => {
    try {
      const response = await fetch("/api/ml/ab-tests");
      const data = await response.json();
      setAbTests(data.tests || []);
    } catch (err) {
      console.error("Failed to fetch A/B tests:", err);
    }
  };

  const fetchDeploymentHistory = async (modelName) => {
    try {
      const response = await fetch(`/api/ml/deployment-history/${modelName}`);
      const data = await response.json();
      setDeploymentHistory(data.history || []);
    } catch (err) {
      console.error("Failed to fetch deployment history:", err);
    }
  };

  const promoteToCanary = async (modelId, modelName) => {
    try {
      const response = await fetch("/api/ml/promote-canary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, model_name: modelName }),
      });
      const data = await response.json();
      if (data.success) {
        fetchModels();
        alert("Model promoted to canary");
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const promoteToProduction = async (modelId, modelName) => {
    try {
      const response = await fetch("/api/ml/promote-production", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, model_name: modelName }),
      });
      const data = await response.json();
      if (data.success) {
        fetchModels();
        alert("Model promoted to production");
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const rollback = async (modelName) => {
    try {
      const response = await fetch("/api/ml/rollback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName }),
      });
      const data = await response.json();
      if (data.success) {
        fetchModels();
        alert("Model rolled back");
      }
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="ml-model-dashboard">
      <div className="dashboard-header">
        <h1>ML Model Management</h1>
        <p>Manage model versions, A/B tests, and deployments</p>
      </div>

      {error && (
        <div className="error-banner">
          <AlertTriangle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div className="dashboard-grid">
        {/* Models List */}
        <div className="models-section">
          <h2>Model Versions</h2>
          <div className="models-list">
            {models.map((model) => (
              <div
                key={model.model_id}
                className={`model-card ${selectedModel?.model_id === model.model_id ? "active" : ""}`}
                onClick={() => {
                  setSelectedModel(model);
                  fetchDeploymentHistory(model.model_name);
                }}
              >
                <div className="model-header">
                  <h3>{model.model_name}</h3>
                  <span className={`status-badge ${model.status}`}>{model.status}</span>
                </div>
                <p className="model-version">v{model.version}</p>
                <div className="model-traffic">
                  {model.canary_traffic_percentage > 0 && (
                    <span>{model.canary_traffic_percentage}% traffic</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Model Details */}
        {selectedModel && (
          <div className="model-details">
            <h2>{selectedModel.model_name} - v{selectedModel.version}</h2>

            {/* Metrics */}
            <div className="metrics-grid">
              {metrics[selectedModel.model_id] && (
                <>
                  <div className="metric-card">
                    <div className="metric-label">MAE</div>
                    <div className="metric-value">
                      {metrics[selectedModel.model_id].mae?.toFixed(4) || "N/A"}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">RMSE</div>
                    <div className="metric-value">
                      {metrics[selectedModel.model_id].rmse?.toFixed(4) || "N/A"}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Latency (ms)</div>
                    <div className="metric-value">
                      {metrics[selectedModel.model_id].latency?.toFixed(2) || "N/A"}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Predictions</div>
                    <div className="metric-value">
                      {metrics[selectedModel.model_id].predictions || 0}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Actions */}
            <div className="actions-section">
              <h3>Actions</h3>
              {selectedModel.status === "draft" && (
                <button
                  className="btn btn-primary"
                  onClick={() => promoteToCanary(selectedModel.model_id, selectedModel.model_name)}
                >
                  <Zap size={16} /> Promote to Canary
                </button>
              )}
              {selectedModel.status === "canary" && (
                <>
                  <button
                    className="btn btn-primary"
                    onClick={() => promoteToCanary(selectedModel.model_id, selectedModel.model_name)}
                  >
                    <TrendingUp size={16} /> Increase Traffic
                  </button>
                  <button
                    className="btn btn-success"
                    onClick={() => promoteToProduction(selectedModel.model_id, selectedModel.model_name)}
                  >
                    <TrendingUp size={16} /> Promote to Production
                  </button>
                </>
              )}
              {selectedModel.status === "staging" && (
                <button
                  className="btn btn-success"
                  onClick={() => promoteToProduction(selectedModel.model_id, selectedModel.model_name)}
                >
                  <TrendingUp size={16} /> Promote to Production
                </button>
              )}
              {selectedModel.status === "production" && (
                <button className="btn btn-danger" onClick={() => rollback(selectedModel.model_name)}>
                  <TrendingDown size={16} /> Rollback
                </button>
              )}
            </div>

            {/* Deployment History */}
            <div className="deployment-history">
              <h3>Deployment History</h3>
              <div className="history-list">
                {deploymentHistory.map((entry, index) => (
                  <div key={index} className="history-entry">
                    <div className="history-action">{entry.action}</div>
                    <div className="history-traffic">{entry.traffic_percentage}% traffic</div>
                    <div className="history-time">
                      {new Date(entry.timestamp).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* A/B Tests */}
        <div className="ab-tests-section">
          <h2>Active A/B Tests</h2>
          <div className="tests-list">
            {abTests.length === 0 ? (
              <p>No active A/B tests</p>
            ) : (
              abTests.map((test) => (
                <div key={test.test_id} className="test-card">
                  <h3>{test.test_name}</h3>
                  <div className="test-arms">
                    <div className="arm">
                      <div className="arm-name">{test.control_arm.name}</div>
                      <div className="arm-allocation">
                        {(test.current_allocation[test.control_arm.model_id] * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="arm vs">vs</div>
                    <div className="arm">
                      <div className="arm-name">{test.variant_arm.name}</div>
                      <div className="arm-allocation">
                        {(test.current_allocation[test.variant_arm.model_id] * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div className="test-metrics">
                    <small>
                      Control: {test.control_arm.predictions} | Variant:{" "}
                      {test.variant_arm.predictions}
                    </small>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLModelDashboard;
