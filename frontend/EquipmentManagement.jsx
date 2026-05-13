import { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from "react-i18next";
import {
  getEquipmentInfo,
  evaluateEquipmentHealth,
} from "./utils/equipmentDatabase";

import EquipmentService from "./services/equipmentApi";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

export default function EquipmentManagement({ onClose }) {
  const { t } = useTranslation();

  const equipmentService = useRef(EquipmentService);

  // =========================
  // SAFE DEFAULT STATES
  // =========================

  const [equipment, setEquipment] = useState([]);
  const [selectedEquipment, setSelectedEquipment] = useState(null);

  const [sensorData, setSensorData] = useState({});

  const [healthData, setHealthData] = useState({
    overall: 0,
    status: "unknown",
    recommendations: [],
  });

  const [maintenanceHistory, setMaintenanceHistory] = useState([]);

  const [analytics, setAnalytics] = useState({
    historicalData: [],
    maintenanceCost: 0,
    operatingHours: 0,
  });

  const [alerts, setAlerts] = useState([]);

  const [showAddEquipment, setShowAddEquipment] = useState(false);
  const [showMaintenance, setShowMaintenance] = useState(false);

  const [loading, setLoading] = useState(false);

  const [realTimeMode, setRealTimeMode] = useState(true);

  const [timeRange, setTimeRange] = useState("7d");

  // =========================
  // INITIAL LOAD
  // =========================

  useEffect(() => {
    loadEquipmentData();
  }, []);

  // =========================
  // REALTIME POLLING
  // =========================

  useEffect(() => {
    const interval = setInterval(() => {
      if (realTimeMode && selectedEquipment?.id) {
        updateSensorData(selectedEquipment.id);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [realTimeMode, selectedEquipment]);

  // =========================
  // HEALTH UPDATE
  // =========================

  useEffect(() => {
    if (
      selectedEquipment &&
      sensorData &&
      Object.keys(sensorData).length > 0
    ) {
      try {
        const health = evaluateEquipmentHealth(
          selectedEquipment.type,
          sensorData
        );

        setHealthData({
          overall: health?.overall || 0,
          status: health?.status || "unknown",
          recommendations: health?.recommendations || [],
        });
      } catch (err) {
        console.error("Health evaluation failed:", err);
      }
    }
  }, [sensorData, selectedEquipment]);

  // =========================
  // LOAD EQUIPMENT
  // =========================

  const loadEquipmentData = async () => {
    setLoading(true);

    try {
      const equipmentList =
        (await equipmentService.current?.getEquipmentList?.()) || [];

      setEquipment(Array.isArray(equipmentList) ? equipmentList : []);

      // auto select first equipment
      if (equipmentList?.length > 0) {
        await selectEquipment(equipmentList[0]);
      }
    } catch (error) {
      console.error("Failed to load equipment data:", error);

      setEquipment([]);
    } finally {
      setLoading(false);
    }
  };

  // =========================
  // SELECT EQUIPMENT
  // =========================

  const selectEquipment = async (eq) => {
    if (!eq?.id) return;

    setSelectedEquipment(eq);

    await Promise.all([
      updateSensorData(eq.id),
      loadMaintenanceHistory(eq.id),
      loadAnalytics(eq.id, timeRange),
      loadAlerts(eq.id),
    ]);
  };

  // =========================
  // SENSOR DATA
  // =========================

  const updateSensorData = async (equipmentId) => {
    if (!equipmentId) return;

    try {
      const data =
        (await equipmentService.current?.getSensorData?.(equipmentId)) || {};

      setSensorData(data || {});
    } catch (error) {
      console.error("Failed to update sensor data:", error);

      setSensorData({});
    }
  };

  // =========================
  // MAINTENANCE
  // =========================

  const loadMaintenanceHistory = async (equipmentId) => {
    try {
      const history =
        (await equipmentService.current?.getMaintenanceHistory?.(
          equipmentId
        )) || [];

      setMaintenanceHistory(Array.isArray(history) ? history : []);
    } catch (error) {
      console.error("Failed to load maintenance history:", error);

      setMaintenanceHistory([]);
    }
  };

  // =========================
  // ANALYTICS
  // =========================

  const loadAnalytics = async (equipmentId, range = timeRange) => {
    try {
      const data =
        (await equipmentService.current?.getEquipmentAnalytics?.(
          equipmentId,
          range
        )) || {};

      setAnalytics({
        historicalData: Array.isArray(data?.historicalData)
          ? data.historicalData
          : [],
        maintenanceCost: Number(data?.maintenanceCost || 0),
        operatingHours: Number(data?.operatingHours || 0),
      });
    } catch (error) {
      console.error("Failed to load analytics:", error);

      setAnalytics({
        historicalData: [],
        maintenanceCost: 0,
        operatingHours: 0,
      });
    }
  };

  // =========================
  // ALERTS
  // =========================

  const loadAlerts = async (equipmentId) => {
    try {
      const alertData =
        (await equipmentService.current?.getPredictiveAlerts?.(
          equipmentId
        )) || [];

      setAlerts(Array.isArray(alertData) ? alertData : []);
    } catch (error) {
      console.error("Failed to load alerts:", error);

      setAlerts([]);
    }
  };

  // =========================
  // HELPERS
  // =========================

  const getHealthColor = (score = 0) => {
    if (score >= 90) return "#16a34a";
    if (score >= 75) return "#84cc16";
    if (score >= 60) return "#f59e0b";
    if (score >= 40) return "#ef4444";
    return "#dc2626";
  };

  const getAlertColor = (type) => {
    switch (type) {
      case "critical":
        return "#dc2626";

      case "warning":
        return "#f59e0b";

      case "info":
        return "#3b82f6";

      default:
        return "#6b7280";
    }
  };

  const formatCurrency = (amount = 0) => {
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
    }).format(amount);
  };

  // =========================
  // RENDER
  // =========================

  return (
    <div
      style={{
        maxWidth: "1400px",
        margin: "40px auto",
        padding: "24px",
        background: "#fff",
        borderRadius: "16px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
      }}
    >
      {/* HEADER */}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "24px",
          alignItems: "center",
        }}
      >
        <h2 style={{ margin: 0, color: "#16a34a" }}>
          🚜 Smart Equipment Management
        </h2>

        <div style={{ display: "flex", gap: "12px" }}>
          <button
            onClick={() => setRealTimeMode(!realTimeMode)}
            style={{
              padding: "8px 16px",
              border: "none",
              borderRadius: "8px",
              background: realTimeMode ? "#16a34a" : "#6b7280",
              color: "#fff",
              cursor: "pointer",
            }}
          >
            {realTimeMode ? "📡 Real-time" : "⏸ Offline"}
          </button>

          <button
            onClick={onClose}
            style={{
              padding: "8px 16px",
              border: "none",
              borderRadius: "8px",
              background: "#6b7280",
              color: "#fff",
              cursor: "pointer",
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* LOADING */}

      {loading && (
        <div
          style={{
            padding: "20px",
            textAlign: "center",
            color: "#6b7280",
          }}
        >
          Loading equipment...
        </div>
      )}

      {/* EQUIPMENT LIST */}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "16px",
          marginBottom: "24px",
        }}
      >
        {equipment.map((eq) => (
          <div
            key={eq.id}
            onClick={() => selectEquipment(eq)}
            style={{
              padding: "16px",
              border:
                selectedEquipment?.id === eq.id
                  ? "2px solid #16a34a"
                  : "1px solid #e5e7eb",
              borderRadius: "12px",
              cursor: "pointer",
              background:
                selectedEquipment?.id === eq.id ? "#f0fdf4" : "#fff",
            }}
          >
            <div
              style={{
                display: "flex",
                gap: "12px",
                alignItems: "center",
              }}
            >
              <span style={{ fontSize: "28px" }}>
                {getEquipmentInfo(eq?.type)?.icon || "🚜"}
              </span>

              <div>
                <h4 style={{ margin: 0 }}>{eq?.name || "Unnamed"}</h4>

                <div
                  style={{
                    fontSize: "12px",
                    color: "#6b7280",
                  }}
                >
                  {eq?.status || "unknown"}
                </div>
              </div>
            </div>

            <div
              style={{
                marginTop: "12px",
                fontSize: "12px",
                color: "#6b7280",
              }}
            >
              Engine Hours: {eq?.engine_hours || 0}h
            </div>

            {eq?.location?.lat != null &&
              eq?.location?.lng != null && (
                <div
                  style={{
                    fontSize: "12px",
                    color: "#6b7280",
                  }}
                >
                  📍 {eq.location.lat.toFixed(4)},{" "}
                  {eq.location.lng.toFixed(4)}
                </div>
              )}
          </div>
        ))}
      </div>

      {/* SELECTED EQUIPMENT */}

      {selectedEquipment && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "24px",
          }}
        >
          {/* SENSOR DATA */}

          <div
            style={{
              padding: "20px",
              background: "#f8fafc",
              borderRadius: "12px",
            }}
          >
            <h3>📊 Real-time Monitoring</h3>

            <div
              style={{
                marginBottom: "20px",
                fontSize: "40px",
                fontWeight: "bold",
                color: getHealthColor(healthData?.overall),
              }}
            >
              {healthData?.overall || 0}%
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(160px, 1fr))",
                gap: "12px",
              }}
            >
              {Object.entries(sensorData || {}).map(([key, value]) => (
                <div
                  key={key}
                  style={{
                    padding: "12px",
                    background: "#fff",
                    borderRadius: "8px",
                  }}
                >
                  <div
                    style={{
                      fontSize: "11px",
                      color: "#6b7280",
                    }}
                  >
                    {key.replace(/_/g, " ").toUpperCase()}
                  </div>

                  <div
                    style={{
                      fontWeight: "bold",
                      marginTop: "6px",
                    }}
                  >
                    {typeof value === "number"
                      ? value.toFixed(1)
                      : String(value)}
                  </div>
                </div>
              ))}
            </div>

            {/* ALERTS */}

            {alerts.length > 0 && (
              <div style={{ marginTop: "24px" }}>
                <h4>Alerts</h4>

                {alerts.map((alert) => (
                  <div
                    key={alert.id}
                    style={{
                      padding: "12px",
                      marginBottom: "10px",
                      background: "#fff",
                      borderLeft: `4px solid ${getAlertColor(
                        alert?.type
                      )}`,
                      borderRadius: "6px",
                    }}
                  >
                    <div>{alert?.message}</div>

                    <div
                      style={{
                        fontSize: "12px",
                        color: "#6b7280",
                        marginTop: "4px",
                      }}
                    >
                      {alert?.recommendation}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ANALYTICS */}

          <div
            style={{
              padding: "20px",
              background: "#f8fafc",
              borderRadius: "12px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "16px",
              }}
            >
              <h3>📈 Analytics Dashboard</h3>

              <select
                value={timeRange}
                onChange={(e) => {
                  const range = e.target.value;

                  setTimeRange(range);

                  if (selectedEquipment?.id) {
                    loadAnalytics(selectedEquipment.id, range);
                  }
                }}
              >
                <option value="1d">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="90d">Last 90 Days</option>
              </select>
            </div>

            {/* SAFE CHART RENDERING */}

            <div
              style={{
                background: "#fff",
                padding: "16px",
                borderRadius: "8px",
                marginBottom: "20px",
              }}
            >
              <h4>Equipment Utilization</h4>

              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={analytics?.historicalData || []}>
                  <CartesianGrid strokeDasharray="3 3" />

                  <XAxis dataKey="date" />

                  <YAxis />

                  <Tooltip />

                  <Line
                    type="monotone"
                    dataKey="utilization"
                    stroke="#16a34a"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div
              style={{
                background: "#fff",
                padding: "16px",
                borderRadius: "8px",
              }}
            >
              <h4>Fuel Consumption</h4>

              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={analytics?.historicalData || []}>
                  <CartesianGrid strokeDasharray="3 3" />

                  <XAxis dataKey="date" />

                  <YAxis />

                  <Tooltip />

                  <Bar
                    dataKey="fuelConsumption"
                    fill="#f59e0b"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* STATS */}

            <div
              style={{
                display: "flex",
                justifyContent: "space-around",
                marginTop: "24px",
              }}
            >
              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontSize: "24px",
                    fontWeight: "bold",
                  }}
                >
                  {formatCurrency(analytics?.maintenanceCost || 0)}
                </div>

                <div
                  style={{
                    fontSize: "12px",
                    color: "#6b7280",
                  }}
                >
                  Maintenance Cost
                </div>
              </div>

              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontSize: "24px",
                    fontWeight: "bold",
                  }}
                >
                  {analytics?.operatingHours || 0}
                </div>

                <div
                  style={{
                    fontSize: "12px",
                    color: "#6b7280",
                  }}
                >
                  Operating Hours
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}