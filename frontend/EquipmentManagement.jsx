import { useState, useEffect, useRef, useCallback } from "react";
import { getEquipmentInfo, evaluateEquipmentHealth } from './utils/equipmentDatabase';
import EquipmentService from './services/equipmentApi';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

export default function EquipmentManagement({ onClose }) {
  const [equipment, setEquipment] = useState([]);
  const [selectedEquipment, setSelectedEquipment] = useState(null);
  const [sensorData, setSensorData] = useState({});
  const [healthData, setHealthData] = useState({});
  const [maintenanceHistory, setMaintenanceHistory] = useState([]);
  const [analytics, setAnalytics] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [showAddEquipment, setShowAddEquipment] = useState(false);
  const [showMaintenance, setShowMaintenance] = useState(false);
  const [loading, setLoading] = useState(false);
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');

  const equipmentService = useRef(new EquipmentService());

  // Synchronization refs
  const selectedIdRef = useRef(null);
  const mountedRef = useRef(true);

  // Request tracking to avoid stale updates
  const requestTrackerRef = useRef({
    equipment: 0,
    sensors: 0,
    health: 0,
    maintenance: 0,
    analytics: 0,
    alerts: 0,
  });

  const abortControllerRef = useRef(null);

async function selectEquipment(eq) {
  if (!eq) return;

  // cancel any previous requests
  if (abortControllerRef.current) {
    abortControllerRef.current.abort();
  }
  const controller = new AbortController();
  abortControllerRef.current = controller;

  setSelectedEquipment(eq);
  selectedIdRef.current = eq.id;

  try {
    await Promise.all([
      updateSensorData(eq.id, { signal: controller.signal }),
      updateHealthData(eq.id, { signal: controller.signal }),
      loadMaintenanceHistory(eq.id, { signal: controller.signal }),
      loadAnalytics(eq.id, { signal: controller.signal }),
      loadAlerts(eq.id, { signal: controller.signal }),
    ]);
  } catch (err) {
    if (err.name === "AbortError") {
      console.log("Request cancelled due to equipment switch");
    } else {
      console.error("Failed to load equipment data", err);
    }
  }
}


  // Lifecycle + realtime polling
  useEffect(() => {
    mountedRef.current = true;

    loadEquipmentData();

    const interval = setInterval(() => {
      if (realTimeMode && selectedIdRef.current) {
        updateSensorData();
      }
    }, 4000);

    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, [realTimeMode]);

  // Keep selected equipment ref synchronized
  useEffect(() => {
    selectedIdRef.current = selectedEquipment?.id ?? null;
  }, [selectedEquipment]);

  const loadEquipmentData = async () => {
    const requestId = ++requestTrackerRef.current.equipment;

    setLoading(true);

    try {
      const equipmentList =
        await equipmentService.current.getEquipmentList();

      // Prevent stale updates
      if (
        !mountedRef.current ||
        requestTrackerRef.current.equipment !== requestId
      ) {
        return;
      }

      setEquipment(equipmentList);

      if (
        equipmentList.length > 0 &&
        !selectedIdRef.current
      ) {
        await selectEquipment(equipmentList[0]);
      }
    } catch (error) {
      console.error(
        'Failed to load equipment data:',
        error
      );
    } finally {
      if (
        mountedRef.current &&
        requestTrackerRef.current.equipment === requestId
      ) {
        setLoading(false);
      }
    }
  };



  const updateSensorData = useCallback(
    async (equipmentId) => {
      const id = equipmentId || selectedIdRef.current;

      if (!id) return;

      const requestId =
        ++requestTrackerRef.current.sensors;

      try {
        const data =
          await equipmentService.current.getSensorData(id);

        if (
          !mountedRef.current ||
          requestTrackerRef.current.sensors !== requestId ||
          selectedIdRef.current !== id
        ) {
          return;
        }

        setSensorData(data);
      } catch (error) {
        console.error(
          'Failed to update sensor data:',
          error
        );
      }
    },
    []
  );

  const updateHealthData = useCallback(
    async (equipmentOverride) => {
      const equipmentTarget =
        equipmentOverride || selectedEquipment;

      const id = equipmentTarget?.id;

      if (!id) return;

      const requestId =
        ++requestTrackerRef.current.health;

      try {
        const data =
          await equipmentService.current.getSensorData(id);

        const health = evaluateEquipmentHealth(
          equipmentTarget.type,
          data
        );

        if (
          !mountedRef.current ||
          requestTrackerRef.current.health !== requestId ||
          selectedIdRef.current !== id
        ) {
          return;
        }

        setHealthData(health);
      } catch (error) {
        console.error(
          'Failed to update health data:',
          error
        );
      }
    },
    [selectedEquipment]
  );

  const loadMaintenanceHistory = useCallback(
    async (equipmentId) => {
      if (!equipmentId) return;

      const requestId =
        ++requestTrackerRef.current.maintenance;

      try {
        const history =
          await equipmentService.current.getMaintenanceHistory(
            equipmentId
          );

        if (
          !mountedRef.current ||
          requestTrackerRef.current.maintenance !== requestId ||
          selectedIdRef.current !== equipmentId
        ) {
          return;
        }

        setMaintenanceHistory(history);
      } catch (error) {
        console.error(
          'Failed to load maintenance history:',
          error
        );
      }
    },
    []
  );

  const loadAnalytics = useCallback(
    async (equipmentId, range) => {
      if (!equipmentId) return;

      const requestId =
        ++requestTrackerRef.current.analytics;

      try {
        const data =
          await equipmentService.current.getEquipmentAnalytics(
            equipmentId,
            range || timeRange
          );

        if (
          !mountedRef.current ||
          requestTrackerRef.current.analytics !== requestId ||
          selectedIdRef.current !== equipmentId
        ) {
          return;
        }

        setAnalytics(data);
      } catch (error) {
        console.error(
          'Failed to load analytics:',
          error
        );
      }
    },
    [timeRange]
  );

  const loadAlerts = useCallback(
    async (equipmentId) => {
      if (!equipmentId) return;

      const requestId =
        ++requestTrackerRef.current.alerts;

      try {
        const alertData =
          await equipmentService.current.getPredictiveAlerts(
            equipmentId
          );

        if (
          !mountedRef.current ||
          requestTrackerRef.current.alerts !== requestId ||
          selectedIdRef.current !== equipmentId
        ) {
          return;
        }

        setAlerts(alertData);
      } catch (error) {
        console.error(
          'Failed to load alerts:',
          error
        );
      }
    },
    []
  );

  const getHealthColor = (score) => {
    if (score >= 90) return '#16a34a';
    if (score >= 75) return '#84cc16';
    if (score >= 60) return '#f59e0b';
    if (score >= 40) return '#ef9800';
    return '#dc2626';
  };

  const getAlertColor = (type) => {
    switch (type) {
      case 'critical':
        return '#dc2626';
      case 'warning':
        return '#f59e0b';
      case 'info':
        return '#3b82f6';
      default:
        return '#6b7280';
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR'
    }).format(amount || 0);
  };

  return (
    <div style={{ 
      maxWidth: "1400px", 
      margin: "40px auto", 
      padding: "24px", 
      background: "#fff", 
      borderRadius: "16px", 
      boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
      position: "relative"
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ color: "#16a34a", fontSize: "28px", margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
          🚜 Smart Equipment Management
          <span style={{ 
            fontSize: '14px', 
            color: realTimeMode ? '#16a34a' : '#6b7280',
            padding: '4px 8px',
            borderRadius: '4px',
            backgroundColor: realTimeMode ? '#dcfce7' : '#f3f4f6'
          }}>
            {realTimeMode ? '🟢 Live' : '⏸️ Offline'}
          </span>
        </h2>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={() => setShowAddEquipment(true)}
            style={{
              padding: '8px 16px',
              backgroundColor: '#16a34a',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            + Add Equipment
          </button>
          <button
            onClick={() => setRealTimeMode(!realTimeMode)}
            style={{
              padding: '8px 16px',
              backgroundColor: realTimeMode ? '#16a34a' : '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            {realTimeMode ? '📡 Real-time' : '📊 Historical'}
          </button>
          <button
            onClick={onClose}
            style={{
              padding: '8px 12px',
              backgroundColor: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Equipment Selection */}
      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '18px', marginBottom: '16px', color: '#111' }}>Equipment Fleet</h3>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
          gap: '16px' 
        }}>
          {equipment.map((eq) => (
            <div
              key={eq.id}
              onClick={() => selectEquipment(eq)}
              style={{
                padding: '16px',
                border: selectedEquipment?.id === eq.id ? '2px solid #16a34a' : '1px solid #e5e7eb',
                borderRadius: '12px',
                backgroundColor: selectedEquipment?.id === eq.id ? '#f0fdf4' : 'white',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '24px' }}>{getEquipmentInfo(eq.type)?.icon || '🚜'}</span>
                  <div>
                    <h4 style={{ margin: 0, fontSize: '16px', color: '#111' }}>{eq.name}</h4>
                    <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#6b7280' }}>
                      {getEquipmentInfo(eq.type)?.name || 'Equipment'} • {eq.status}
                    </p>
                  </div>
                </div>
                <div style={{ 
                  width: '12px', 
                  height: '12px', 
                  borderRadius: '50%', 
                  backgroundColor: eq.status === 'operational' ? '#16a34a' : '#f59e0b' 
                }} />
              </div>
              <div style={{ fontSize: '12px', color: '#6b7280' }}>
                Engine Hours: {eq.engine_hours || 0}h
              </div>
              {eq.location && (
                <div style={{ fontSize: '12px', color: '#6b7280' }}>
                  📍 {eq.location.lat.toFixed(4)}, {eq.location.lng.toFixed(4)}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Selected Equipment Details */}
      {selectedEquipment && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          {/* Real-time Monitoring */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: '#f8fafc', 
            borderRadius: '12px', 
            border: '1px solid #e2e8f0' 
          }}>
            <h3 style={{ fontSize: '18px', marginBottom: '16px', color: '#111', display: 'flex', alignItems: 'center', gap: '8px' }}>
              📊 Real-time Monitoring
              {healthData.status && (
                <span style={{
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontSize: '12px',
                  fontWeight: 'bold',
                  backgroundColor: getHealthColor(healthData.overall),
                  color: 'white'
                }}>
                  {healthData.status.toUpperCase()}
                </span>
              )}
            </h3>
            
            {/* Sensor Data */}
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ fontSize: '14px', marginBottom: '12px', color: '#111' }}>Live Sensor Data</h4>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                gap: '12px' 
              }}>
                {Object.entries(sensorData).map(([key, value]) => (
                  <div key={key} style={{ 
                    padding: '12px', 
                    backgroundColor: 'white', 
                    borderRadius: '8px',
                    border: '1px solid #e5e7eb'
                  }}>
                    <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>
                      {key.replace(/_/g, ' ').toUpperCase()}
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#111' }}>
                      {typeof value === 'number' ? value.toFixed(1) : value}
                      {key.includes('temperature') && '°C'}
                      {key.includes('pressure') && ' PSI'}
                      {key.includes('efficiency') && '%'}
                      {key.includes('consumption') && ' L/h'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Health Score */}
            {healthData.overall && (
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ fontSize: '14px', marginBottom: '12px', color: '#111' }}>Equipment Health</h4>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '16px',
                  padding: '16px',
                  backgroundColor: 'white',
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '36px', fontWeight: 'bold', color: getHealthColor(healthData.overall) }}>
                      {healthData.overall}%
                    </div>
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
                      Overall Health Score
                    </div>
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '8px' }}>
                      Status: {healthData.status}
                    </div>
                    {healthData.recommendations && (
                      <div style={{ marginTop: '8px' }}>
                        <h5 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#111' }}>Recommendations:</h5>
                        {healthData.recommendations.map((rec, index) => (
                          <div key={index} style={{ 
                            padding: '8px', 
                            backgroundColor: '#f0fdf4', 
                            borderRadius: '4px', 
                            marginBottom: '4px',
                            borderLeft: `3px solid ${rec.priority === 'high' ? '#dc2626' : rec.priority === 'medium' ? '#f59e0b' : '#3b82f6'}`
                          }}>
                            <div style={{ fontSize: '12px', fontWeight: '500', color: '#111' }}>
                              {rec.message}
                            </div>
                            <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
                              Est. Cost: {rec.estimatedCost}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Alerts */}
            {alerts.length > 0 && (
              <div>
                <h4 style={{ fontSize: '14px', marginBottom: '12px', color: '#111' }}>Active Alerts</h4>
                <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                  {alerts.map((alert) => (
                    <div key={alert.id} style={{
                      padding: '12px',
                      marginBottom: '8px',
                      backgroundColor: 'white',
                      borderRadius: '8px',
                      borderLeft: `4px solid ${getAlertColor(alert.type)}`
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: '500', color: '#111' }}>{alert.message}</span>
                        <span style={{
                          padding: '2px 6px',
                          backgroundColor: getAlertColor(alert.type),
                          color: 'white',
                          borderRadius: '4px',
                          fontSize: '10px'
                        }}>
                          {alert.type.toUpperCase()}
                        </span>
                      </div>
                      <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
                        {alert.recommendation}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Analytics Dashboard */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: '#f8fafc', 
            borderRadius: '12px', 
            border: '1px solid #e2e8f0' 
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '18px', margin: 0, color: '#111' }}>📈 Analytics Dashboard</h3>
              <select
                value={timeRange}
                onChange={(e) => {
                  setTimeRange(e.target.value);
                  if (selectedEquipment) {
                    loadAnalytics(selectedEquipment.id, e.target.value);
                  }
                }}
                style={{
                  padding: '8px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px'
                }}
              >
                <option value="1d">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="90d">Last 90 Days</option>
              </select>
            </div>

            {/* Analytics Charts */}
            {analytics.historicalData && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                {/* Utilization Chart */}
                <div style={{ backgroundColor: 'white', padding: '16px', borderRadius: '8px' }}>
                  <h4 style={{ fontSize: '14px', marginBottom: '12px', color: '#111' }}>Equipment Utilization</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={analytics.historicalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="utilization" 
                        stroke="#16a34a" 
                        strokeWidth={2}
                        dot={{ fill: "#16a34a" }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Fuel Consumption Chart */}
                <div style={{ backgroundColor: 'white', padding: '16px', borderRadius: '8px' }}>
                  <h4 style={{ fontSize: '14px', marginBottom: '12px', color: '#111' }}>Fuel Consumption</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={analytics.historicalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="fuelConsumption" fill="#f59e0b" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Maintenance Costs */}
                <div style={{ backgroundColor: 'white', padding: '16px', borderRadius: '8px' }}>
                  <h4 style={{ fontSize: '14px', marginBottom: '12px', color: '#111' }}>Maintenance Costs</h4>
                  <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#dc2626' }}>
                        {formatCurrency(analytics.maintenanceCost)}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280' }}>Total Cost</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#16a34a' }}>
                        {formatCurrency(analytics.maintenanceCost / (analytics.historicalData.length || 1))}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280' }}>Avg Cost</div>
                    </div>
                  </div>
                </div>

                {/* Operating Hours */}
                <div style={{ backgroundColor: 'white', padding: '16px', borderRadius: '8px' }}>
                  <h4 style={{ fontSize: '14px', marginBottom: '12px', color: '#111' }}>Operating Hours</h4>
                  <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#111' }}>
                        {analytics.operatingHours}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280' }}>Total Hours</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#16a34a' }}>
                        {(analytics.operatingHours / (analytics.historicalData.length || 1)).toFixed(1)}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280' }}>Daily Avg</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Add Equipment Modal */}
{showAddEquipment && (
  <div style={{
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000
  }}>
    <div style={{
      position: 'relative',   //  added
      backgroundColor: 'white',
      padding: '32px',
      borderRadius: '16px',
      maxWidth: '500px',
      width: '90%'
    }}>
      <h3 style={{ marginBottom: '24px', color: '#111' }}>Add New Equipment</h3>
      <button
        onClick={() => setShowAddEquipment(false)}
        style={{
          position: 'absolute',   //  changed from relative
          top: '16px',
          right: '16px',
          backgroundColor: '#f3f4f6',
          border: '1px solid #d1d5db',
          borderRadius: '8px',
          fontSize: '16px',
          cursor: 'pointer'
        }}
      >
        ✕
      </button>
      <div style={{ textAlign: 'center', color: '#6b7280' }}>
        Equipment registration form would go here
      </div>
    </div>
  </div>
)}

{/* Maintenance Modal */}
{showMaintenance && selectedEquipment && (
  <div style={{
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000
  }}>
    <div style={{
      position: 'relative',   //  added
      backgroundColor: 'white',
      padding: '32px',
      borderRadius: '16px',
      maxWidth: '600px',
      width: '90%'
    }}>
      <h3 style={{ marginBottom: '24px', color: '#111' }}>Schedule Maintenance</h3>
      <button
        onClick={() => setShowMaintenance(false)}
        style={{
          position: 'absolute',
          top: '16px',
          right: '16px',
          backgroundColor: '#f3f4f6',
          border: '1px solid #d1d5db',
          borderRadius: '8px',
          fontSize: '16px',
          cursor: 'pointer'
        }}
      >
        ✕
      </button>
      <div style={{ textAlign: 'center', color: '#6b7280' }}>
        Maintenance scheduling form would go here
      </div>
    </div>
  </div>
)}

    </div>
  );
}
