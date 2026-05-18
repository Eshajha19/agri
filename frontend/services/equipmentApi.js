// Equipment Management API Service
import { getAllEquipmentData, saveEquipmentData, getEquipmentData } from '../utils/equipmentDatabase';

class EquipmentService {
  constructor() {
    this.baseURL = import.meta.env.VITE_EQUIPMENT_API_URL || 'http://localhost:8000/api/equipment';
    this.mockMode = !import.meta.env.VITE_EQUIPMENT_API_URL;
  }

  // Simulate real-time sensor data
  async getSensorData(equipmentId) {
    if (this.mockMode) {
      return this.generateMockSensorData(equipmentId);
    }

    try {
      const response = await fetch(`${this.baseURL}/${equipmentId}/sensors`);
      if (!response.ok) throw new Error(`Failed to fetch sensor data: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching sensor data:', error);
      return this.generateMockSensorData(equipmentId);
    }
  }

  // Generate realistic mock sensor data
  generateMockSensorData(equipmentId) {
    const equipment = getEquipmentData(equipmentId);
    const baseData = {
      timestamp: new Date().toISOString(),
      equipmentId,
      status: 'operational'
    };

    // Add equipment-specific sensor data
    switch (equipment?.type) {
      case 'tractor':
        return {
          ...baseData,
          engine_hours: Math.floor(Math.random() * 500) + 100,
          fuel_consumption: Math.random() * 3 + 2,
          engine_temperature: Math.random() * 30 + 80,
          oil_pressure: Math.random() * 20 + 30,
          vibration: Math.random() * 5,
          location: {
            lat: 28.6139 + (Math.random() - 0.01),
            lng: 77.2090 + (Math.random() - 0.01)
          }
        };
      
      case 'harvester':
        return {
          ...baseData,
          engine_hours: Math.floor(Math.random() * 400) + 150,
          fuel_consumption: Math.random() * 8 + 15,
          engine_temperature: Math.random() * 25 + 85,
          grain_moisture: Math.random() * 5 + 12,
          hydraulic_pressure: Math.random() * 1000 + 1500,
          location: {
            lat: 28.6139 + (Math.random() - 0.01),
            lng: 77.2090 + (Math.random() - 0.01)
          }
        };
      
      case 'irrigation_system':
        return {
          ...baseData,
          water_pressure: Math.random() * 30 + 40,
          flow_rate: Math.random() * 200 + 150,
          soil_moisture: Math.random() * 20 + 30,
          pump_status: Math.random() > 0.5 ? 'running' : 'idle',
          energy_consumption: Math.random() * 5 + 10,
          location: {
            lat: 28.6139 + (Math.random() - 0.01),
            lng: 77.2090 + (Math.random() - 0.01)
          }
        };
      
      default:
        return {
          ...baseData,
          engine_hours: Math.floor(Math.random() * 300) + 50,
          fuel_consumption: Math.random() * 2 + 1,
          engine_temperature: Math.random() * 20 + 75,
          location: {
            lat: 28.6139 + (Math.random() - 0.01),
            lng: 77.2090 + (Math.random() - 0.01)
          }
        };
    }
  }

  // Get equipment list
  async getEquipmentList() {
    if (this.mockMode) {
      return this.generateMockEquipmentList();
    }

    try {
      const response = await fetch(`${this.baseURL}`);
      if (!response.ok) throw new Error(`Failed to fetch equipment list: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching equipment list:', error);
      return this.generateMockEquipmentList();
    }
  }

  // Generate mock equipment list
  generateMockEquipmentList() {
    const storedData = getAllEquipmentData();
    const mockEquipment = [
      {
        id: 'tractor_001',
        name: 'John Deere 5075E',
        type: 'tractor',
        status: 'operational',
        engine_hours: 342,
        location: { lat: 28.6139, lng: 77.2090 },
        lastMaintenance: '2024-01-15',
        nextMaintenance: '2024-03-15'
      },
      {
        id: 'harvester_001',
        name: 'Claas LEXION 750',
        type: 'harvester',
        status: 'maintenance',
        engine_hours: 187,
        location: { lat: 28.6140, lng: 77.2085 },
        lastMaintenance: '2024-02-10',
        nextMaintenance: '2024-03-10'
      },
      {
        id: 'irrigation_001',
        name: 'Valley Center Pivot',
        type: 'irrigation_system',
        status: 'operational',
        engine_hours: 523,
        location: { lat: 28.6135, lng: 77.2095 },
        lastMaintenance: '2024-01-20',
        nextMaintenance: '2024-04-20'
      }
    ];

    // Merge with stored data
    return mockEquipment.map(eq => ({
      ...eq,
      ...(storedData[eq.id] || {})
    }));
  }

  // Update equipment data
  async updateEquipment(equipmentId, data) {
    if (this.mockMode) {
      return this.mockUpdateEquipment(equipmentId, data);
    }

    try {
      const response = await fetch(`${this.baseURL}/${equipmentId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) throw new Error(`Failed to update equipment: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error updating equipment:', error);
      return this.mockUpdateEquipment(equipmentId, data);
    }
  }

  // Mock update equipment
  mockUpdateEquipment(equipmentId, data) {
    const success = saveEquipmentData(equipmentId, data);
    return {
      success,
      message: success ? 'Equipment updated successfully' : 'Failed to update equipment'
    };
  }

  // Get maintenance history
  async getMaintenanceHistory(equipmentId) {
    if (this.mockMode) {
      return this.generateMockMaintenanceHistory(equipmentId);
    }

    try {
      const response = await fetch(`${this.baseURL}/${equipmentId}/maintenance`);
      if (!response.ok) throw new Error(`Failed to fetch maintenance history: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching maintenance history:', error);
      return this.generateMockMaintenanceHistory(equipmentId);
    }
  }

  // Generate mock maintenance history
  generateMockMaintenanceHistory(equipmentId) {
    const equipment = getEquipmentData(equipmentId);
    const history = equipment?.maintenanceHistory || [];
    
    // Generate some mock history if empty
    if (history.length === 0) {
      return [
        {
          id: 'maint_001',
          date: '2024-01-15',
          type: 'oil_change',
          cost: 1500,
          downtime: 4,
          technician: 'John Doe',
          notes: 'Regular oil change performed'
        },
        {
          id: 'maint_002',
          date: '2024-02-10',
          type: 'filter_replacement',
          cost: 800,
          downtime: 2,
          technician: 'Jane Smith',
          notes: 'Air and fuel filters replaced'
        }
      ];
    }
    
    return history;
  }

  // Schedule maintenance
  async scheduleMaintenance(equipmentId, maintenanceData) {
    if (this.mockMode) {
      return this.mockScheduleMaintenance(equipmentId, maintenanceData);
    }

    try {
      const response = await fetch(`${this.baseURL}/${equipmentId}/maintenance`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(maintenanceData)
      });

      if (!response.ok) throw new Error(`Failed to schedule maintenance: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error scheduling maintenance:', error);
      return this.mockScheduleMaintenance(equipmentId, maintenanceData);
    }
  }

  // Mock schedule maintenance
  mockScheduleMaintenance(equipmentId, maintenanceData) {
    const equipment = getEquipmentData(equipmentId);
    if (!equipment) return { success: false, message: 'Equipment not found' };

    const maintenanceHistory = equipment.maintenanceHistory || [];
    const newMaintenance = {
      id: `maint_${Date.now()}`,
      date: new Date().toISOString().split('T')[0],
      ...maintenanceData,
      status: 'scheduled'
    };

    maintenanceHistory.push(newMaintenance);
    saveEquipmentData(equipmentId, { ...equipment, maintenanceHistory });

    return {
      success: true,
      message: 'Maintenance scheduled successfully',
      maintenanceId: newMaintenance.id
    };
  }

  // Get predictive maintenance alerts
  async getPredictiveAlerts(equipmentId) {
    if (this.mockMode) {
      return this.generateMockAlerts(equipmentId);
    }

    try {
      const response = await fetch(`${this.baseURL}/${equipmentId}/alerts`);
      if (!response.ok) throw new Error(`Failed to fetch alerts: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching alerts:', error);
      return this.generateMockAlerts(equipmentId);
    }
  }

  // Generate mock alerts
  generateMockAlerts(equipmentId) {
    const equipment = getEquipmentData(equipmentId);
    const sensorData = this.generateMockSensorData(equipmentId);
    
    // Simulate some alerts based on sensor data
    const alerts = [];

    if (sensorData.engine_temperature > 100) {
      alerts.push({
        id: `alert_${Date.now()}_1`,
        type: 'critical',
        message: 'Engine temperature is critically high',
        recommendation: 'Immediate inspection required',
        timestamp: new Date().toISOString()
      });
    }

    if (sensorData.vibration > 8) {
      alerts.push({
        id: `alert_${Date.now()}_2`,
        type: 'warning',
        message: 'Excessive vibration detected',
        recommendation: 'Check for loose components',
        timestamp: new Date().toISOString()
      });
    }

    if (sensorData.fuel_consumption > 5) {
      alerts.push({
        id: `alert_${Date.now()}_3`,
        type: 'info',
        message: 'High fuel consumption detected',
        recommendation: 'Consider engine tuning',
        timestamp: new Date().toISOString()
      });
    }

    return alerts;
  }

  // Get equipment analytics
  async getEquipmentAnalytics(equipmentId, timeRange = '7d') {
    if (this.mockMode) {
      return this.generateMockAnalytics(equipmentId, timeRange);
    }

    try {
      const response = await fetch(`${this.baseURL}/${equipmentId}/analytics?range=${timeRange}`);
      if (!response.ok) throw new Error(`Failed to fetch analytics: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching analytics:', error);
      return this.generateMockAnalytics(equipmentId, timeRange);
    }
  }

  // Generate mock analytics
  generateMockAnalytics(equipmentId, timeRange) {
    const equipment = getEquipmentData(equipmentId);
    if (!equipment) return null;

    // Generate time series data for the specified range
    const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 1;
    const analytics = {
      equipmentId,
      timeRange,
      utilization: Math.random() * 30 + 60, // percentage
      fuelEfficiency: Math.random() * 2 + 3,
      operatingHours: Math.floor(Math.random() * 8) + 6,
      downtime: Math.random() * 2,
      maintenanceCost: Math.random() * 1000 + 500,
      productivity: Math.random() * 20 + 70 // percentage
    };

    // Generate historical data points
    analytics.historicalData = [];
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i - 1));
      
      analytics.historicalData.push({
        date: date.toISOString().split('T')[0],
        utilization: Math.random() * 30 + 60,
        fuelConsumption: Math.random() * 3 + 2,
        operatingHours: Math.random() * 10 + 4,
        alerts: Math.random() > 0.7 ? 1 : 0
      });
    }

    return analytics;
  }
}

export default new EquipmentService();
