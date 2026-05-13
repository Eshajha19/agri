// Comprehensive equipment database with maintenance schedules and IoT integration
export const equipmentDatabase = {
  equipmentTypes: {
    tractor: {
      name: "Tractor",
      icon: "🚜",
      category: "heavy_machinery",
      sensors: ["engine_hours", "fuel_consumption", "temperature", "vibration", "oil_pressure"],
      maintenanceInterval: 250, // hours
      criticalParameters: {
        engine_temperature: { min: 70, max: 110, unit: "°C" },
        oil_pressure: { min: 20, max: 60, unit: "PSI" },
        vibration: { min: 0, max: 10, unit: "mm/s" },
        fuel_efficiency: { min: 2, max: 8, unit: "l/hr" }
      }
    },
    harvester: {
      name: "Harvester",
      icon: "🌾",
      category: "harvesting_equipment",
      sensors: ["engine_hours", "fuel_consumption", "grain_moisture", "temperature", "hydraulic_pressure"],
      maintenanceInterval: 200, // hours
      criticalParameters: {
        engine_temperature: { min: 70, max: 110, unit: "°C" },
        hydraulic_pressure: { min: 1000, max: 3000, unit: "PSI" },
        grain_loss: { min: 0, max: 5, unit: "%" },
        fuel_efficiency: { min: 10, max: 25, unit: "l/hr" }
      }
    },
    irrigation_system: {
      name: "Irrigation System",
      icon: "💧",
      category: "irrigation_equipment",
      sensors: ["water_pressure", "flow_rate", "soil_moisture", "pump_status", "energy_consumption"],
      maintenanceInterval: 500, // hours
      criticalParameters: {
        water_pressure: { min: 20, max: 80, unit: "PSI" },
        flow_rate: { min: 50, max: 500, unit: "l/min" },
        energy_efficiency: { min: 60, max: 95, unit: "%" }
      }
    },
    plow: {
      name: "Plow",
      icon: "🔧",
      category: "tillage_equipment",
      sensors: ["engine_hours", "fuel_consumption", "depth", "angle", "temperature"],
      maintenanceInterval: 150, // hours
      criticalParameters: {
        engine_temperature: { min: 70, max: 110, unit: "°C" },
        hydraulic_pressure: { min: 500, max: 2000, unit: "PSI" },
        plow_depth: { min: 10, max: 40, unit: "cm" }
      }
    },
    sprayer: {
      name: "Sprayer",
      icon: "🚿",
      category: "application_equipment",
      sensors: ["engine_hours", "fuel_consumption", "flow_rate", "pressure", "tank_level"],
      maintenanceInterval: 100, // hours
      criticalParameters: {
        spray_pressure: { min: 20, max: 60, unit: "PSI" },
        flow_rate: { min: 5, max: 50, unit: "l/min" },
        coverage_efficiency: { min: 70, max: 95, unit: "%" }
      }
    }
  },

  maintenanceTasks: {
    oil_change: {
      name: "Oil Change",
      priority: "high",
      estimatedTime: 2, // hours
      requiredParts: ["oil_filter", "engine_oil", "drain_plug"],
      costRange: { min: 500, max: 2000, unit: "₹" }
    },
    filter_replacement: {
      name: "Filter Replacement",
      priority: "medium",
      estimatedTime: 1, // hours
      requiredParts: ["air_filter", "fuel_filter", "oil_filter"],
      costRange: { min: 200, max: 800, unit: "₹" }
    },
    belt_adjustment: {
      name: "Belt Adjustment",
      priority: "low",
      estimatedTime: 0.5, // hours
      requiredParts: ["v_belt", "tensioner"],
      costRange: { min: 100, max: 400, unit: "₹" }
    },
    hydraulic_service: {
      name: "Hydraulic System Service",
      priority: "high",
      estimatedTime: 4, // hours
      requiredParts: ["hydraulic_oil", "seals_kit", "filters"],
      costRange: { min: 1000, max: 5000, unit: "₹" }
    },
    tire_replacement: {
      name: "Tire Replacement",
      priority: "medium",
      estimatedTime: 3, // hours
      requiredParts: ["tire", "tube", "rim"],
      costRange: { min: 2000, max: 15000, unit: "₹" }
    }
  },

  alerts: {
    critical: {
      threshold: 0.9,
      actions: ["immediate_shutdown", "operator_alert", "supervisor_notification"]
    },
    warning: {
      threshold: 0.7,
      actions: ["schedule_maintenance", "operator_warning", "log_event"]
    },
    info: {
      threshold: 0.5,
      actions: ["log_data", "update_dashboard", "periodic_report"]
    }
  }
};

export const getEquipmentInfo = (equipmentType) => {
  return equipmentDatabase.equipmentTypes[equipmentType] || null;
};

export const getMaintenanceSchedule = (equipmentType, engineHours) => {
  const equipment = getEquipmentInfo(equipmentType);
  if (!equipment) return null;

  const interval = equipment.maintenanceInterval;
  const nextMaintenance = interval - (engineHours % interval);
  
  return {
    nextMaintenanceIn: nextMaintenance,
    nextMaintenanceDate: new Date(Date.now() + (nextMaintenance * 3600000)), // approximate
    maintenanceDue: nextMaintenance <= 10, // due within 10 hours
    tasks: generateMaintenanceTasks(equipmentType, engineHours)
  };
};

const generateMaintenanceTasks = (equipmentType, engineHours) => {
  const tasks = [];
  const equipment = getEquipmentInfo(equipmentType);
  
  if (engineHours > 200) tasks.push(equipmentDatabase.maintenanceTasks.oil_change);
  if (engineHours > 150) tasks.push(equipmentDatabase.maintenanceTasks.filter_replacement);
  if (engineHours > 100) tasks.push(equipmentDatabase.maintenanceTasks.belt_adjustment);
  if (engineHours > 300) tasks.push(equipmentDatabase.maintenanceTasks.hydraulic_service);
  if (engineHours > 500) tasks.push(equipmentDatabase.maintenanceTasks.tire_replacement);
  
  return tasks;
};

export const evaluateEquipmentHealth = (equipmentType, sensorData) => {
  const equipment = getEquipmentInfo(equipmentType);
  if (!equipment || !sensorData) return null;

  const criticalParams = equipment.criticalParameters;
  const healthScore = calculateHealthScore(criticalParams, sensorData);
  
  return {
    overall: healthScore,
    status: getHealthStatus(healthScore),
    alerts: generateAlerts(criticalParams, sensorData, healthScore),
    recommendations: generateRecommendations(criticalParams, sensorData, healthScore)
  };
};

const calculateHealthScore = (criticalParams, sensorData) => {
  let totalScore = 100;
  let paramCount = 0;
  
  Object.keys(criticalParams).forEach(param => {
    if (sensorData[param] !== undefined) {
      const { min, max } = criticalParams[param];
      const value = sensorData[param];
      
      let paramScore = 100;
      if (value < min || value > max) {
        const deviation = Math.min(Math.abs(value - min), Math.abs(value - max));
        const range = max - min;
        paramScore = Math.max(0, 100 - (deviation / range * 100));
      }
      
      totalScore += paramScore;
      paramCount++;
    }
  });
  
  return paramCount > 0 ? Math.round(totalScore / paramCount) : 100;
};

const getHealthStatus = (score) => {
  if (score >= 90) return "excellent";
  if (score >= 75) return "good";
  if (score >= 60) return "fair";
  if (score >= 40) return "poor";
  return "critical";
};

const generateAlerts = (criticalParams, sensorData, healthScore) => {
  const alerts = [];
  
  Object.keys(criticalParams).forEach(param => {
    if (sensorData[param] !== undefined) {
      const { min, max } = criticalParams[param];
      const value = sensorData[param];
      
      if (value < min || value > max) {
        alerts.push({
          type: "parameter_out_of_range",
          parameter: param,
          severity: healthScore < 40 ? "critical" : healthScore < 70 ? "warning" : "info",
          message: `${param.replace(/_/g, ' ')} is out of normal range`,
          value,
          range: criticalParams[param]
        });
      }
    }
  });
  
  return alerts;
};

const generateRecommendations = (criticalParams, sensorData, healthScore) => {
  const recommendations = [];
  
  if (healthScore < 60) {
    recommendations.push({
      type: "immediate_maintenance",
      priority: "high",
      message: "Schedule immediate maintenance to prevent equipment failure",
      estimatedCost: "₹2000-5000"
    });
  }
  
  if (healthScore < 80) {
    recommendations.push({
      type: "operator_training",
      priority: "medium",
      message: "Consider operator training for better efficiency",
      estimatedCost: "₹500-1000"
    });
  }
  
  // Check for efficiency issues
  if (sensorData.fuel_efficiency !== undefined) {
    const equipment = getEquipmentInfo(Object.keys(equipmentDatabase.equipmentTypes).find(key => 
      equipmentDatabase.equipmentTypes[key].sensors.includes("fuel_efficiency")
    ));
    
    if (equipment && sensorData.fuel_efficiency < equipment.criticalParameters.fuel_efficiency.min) {
      recommendations.push({
        type: "efficiency_improvement",
        priority: "medium",
        message: "Fuel efficiency is below optimal range - check engine tuning",
        estimatedCost: "₹1000-3000"
      });
    }
  }
  
  return recommendations;
};

export const saveEquipmentData = (equipmentId, data) => {
  try {
    const equipmentData = JSON.parse(localStorage.getItem('equipmentData') || '{}');
    equipmentData[equipmentId] = {
      ...data,
      lastUpdated: new Date().toISOString()
    };
    localStorage.setItem('equipmentData', JSON.stringify(equipmentData));
    return true;
  } catch (error) {
    return false;
  }
};

export const getEquipmentData = (equipmentId) => {
  try {
    const equipmentData = JSON.parse(localStorage.getItem('equipmentData') || '{}');
    return equipmentData[equipmentId] || null;
  } catch (error) {
    return null;
  }
};

export const getAllEquipmentData = () => {
  try {
    return JSON.parse(localStorage.getItem('equipmentData') || '{}');
  } catch (error) {
    return {};
  }
};

export const generateMaintenanceReport = (equipmentId) => {
  const equipment = getEquipmentData(equipmentId);
  if (!equipment) return null;
  
  const maintenanceHistory = equipment.maintenanceHistory || [];
  const totalCost = maintenanceHistory.reduce((sum, record) => sum + (record.cost || 0), 0);
  const totalDowntime = maintenanceHistory.reduce((sum, record) => sum + (record.downtime || 0), 0);
  
  return {
    equipmentId,
    totalMaintenanceCost: totalCost,
    totalDowntime,
    maintenanceCount: maintenanceHistory.length,
    lastMaintenance: maintenanceHistory[maintenanceHistory.length - 1],
    averageCostPerMaintenance: totalCost / maintenanceHistory.length,
    uptime: calculateUptime(equipment.engineHours || 0, totalDowntime)
  };
};

const calculateUptime = (engineHours, downtime) => {
  const totalOperatingHours = engineHours + downtime;
  return totalOperatingHours > 0 ? ((engineHours / totalOperatingHours) * 100).toFixed(1) : 0;
};
