"""
Real-time weather alerts module with crop-specific warnings.
Uses Open-Meteo API (free, no API key required) for weather data.
"""

import os
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# Constants and Enums
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WeatherCondition(Enum):
    """Weather conditions that require alerts"""
    EXTREME_HEAT = "extreme_heat"  # Temp > 40°C
    FROST = "frost"  # Temp < 0°C
    HEAVY_RAIN = "heavy_rain"  # Rain > 50mm
    DROUGHT = "drought"  # No rain expected for 7+ days
    STRONG_WIND = "strong_wind"  # Wind > 40 km/h
    HAIL = "hail"  # Hail possible
    FLOOD_RISK = "flood_risk"  # Heavy rain + high humidity


# Crop-specific thresholds
CROP_THRESHOLDS = {
    "rice": {
        "optimal_temp_min": 15,
        "optimal_temp_max": 35,
        "critical_temp_max": 40,
        "critical_temp_min": 5,
        "rainfall_min_mm": 2,  # Needs regular water
        "sensitive_to": ["EXTREME_HEAT", "FLOOD_RISK", "STRONG_WIND"],
    },
    "wheat": {
        "optimal_temp_min": 10,
        "optimal_temp_max": 25,
        "critical_temp_max": 35,
        "critical_temp_min": -5,  # Can tolerate frost
        "rainfall_min_mm": 1,
        "sensitive_to": ["FROST", "HEAVY_RAIN", "EXTREME_HEAT"],
    },
    "maize": {
        "optimal_temp_min": 15,
        "optimal_temp_max": 30,
        "critical_temp_max": 38,
        "critical_temp_min": 5,
        "rainfall_min_mm": 1.5,
        "sensitive_to": ["EXTREME_HEAT", "DROUGHT", "STRONG_WIND"],
    },
    "cotton": {
        "optimal_temp_min": 20,
        "optimal_temp_max": 35,
        "critical_temp_max": 42,
        "critical_temp_min": 10,
        "rainfall_min_mm": 1,
        "sensitive_to": ["EXTREME_HEAT", "HEAVY_RAIN", "HAIL"],
    },
    "sugarcane": {
        "optimal_temp_min": 20,
        "optimal_temp_max": 30,
        "critical_temp_max": 38,
        "critical_temp_min": 10,
        "rainfall_min_mm": 2,
        "sensitive_to": ["DROUGHT", "EXTREME_HEAT", "STRONG_WIND"],
    },
    "potato": {
        "optimal_temp_min": 15,
        "optimal_temp_max": 20,
        "critical_temp_max": 30,
        "critical_temp_min": -2,
        "rainfall_min_mm": 1.5,
        "sensitive_to": ["FROST", "HEAVY_RAIN", "DROUGHT"],
    },
    "tomato": {
        "optimal_temp_min": 18,
        "optimal_temp_max": 28,
        "critical_temp_max": 35,
        "critical_temp_min": 10,
        "rainfall_min_mm": 1,
        "sensitive_to": ["EXTREME_HEAT", "HEAVY_RAIN"],
    },
    "onion": {
        "optimal_temp_min": 15,
        "optimal_temp_max": 25,
        "critical_temp_max": 35,
        "critical_temp_min": 5,
        "rainfall_min_mm": 0.5,
        "sensitive_to": ["EXTREME_HEAT", "HEAVY_RAIN"],
    },
}

CROP_SPECIFIC_ACTIONS = {
    "rice": {
        "EXTREME_HEAT": "Apply mulch and increase irrigation frequency. Spray urea at 1% concentration.",
        "FLOOD_RISK": "Ensure drainage channels are clear. May cause blast disease.",
        "STRONG_WIND": "Install windbreaks if available.",
    },
    "wheat": {
        "FROST": "Alert: Frost may damage flowering stage. Use smoke pots if available.",
        "HEAVY_RAIN": "Risk of disease. Apply fungicide post-rain.",
        "EXTREME_HEAT": "Premature ripening possible. Harvest early if needed.",
    },
    "maize": {
        "EXTREME_HEAT": "Check irrigation at tasseling stage. Critical period.",
        "DROUGHT": "Plan supplemental irrigation.",
        "STRONG_WIND": "Risk of lodging. Ensure adequate nitrogen.",
    },
    "cotton": {
        "EXTREME_HEAT": "Increase irrigation. Risk of boll damage.",
        "HEAVY_RAIN": "Risk of pest outbreak after rain.",
        "HAIL": "Hail can severely damage fiber. Prepare pesticide spray.",
    },
    "sugarcane": {
        "DROUGHT": "Increase irrigation frequency to every 7-10 days.",
        "EXTREME_HEAT": "Risk of crop water stress.",
        "STRONG_WIND": "Risk of lodging.",
    },
    "potato": {
        "FROST": "Late blight risk after frost. Spray fungicide.",
        "HEAVY_RAIN": "High late blight risk. Apply preventive fungicides.",
        "DROUGHT": "Ensure regular irrigation for tuber development.",
    },
    "tomato": {
        "EXTREME_HEAT": "Risk of fruit cracking and flower drop. Increase irrigation.",
        "HEAVY_RAIN": "Fungal disease risk. Ensure good drainage.",
    },
    "onion": {
        "EXTREME_HEAT": "Risk of bulb damage. Shade may help.",
        "HEAVY_RAIN": "Root rot risk. Check drainage.",
    },
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class WeatherData:
    """Current weather data"""
    temperature: float
    humidity: float
    rainfall: float  # mm
    wind_speed: float  # km/h
    cloud_cover: int  # 0-100%
    weather_code: int  # WMO code
    timestamp: datetime
    location: str = "Unknown"

    def __str__(self):
        return (
            f"Temp: {self.temperature}°C, Humidity: {self.humidity}%, "
            f"Rain: {self.rainfall}mm, Wind: {self.wind_speed} km/h"
        )


@dataclass
class WeatherAlert:
    """Alert object"""
    id: str
    type: str = "alert"
    alert_type: str = "weather"
    severity: AlertSeverity
    condition: WeatherCondition
    title: str
    message: str
    crop: Optional[str] = None
    recommended_action: Optional[str] = None
    timestamp: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "type": self.type,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "condition": self.condition.value,
            "title": self.title,
            "message": self.message,
            "crop": self.crop,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


# ============================================================================
# Weather Service
# ============================================================================

class WeatherAlertsService:
    """
    Service for fetching weather data and generating alerts.
    Uses Open-Meteo API (free, no API key required).
    """

    # Open-Meteo API endpoints
    BASE_URL = "https://api.open-meteo.com/v1"
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1"

    def __init__(self, cache_duration_seconds: int = 600):
        """
        Initialize weather service.
        
        Args:
            cache_duration_seconds: Cache weather data for this duration
        """
        self.cache_duration = timedelta(seconds=cache_duration_seconds)
        self._weather_cache: Dict[str, tuple] = {}  # (data, timestamp)
        self.alert_history: List[WeatherAlert] = []

    async def get_coordinates(self, location: str) -> tuple:
        """
        Get latitude and longitude for a location.
        
        Args:
            location: City name or region name
            
        Returns:
            (latitude, longitude) tuple
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "name": location,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                }
                async with session.get(
                    f"{self.GEOCODING_URL}/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("results"):
                            result = data["results"][0]
                            return (result["latitude"], result["longitude"], result.get("name", location))
        except Exception as e:
            logger.error(f"Geocoding error for '{location}': {e}")
        return None

    async def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        location: str = "Unknown"
    ) -> Optional[WeatherData]:
        """
        Fetch current weather data for a location.
        Uses caching to reduce API calls.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            location: Location name for display
            
        Returns:
            WeatherData object or None if fetch fails
        """
        cache_key = f"{latitude},{longitude}"
        
        # Check cache
        if cache_key in self._weather_cache:
            cached_data, cached_time = self._weather_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data

        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,rainfall,weather_code,cloud_cover,wind_speed_10m",
                    "timezone": "auto",
                    "forecast_days": 1,
                    "hourly": "rainfall",
                }
                
                async with session.get(
                    f"{self.BASE_URL}/forecast",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        current = data.get("current", {})
                        
                        weather = WeatherData(
                            temperature=current.get("temperature_2m", 0),
                            humidity=current.get("relative_humidity_2m", 0),
                            rainfall=current.get("rainfall", 0),
                            wind_speed=current.get("wind_speed_10m", 0),
                            cloud_cover=current.get("cloud_cover", 0),
                            weather_code=current.get("weather_code", 0),
                            timestamp=datetime.now(),
                            location=location,
                        )
                        
                        # Cache the result
                        self._weather_cache[cache_key] = (weather, datetime.now())
                        return weather
        except asyncio.TimeoutError:
            logger.warning(f"Weather API timeout for {location}")
        except Exception as e:
            logger.error(f"Weather fetch error: {e}")
        
        return None

    def analyze_weather(
        self,
        weather: WeatherData,
        crop: Optional[str] = None
    ) -> List[WeatherAlert]:
        """
        Analyze weather data and generate alerts.
        
        Args:
            weather: WeatherData object
            crop: Crop type for crop-specific alerts
            
        Returns:
            List of WeatherAlert objects
        """
        alerts = []
        alert_id_counter = len(self.alert_history) + 1

        # Temperature alerts
        if weather.temperature > 40:
            alerts.append(WeatherAlert(
                id=f"weather_{alert_id_counter}",
                severity=AlertSeverity.CRITICAL,
                condition=WeatherCondition.EXTREME_HEAT,
                title="🔥 Extreme Heat Alert",
                message=f"Temperature reached {weather.temperature}°C. High risk of crop stress.",
                crop=crop,
                timestamp=weather.timestamp,
                expires_at=weather.timestamp + timedelta(hours=6),
            ))
            alert_id_counter += 1
        elif weather.temperature > 35 and crop in CROP_THRESHOLDS:
            thresholds = CROP_THRESHOLDS[crop]
            if weather.temperature > thresholds.get("critical_temp_max", 40):
                alerts.append(WeatherAlert(
                    id=f"weather_{alert_id_counter}",
                    severity=AlertSeverity.HIGH,
                    condition=WeatherCondition.EXTREME_HEAT,
                    title="⚠️ High Temperature Warning",
                    message=f"Temperature {weather.temperature}°C is above optimal range for {crop}.",
                    crop=crop,
                    timestamp=weather.timestamp,
                    expires_at=weather.timestamp + timedelta(hours=6),
                ))
                alert_id_counter += 1

        if weather.temperature < 0:
            alerts.append(WeatherAlert(
                id=f"weather_{alert_id_counter}",
                severity=AlertSeverity.CRITICAL,
                condition=WeatherCondition.FROST,
                title="❄️ Frost Alert",
                message=f"Temperature dropped to {weather.temperature}°C. Frost risk detected.",
                crop=crop,
                timestamp=weather.timestamp,
                expires_at=weather.timestamp + timedelta(hours=6),
            ))
            alert_id_counter += 1
        elif weather.temperature < 5 and crop in CROP_THRESHOLDS:
            thresholds = CROP_THRESHOLDS[crop]
            if weather.temperature < thresholds.get("critical_temp_min", 0):
                alerts.append(WeatherAlert(
                    id=f"weather_{alert_id_counter}",
                    severity=AlertSeverity.HIGH,
                    condition=WeatherCondition.FROST,
                    title="❄️ Low Temperature Warning",
                    message=f"Temperature {weather.temperature}°C may affect {crop}.",
                    crop=crop,
                    timestamp=weather.timestamp,
                    expires_at=weather.timestamp + timedelta(hours=12),
                ))
                alert_id_counter += 1

        # Rainfall alerts
        if weather.rainfall > 50:
            alerts.append(WeatherAlert(
                id=f"weather_{alert_id_counter}",
                severity=AlertSeverity.HIGH,
                condition=WeatherCondition.HEAVY_RAIN,
                title="🌧️ Heavy Rain Alert",
                message=f"Heavy rainfall ({weather.rainfall}mm) expected. Flood risk possible.",
                crop=crop,
                timestamp=weather.timestamp,
                expires_at=weather.timestamp + timedelta(hours=6),
            ))
            alert_id_counter += 1

            # Additional alert for flood-sensitive crops
            if crop in CROP_THRESHOLDS and "FLOOD_RISK" in CROP_THRESHOLDS[crop]["sensitive_to"]:
                alerts.append(WeatherAlert(
                    id=f"weather_{alert_id_counter}",
                    severity=AlertSeverity.HIGH,
                    condition=WeatherCondition.FLOOD_RISK,
                    title=f"🌊 Flood Risk for {crop.title()}",
                    message=f"Heavy rain may cause waterlogging. Ensure drainage for {crop}.",
                    crop=crop,
                    recommended_action=CROP_SPECIFIC_ACTIONS.get(crop, {}).get("FLOOD_RISK"),
                    timestamp=weather.timestamp,
                    expires_at=weather.timestamp + timedelta(hours=24),
                ))
                alert_id_counter += 1

        # Wind alerts
        if weather.wind_speed > 40:
            alerts.append(WeatherAlert(
                id=f"weather_{alert_id_counter}",
                severity=AlertSeverity.HIGH,
                condition=WeatherCondition.STRONG_WIND,
                title="💨 Strong Wind Alert",
                message=f"Wind speed {weather.wind_speed} km/h. Risk of crop damage.",
                crop=crop,
                timestamp=weather.timestamp,
                expires_at=weather.timestamp + timedelta(hours=6),
            ))
            alert_id_counter += 1

        # Crop-specific recommendations
        if crop and crop.lower() in CROP_THRESHOLDS:
            for alert in alerts:
                if alert.crop and alert.recommended_action is None:
                    action_key = alert.condition.value.upper()
                    action = CROP_SPECIFIC_ACTIONS.get(crop.lower(), {}).get(action_key)
                    if action:
                        alert.recommended_action = action

        # Store alerts in history
        self.alert_history.extend(alerts)
        
        # Keep history size manageable
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        return alerts

    def get_alerts_summary(self, alerts: List[WeatherAlert]) -> Dict[str, Any]:
        """Get a summary of alerts for display"""
        critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        high = [a for a in alerts if a.severity == AlertSeverity.HIGH]
        
        return {
            "total_alerts": len(alerts),
            "critical_count": len(critical),
            "high_count": len(high),
            "alerts": [alert.to_dict() for alert in alerts],
        }


# ============================================================================
# Singleton instance
# ============================================================================

weather_service = WeatherAlertsService()
