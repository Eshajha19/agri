const WEATHER_CODE_LABELS = {
  0: "Clear sky",
  1: "Mostly clear",
  2: "Partly cloudy",
  3: "Overcast",
  45: "Fog",
  48: "Depositing rime fog",
  51: "Light drizzle",
  53: "Moderate drizzle",
  55: "Dense drizzle",
  56: "Light freezing drizzle",
  57: "Dense freezing drizzle",
  61: "Slight rain",
  63: "Moderate rain",
  65: "Heavy rain",
  66: "Light freezing rain",
  67: "Heavy freezing rain",
  71: "Slight snowfall",
  73: "Moderate snowfall",
  75: "Heavy snowfall",
  77: "Snow grains",
  80: "Rain showers",
  81: "Moderate rain showers",
  82: "Violent rain showers",
  85: "Slight snow showers",
  86: "Heavy snow showers",
  95: "Thunderstorm",
  96: "Thunderstorm with hail",
  99: "Severe thunderstorm with hail",
};

const CROP_ADVICE = {
  paddy: {
    rain: "Pause irrigation and strengthen drainage channels to avoid waterlogging in standing fields.",
    heatwave: "Maintain shallow standing water and avoid transplanting during the hottest afternoon window.",
    frost: "Young paddy is vulnerable to cold shock. Delay transplanting and protect nurseries overnight.",
  },
  wheat: {
    rain: "Inspect for lodging and fungal pressure after persistent rain and improve field drainage.",
    heatwave: "Increase irrigation frequency during grain filling and avoid fertilizer sprays in peak heat.",
    frost: "Frost can damage flowering heads. Use a light irrigation before dawn if water is available.",
  },
  maize: {
    rain: "Watch root zones for saturation and postpone nitrogen application during heavy rain spells.",
    heatwave: "Silking maize is heat-sensitive. Irrigate early morning and mulch to retain soil moisture.",
    frost: "Protect seedlings from frost injury and avoid sowing if cold nights are forecast.",
  },
  cotton: {
    rain: "High humidity increases pest and boll-rot risk. Scout fields once rain eases.",
    heatwave: "Retain soil moisture with drip or mulching and avoid pruning during extreme heat.",
    frost: "Cotton is highly frost-sensitive. Harvest mature bolls early when frost risk rises.",
  },
  potato: {
    rain: "Prevent standing water to reduce tuber rot and late blight pressure.",
    heatwave: "Heat stress reduces tuber bulking. Increase irrigation scheduling and keep ridges mulched.",
    frost: "Cover tender foliage where possible and plan a protective irrigation on frost-prone nights.",
  },
  tomato: {
    rain: "Persistent moisture raises fruit cracking and disease risk. Improve staking and airflow.",
    heatwave: "Provide frequent light irrigation and avoid pesticide sprays during midday heat.",
    frost: "Use row covers or nursery protection because tomato foliage is frost-sensitive.",
  },
  sugarcane: {
    rain: "Check lodged canes and clear excess water from low-lying patches.",
    heatwave: "Increase irrigation intervals for young cane and conserve moisture with trash mulching.",
    frost: "Severe cold slows cane growth. Delay ratoon operations when frost is likely.",
  },
};

const DEFAULT_CROP = "paddy";
const CACHE_KEY = "agriWeatherSnapshot";
const LOCATION_SOURCES = new Set(["gps", "ip", "search", "coords", "manual"]);
export const WEATHER_SNAPSHOT_EVENT = "agri:weather-updated";

function getSeverity(rank) {
  return ["info", "watch", "warning", "critical"][rank] || "info";
}

function buildAlert(type, rank, title, message) {
  return {
    type,
    severity: getSeverity(rank),
    title,
    message,
  };
}

function pickLocationName(location) {
  const parts = [location?.city, location?.admin1, location?.country].filter(Boolean);
  return parts.join(", ");
}

function normaliseSource(source, fallback = "manual") {
  const candidate = String(source || fallback || "manual").toLowerCase();
  return LOCATION_SOURCES.has(candidate) ? candidate : fallback;
}

function toCoordinate(value) {
  const parsed = typeof value === "number" ? value : parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normaliseLocation(result) {
  const city = result.city || result.name || result.locality || result.admin2 || "Your area";
  const admin1 = result.admin1 || result.state || "";
  const country = result.country || "";
  return {
    name: pickLocationName({ city, admin1, country }) || city,
    city,
    admin1,
    country,
    latitude: toCoordinate(result.latitude),
    longitude: toCoordinate(result.longitude),
    source: normaliseSource(result.source, "manual"),
  };
}

function toNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function getDayValue(values, index = 0) {
  return Array.isArray(values) && values.length > index ? toNumber(values[index]) : 0;
}

function getWeatherLabel(code) {
  return WEATHER_CODE_LABELS[code] || "Current conditions";
}

export function getAvailableCrops() {
  return Object.keys(CROP_ADVICE).map((crop) => ({
    value: crop,
    label: crop.charAt(0).toUpperCase() + crop.slice(1),
  }));
}

export function getStoredWeatherSnapshot() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) {
      return null;
    }

    const snapshot = JSON.parse(raw);
    const isObject =
      typeof snapshot === "object" && snapshot !== null && !Array.isArray(snapshot);
    const hasValidAlerts = Array.isArray(snapshot?.alerts);
    const hasValidCurrent =
      typeof snapshot?.current === "object" && snapshot.current !== null;
    const hasValidUnits = typeof snapshot?.units === "object" && snapshot.units !== null;

    if (!isObject || !hasValidAlerts || !hasValidCurrent || !hasValidUnits) {
      return null;
    }

    return snapshot;
  } catch {
    return null;
  }
}

export function storeWeatherSnapshot(snapshot) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(snapshot));
  } catch {
    // Ignore storage failures.
  }
}

export function notifyWeatherSnapshotUpdated(snapshot) {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent(WEATHER_SNAPSHOT_EVENT, {
      detail: snapshot,
    })
  );
}

export async function searchLocationByName(query) {
  const response = await fetch(
    `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(
      query
    )}&count=1&language=en&format=json`
  );
  if (!response.ok) {
    throw new Error("Unable to search for that location right now.");
  }

  const data = await response.json();
  if (!data.results?.length) {
    throw new Error("No matching location found.");
  }

  return normaliseLocation({ ...data.results[0], source: "search" });
}

export async function reverseGeocode(latitude, longitude, source = "manual") {
  try {
    const response = await fetch(
      `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${latitude}&longitude=${longitude}&localityLanguage=en`
    );
    if (response.ok) {
      const data = await response.json();
      return {
        city: data.locality || data.city || "Your area",
        admin1: data.principalSubdivision || "",
        country: data.countryName || "",
        latitude,
        longitude,
        name: data.locality || "Your area",
        source: normaliseSource(source, "manual"),
      };
    }
  } catch {
    // Silently handle fetch failures
  }

  return {
    city: "Your area",
    admin1: "",
    country: "",
    latitude,
    longitude,
    name: "Your area",
    source: normaliseSource(source, "manual"),
  };
}

export async function getCurrentPosition() {
  if (!navigator.geolocation) {
    throw new Error("Geolocation is not supported on this device.");
  }

  const position = await new Promise((resolve, reject) => {
    navigator.geolocation.getCurrentPosition(resolve, reject, {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 300000,
    });
  });

  const { latitude, longitude } = position.coords;
  const location = await reverseGeocode(latitude, longitude, "gps");

  return {
    ...location,
    latitude,
    longitude,
    source: "gps",
  };
}

function deriveAlerts(weather) {
  const current = weather.current || {};
  const daily = weather.daily || {};
  const hourly = weather.hourly || {};
  const alerts = [];

  const todayMax = getDayValue(daily.temperature_2m_max);
  const todayMin = getDayValue(daily.temperature_2m_min);
  const todayRain = getDayValue(daily.precipitation_sum);
  const maxWind = getDayValue(daily.wind_speed_10m_max);
  const nextRainChance = Math.max(...(hourly.precipitation_probability || [0]).slice(0, 12), 0);
  const currentRain = toNumber(current.rain) + toNumber(current.showers) + toNumber(current.precipitation);

  if (todayRain >= 35 || currentRain >= 10 || nextRainChance >= 80) {
    const rank = todayRain >= 60 || nextRainChance >= 90 ? 3 : 2;
    alerts.push(
      buildAlert(
        "rain",
        rank,
        "Heavy rain risk",
        "Plan drainage, postpone spraying, and protect harvested produce from persistent rain."
      )
    );
  }

  if (todayMin <= 2 || toNumber(current.temperature_2m) <= 2) {
    const rank = todayMin <= 0 ? 3 : 2;
    alerts.push(
      buildAlert(
        "frost",
        rank,
        "Frost risk overnight",
        "Tender seedlings may face cold stress. Light irrigation or covers can reduce frost damage."
      )
    );
  }

  if (todayMax >= 40 || toNumber(current.apparent_temperature) >= 39) {
    const rank = todayMax >= 44 ? 3 : 2;
    alerts.push(
      buildAlert(
        "heatwave",
        rank,
        "Heatwave conditions",
        "Prioritize early irrigation, avoid transplanting at midday, and watch livestock water needs."
      )
    );
  }

  if (maxWind >= 40) {
    alerts.push(
      buildAlert(
        "wind",
        maxWind >= 55 ? 2 : 1,
        "Strong wind watch",
        "Delay foliar sprays and secure shade nets, nursery covers, and loose irrigation lines."
      )
    );
  }

  if (!alerts.length) {
    alerts.push(
      buildAlert(
        "stable",
        0,
        "No extreme weather today",
        "Conditions look manageable for routine field work. Keep monitoring changes through the day."
      )
    );
  }

  return alerts.sort(
    (first, second) =>
      ["info", "watch", "warning", "critical"].indexOf(second.severity) -
      ["info", "watch", "warning", "critical"].indexOf(first.severity)
  );
}

export function getCropWarnings(alerts, crop = DEFAULT_CROP) {
  const cropKey = (crop || DEFAULT_CROP).toLowerCase();
  const advice = CROP_ADVICE[cropKey] || CROP_ADVICE[DEFAULT_CROP];

  return alerts
    .filter((alert) => advice[alert.type])
    .map((alert) => ({
      type: alert.type,
      severity: alert.severity,
      title: `${cropKey.charAt(0).toUpperCase() + cropKey.slice(1)} advisory`,
      message: advice[alert.type],
    }));
}

export async function fetchWeatherByCoords(latitude, longitude) {
  const location = await reverseGeocode(latitude, longitude, "coords");
  return fetchWeatherByLocation(location);
}

export async function fetchWeatherByLocation(location) {
  const normalisedLocation = normaliseLocation({
    ...location,
    source: normaliseSource(location?.source, "manual"),
  });

  if (!Number.isFinite(normalisedLocation.latitude) || !Number.isFinite(normalisedLocation.longitude)) {
    throw new Error("A valid location is required to fetch weather data.");
  }

  const response = await fetch(
    `https://api.open-meteo.com/v1/forecast?latitude=${normalisedLocation.latitude}&longitude=${normalisedLocation.longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code,precipitation,rain,showers,is_day&hourly=temperature_2m,precipitation_probability,precipitation,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max&timezone=auto&forecast_days=3`
  );

  if (!response.ok) {
    throw new Error("Unable to fetch weather data right now.");
  }

  const weather = await response.json();
  const alerts = deriveAlerts(weather);

  const snapshot = {
    location: normalisedLocation,
    current: weather.current,
    daily: weather.daily,
    hourly: weather.hourly,
    units: weather.current_units,
    summary: getWeatherLabel(weather.current?.weather_code),
    alerts,
    fetchedAt: new Date().toISOString(),
  };

  storeWeatherSnapshot(snapshot);
  notifyWeatherSnapshotUpdated(snapshot);
  return snapshot;
}

export async function getLocationByIP() {
  try {
    const response = await fetch("https://get.geojs.io/v1/ip/geo.json");
    if (!response.ok) {
      throw new Error("IP fetch failed");
    }
    const data = await response.json();
    const latitude = parseFloat(data.latitude);
    const longitude = parseFloat(data.longitude);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      throw new Error("Invalid IP location response");
    }
    return {
      latitude,
      longitude,
      city: data.city || "Your area",
      admin1: data.region || "",
      country: data.country || "",
      name: pickLocationName({
        city: data.city || "Your area",
        admin1: data.region || "",
        country: data.country || "",
      }) || data.city || "Your area",
      source: "ip",
    };
  } catch {
    throw new Error("Could not determine location by IP.");
  }
}

export async function fetchWeatherByIP() {
  const location = await getLocationByIP();
  return fetchWeatherByLocation(location);
}
