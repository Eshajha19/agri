/**
 * Real-time Indian Agricultural Market Prices API
 * Fetches data from data.gov.in (Agmarknet)
 */

const API_KEY = "579b464db66ec23bdd000001cdd39465d03e4d774860900bb102e473"; // Generic public key for demo
const RESOURCE_ID = "9ef27db9-c68f-406d-99fd-3c373b06019a";
const BASE_URL = "https://api.data.gov.in/resource";

const MOCK_FALLBACK = [
  { id: 1, commodity: "Paddy (Dhan)", state: "Punjab", district: "Amritsar", mandi: "Amritsar", minPrice: 2100, maxPrice: 2350, modalPrice: 2200, arrivalDate: "2026-04-20" },
  { id: 2, commodity: "Wheat", state: "Haryana", district: "Karnal", mandi: "Karnal", minPrice: 2000, maxPrice: 2275, modalPrice: 2150, arrivalDate: "2026-04-21" },
  { id: 3, commodity: "Cotton", state: "Gujarat", district: "Rajkot", mandi: "Rajkot", minPrice: 7500, maxPrice: 8800, modalPrice: 8200, arrivalDate: "2026-04-22" },
  { id: 4, commodity: "Onion", state: "Maharashtra", district: "Nashik", mandi: "Lasalgaon", minPrice: 1200, maxPrice: 1800, modalPrice: 1550, arrivalDate: "2026-04-22" },
  { id: 5, commodity: "Tomato", state: "Karnataka", district: "Kolar", mandi: "Kolar", minPrice: 800, maxPrice: 1400, modalPrice: 1100, arrivalDate: "2026-04-21" },
  { id: 6, commodity: "Paddy (Dhan)", state: "West Bengal", district: "Bardhaman", mandi: "Bardhaman", minPrice: 1950, maxPrice: 2150, modalPrice: 2050, arrivalDate: "2026-04-22" },
  { id: 7, commodity: "Wheat", state: "Uttar Pradesh", district: "Gorakhpur", mandi: "Gorakhpur", minPrice: 1900, maxPrice: 2100, modalPrice: 2025, arrivalDate: "2026-04-20" },
  { id: 8, commodity: "Potato", state: "Uttar Pradesh", district: "Agra", mandi: "Agra", minPrice: 900, maxPrice: 1300, modalPrice: 1150, arrivalDate: "2026-04-22" },
  { id: 9, commodity: "Cotton", state: "Telangana", district: "Warangal", mandi: "Warangal", minPrice: 7200, maxPrice: 8400, modalPrice: 7900, arrivalDate: "2026-04-21" },
  { id: 10, commodity: "Soybean", state: "Madhya Pradesh", district: "Indore", mandi: "Indore", minPrice: 4200, maxPrice: 4850, modalPrice: 4600, arrivalDate: "2026-04-22" },
  { id: 11, commodity: "Maize", state: "Bihar", district: "Gulabbagh", mandi: "Gulabbagh", minPrice: 1800, maxPrice: 2100, modalPrice: 1950, arrivalDate: "2026-04-22" },
  { id: 12, commodity: "Mustard", state: "Rajasthan", district: "Alwar", mandi: "Alwar", minPrice: 5200, maxPrice: 5800, modalPrice: 5550, arrivalDate: "2026-04-21" },
];

;;;;;;
export const fetchMarketPrices = async (filters = {}) => {
  try {
    const url = `${BASE_URL}/${RESOURCE_ID}?api-key=${API_KEY}&format=json&limit=100`;

    const response = await fetch(url);

    // Check HTTP status
    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    // Parse JSON safely
    const data = await response.json();

    // Validate structure
    if (!data || !Array.isArray(data.records)) {
      throw new Error("Invalid API response structure");
    }

    // Map records
    let records = data.records.map((r, index) => ({
      id: `${r.state}-${r.market}-${r.arrival_date}-${index}`, //stable id
      commodity: r.commodity,
      state: r.state,
      district: r.district,
      mandi: r.market,
      minPrice: parseInt(r.min_price) || 0,
      maxPrice: parseInt(r.max_price) || 0,
      modalPrice: parseInt(r.modal_price) || 0,
      arrivalDate: r.arrival_date
    }));

    // Apply filters (same as before)
    if (filters.state && filters.state !== "All") {
      records = records.filter(item => item.state === filters.state);
    }

    if (filters.commodity && filters.commodity !== "All") {
      records = records.filter(item => item.commodity === filters.commodity);
    }

    if (filters.search) {
      const s = filters.search.toLowerCase();
      records = records.filter(item =>
        item.commodity.toLowerCase().includes(s) ||
        item.mandi.toLowerCase().includes(s) ||
        item.district.toLowerCase().includes(s)
      );
    }

    //Handle empty API result properly
    return records; // even if empty → valid case

  } catch (error) {
    console.error("API failed, using fallback data:", error);
  }

  // Fallback (unchanged logic)
  let filtered = [...MOCK_FALLBACK];

  if (filters.state && filters.state !== "All") {
    filtered = filtered.filter(item => item.state === filters.state);
  }

  if (filters.commodity && filters.commodity !== "All") {
    filtered = filtered.filter(item => item.commodity === filters.commodity);
  }

  if (filters.search) {
    const s = filters.search.toLowerCase();
    filtered = filtered.filter(item =>
      item.commodity.toLowerCase().includes(s) ||
      item.mandi.toLowerCase().includes(s) ||
      item.district.toLowerCase().includes(s)
    );
  }

  return filtered;
};

export const fetchPriceTrends = async (commodity) => {
  // For demo, return same trend but with slight variations based on commodity name
  const base = commodity === "Cotton" ? 8000 : 2000;
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  return days.map(day => ({
    name: day,
    price: base + Math.floor(Math.random() * 200) - 100
  }));
};

export const getUniqueStates = (records = MOCK_FALLBACK) => {
  return ["All", ...new Set(records.map(item => item.state))].sort();
};

export const getUniqueCommodities = (records = MOCK_FALLBACK) => {
  return ["All", ...new Set(records.map(item => item.commodity))].sort();
};
