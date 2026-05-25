import axios from "axios";
import { reportErrorToBackend } from "../utils/errorReporting";

const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

/**
 * Fetch farming news articles
 */
export async function fetchFarmingNews(
  page = 1,
  pageSize = 10,
  category = null,
  search = null
) {
  try {
    const params = new URLSearchParams();

    params.append("page", page);
    params.append("page_size", pageSize);

    if (category && category !== "All") {
      params.append("category", category);
    }

    if (search?.trim()) {
      params.append("search", search.trim());
    }

    const url = `${API_BASE}/api/farming-news?${params.toString()}`;

    console.log("Fetching news:", url);

    const response = await axios.get(url, {
      timeout: 15000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    // Handle multiple backend response shapes safely
    const articles =
      response.data?.articles ||
      response.data?.data ||
      response.data ||
      [];

    return {
      articles: Array.isArray(articles) ? articles : [],
      total: response.data?.total || articles.length || 0,
    };
  } catch (error) {
    console.error("Error fetching farming news:", error);

    reportErrorToBackend({
      message: error.message || "Failed to fetch farming news",
      source: "newsApi.js",
      stack: error.stack,
      level: "error",
    });

    throw new Error(
      error.response?.data?.message ||
        error.message ||
        "Unable to load farming news."
    );
  }
}

/**
 * Categories
 */
export function getNewsCategories() {
  return [
    "All",
    "Weather",
    "Government Schemes",
    "Crop Management",
    "Technology",
    "Insurance",
    "Organic Farming",
    "Market Prices",
    "Soil Management",
  ];
}

/**
 * Date formatter
 */
export function formatNewsDate(dateStr) {
  try {
    const date = new Date(dateStr);

    if (isNaN(date.getTime())) {
      return "Unknown date";
    }

    const now = new Date();
    const diffTime = now - date;

    const diffDays = Math.floor(
      diffTime / (1000 * 60 * 60 * 24)
    );

    if (diffDays === 0) return "Today";

    if (diffDays === 1) return "Yesterday";

    if (diffDays < 7) {
      return `${diffDays} days ago`;
    }

    if (diffDays < 30) {
      const weeks = Math.floor(diffDays / 7);

      return `${weeks} week${weeks > 1 ? "s" : ""} ago`;
    }

    return date.toLocaleDateString("en-IN", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return "Unknown date";
  }
}