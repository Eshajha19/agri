import axios from "axios";
import { reportErrorToBackend } from "../utils/errorReporting";

const GNEWS_API_KEY = import.meta.env.VITE_GNEWS_API_KEY || "";
const NEWS_API_KEY = import.meta.env.VITE_NEWSAPI_KEY || "";

const GNEWS_BASE = "https://gnews.io/api/v4/search";
const NEWSAPI_BASE = "https://newsapi.org/v2/everything";

const FALLBACK_ARTICLES = [
  {
    id: "fallback-1",
    title: "Government announces new MSP for Kharif crops",
    description: "The government has announced revised Minimum Support Prices for Kharif crops to ensure better returns for farmers.",
    url: "https://pib.gov.in",
    image: "https://via.placeholder.com/400x200?text=Farming+News",
    category: "Government Schemes",
    source: { name: "PIB" },
    publishedAt: new Date().toISOString(),
    author: "Agri News Desk",
  },
  {
    id: "fallback-2",
    title: "Monsoon forecast: Good rainfall expected this season",
    description: "IMD forecasts above-normal rainfall for the upcoming monsoon season, bringing relief to farmers.",
    url: "https://mausam.imd.gov.in",
    image: "https://via.placeholder.com/400x200?text=Weather",
    category: "Weather",
    source: { name: "IMD" },
    publishedAt: new Date().toISOString(),
    author: "Weather Desk",
  },
];

function mapGNewsArticle(article) {
  return {
    id: article.url || article.title,
    title: article.title || "Untitled Article",
    description: article.description || "No description available",
    url: article.url,
    image: article.image || article.imageUrl || "https://via.placeholder.com/400x200?text=Farming+News",
    category: "Agriculture",
    source: { name: article.source?.name || "News" },
    publishedAt: article.publishedAt,
    author: article.author || "Agri News",
  };
}

function mapNewsApiArticle(article) {
  return {
    id: article.url || article.title,
    title: article.title || "Untitled Article",
    description: article.description || "No description available",
    url: article.url,
    image: article.urlToImage || "https://via.placeholder.com/400x200?text=Farming+News",
    category: "Agriculture",
    source: { name: article.source?.name || "News" },
    publishedAt: article.publishedAt,
    author: article.author || "Agri News",
  };
}

export async function fetchFarmingNews(
  page = 1,
  pageSize = 10,
  category = null,
  search = null,
) {
  try {
    const farmingKeywords = [
      "agriculture", "farming", "crop", "farmer", "farm",
      "kisan", "tractor", "fertilizer", "irrigation", "MSP",
      "monsoon", "weather farming", "agritech", "agri news",
    ];

    const queryParts = search?.trim()
      ? `${search} ${category || ""}`
      : `${category || "agriculture"} ${farmingKeywords.join(" OR ")}`.trim();

    const params = new URLSearchParams({
      q: queryParts,
      lang: "en",
      max: String(pageSize),
      page: String(page),
    });

    let response;
    let articles = [];

    if (GNEWS_API_KEY) {
      const url = `${GNEWS_BASE}?${params.toString()}&token=${GNEWS_API_KEY}`;
      console.log("Fetching news from GNews:", url);
      response = await axios.get(url, { timeout: 15000 });
      articles = (response.data?.articles || []).map(mapGNewsArticle);
    } else if (NEWS_API_KEY) {
      const newsParams = new URLSearchParams({
        q: queryParts,
        language: "en",
        pageSize: String(pageSize),
        page: String(page),
        apiKey: NEWS_API_KEY,
      });
      const url = `${NEWSAPI_BASE}?${newsParams.toString()}`;
      console.log("Fetching news from NewsAPI:", url);
      response = await axios.get(url, { timeout: 15000 });
      articles = (response.data?.articles || []).map(mapNewsApiArticle);
    } else {
      console.warn("No news API key configured, using fallback articles");
      articles = [...FALLBACK_ARTICLES];
    }

    return {
      articles: Array.isArray(articles) ? articles : [],
      total: response?.data?.totalArticles || response?.data?.totalResults || articles.length || 0,
      has_more: articles.length >= pageSize,
    };
  } catch (error) {
    console.error("Error fetching farming news:", error);

    reportErrorToBackend({
      message: error.message || "Failed to fetch farming news",
      source: "newsApi.js",
      stack: error.stack,
      level: "error",
    });

    return {
      articles: [...FALLBACK_ARTICLES],
      total: FALLBACK_ARTICLES.length,
      has_more: false,
    };
  }
}

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

export function formatNewsDate(dateStr) {
  try {
    const date = new Date(dateStr);

    if (isNaN(date.getTime())) {
      return "Unknown date";
    }

    const now = new Date();
    const diffTime = now - date;

    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

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