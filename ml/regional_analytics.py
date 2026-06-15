"""
Regional Analytics Engine
=========================
Statistical analysis, bootstrapping, cohort detection, and significance testing
for federated regional yield benchmarking.

Privacy-preserving by design — only anonymized aggregates are computed and returned.
"""

import json
import logging
import math
import os
from datetime import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

_BOOTSTRAP_SAMPLES = 1000
_SIGNIFICANCE_LEVEL = 0.05
_MIN_COHORT_SIZE = 5
_REPORTS_DIR = Path("regional_reports")
_REPORTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# DATA INGESTION (privacy-preserving)
# =============================================================================

def _fetch_regional_data(db, region_filter: Optional[str] = None) -> List[dict]:
    """
    Fetch anonymized yield records from Firestore farm_intelligence_history.
    Returns list of dicts with: yield_proxy, crop_type, season, region, soil_type.
    No UIDs or identifying fields are included.
    """
    records = []
    try:
        users = db.collection("users").stream()
        for user_doc in users:
            data = user_doc.to_dict() or {}
            if not data.get("profileCompleted"):
                continue

            # Extract region from address or location
            address = data.get("address", "")
            region = _extract_region_from_address(address)
            if region_filter and region != region_filter:
                continue

            crop_type = data.get("cropType", "other")
            language = data.get("language", "en")

            # Fetch yield history
            hist_docs = (
                db.collection("users")
                .document(user_doc.id)
                .collection("farm_intelligence_history")
                .order_by("createdAt", direction="DESCENDING")
                .limit(10)
                .stream()
            )

            for hdoc in hist_docs:
                hdata = hdoc.to_dict() or {}
                scores = hdata.get("scores", {})
                if not isinstance(scores, dict):
                    continue

                # Use composite score as yield proxy
                pest = scores.get("pest_risk", 0)
                irr = scores.get("irrigation", 0)
                mkt = scores.get("market", 0)
                yield_proxy = 100 - (pest + irr) / 2 + mkt / 2

                records.append({
                    "yield_proxy": yield_proxy,
                    "crop_type": crop_type,
                    "season": hdata.get("season", "Unknown"),
                    "region": region,
                    "language": language,
                    "created_at": hdata.get("createdAt", ""),
                })

    except Exception as exc:
        logger.error("Failed fetching regional data: %s", exc)

    return records


def _extract_region_from_address(address: str) -> str:
    """Extract Indian state from address string."""
    if not address:
        return "Unknown"
    address_lower = address.lower()
    state_patterns = {
        "maharashtra": "Maharashtra",
        "punjab": "Punjab",
        "haryana": "Haryana",
        "karnataka": "Karnataka",
        "tamil nadu": "Tamil Nadu",
        "telangana": "Telangana",
        "gujarat": "Gujarat",
        "rajasthan": "Rajasthan",
        "uttar pradesh": "Uttar Pradesh",
        "bihar": "Bihar",
        "west bengal": "West Bengal",
        "madhya pradesh": "Madhya Pradesh",
        "andhra pradesh": "Andhra Pradesh",
        "kerala": "Kerala",
        "odisha": "Odisha",
        "assam": "Assam",
        "jharkhand": "Jharkhand",
        "chhattisgarh": "Chhattisgarh",
        "delhi": "Delhi",
    }
    for pattern, state in state_patterns.items():
        if pattern in address_lower:
            return state
    return "Other"


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

class RegionalAnalytics:
    """
    Statistical engine for regional yield benchmarking.
    """

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._cache_timestamp: Optional[str] = None

    # -------------------------------------------------------------------------
    # AGGREGATION
    # -------------------------------------------------------------------------

    def aggregate_by_region_crop(self, db, region: Optional[str] = None) -> Dict[str, dict]:
        """Compute regional yield baselines with bootstrapped confidence bounds."""
        records = _fetch_regional_data(db, region)
        if not records:
            return {"error": "No data available for aggregation"}

        # Group by region + crop
        cohorts: Dict[str, List[float]] = {}
        for r in records:
            key = f"{r['region']}::{r['crop_type']}"
            cohorts.setdefault(key, []).append(r["yield_proxy"])

        results = {}
        for key, yields in cohorts.items():
            if len(yields) < _MIN_COHORT_SIZE:
                continue

            yields_arr = np.array(yields)

            # Bootstrapped confidence intervals
            boot_means = []
            for _ in range(_BOOTSTRAP_SAMPLES):
                sample = np.random.choice(yields_arr, size=len(yields_arr), replace=True)
                boot_means.append(np.mean(sample))

            boot_means = sorted(boot_means)
            mean_yield = float(np.mean(yields_arr))
            std_yield = float(np.std(yields_arr))
            p5 = float(np.percentile(boot_means, 5))
            p95 = float(np.percentile(boot_means, 95))

            results[key] = {
                "region": key.split("::")[0],
                "crop_type": key.split("::")[1],
                "sample_size": len(yields),
                "mean_yield": round(mean_yield, 2),
                "std_yield": round(std_yield, 2),
                "median_yield": round(float(np.median(yields_arr)), 2),
                "confidence_interval": {
                    "lower": round(p5, 2),
                    "upper": round(p95, 2),
                },
                "confidence_level": 0.90,
                "min_yield": round(float(np.min(yields_arr)), 2),
                "max_yield": round(float(np.max(yields_arr)), 2),
                "coefficient_of_variation": round(std_yield / mean_yield, 4) if mean_yield > 0 else 0,
            }

        return {
            "cohorts": results,
            "total_records": len(records),
            "generated_at": _dt.utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # PERCENTILE RANK
    # -------------------------------------------------------------------------

    def compute_percentile(self, db, farmer_yield: float, region: str, crop_type: str) -> Optional[dict]:
        """Compute farmer's percentile rank within regional cohort."""
        records = _fetch_regional_data(db, region)
        cohort_yields = [
            r["yield_proxy"]
            for r in records
            if r["crop_type"].lower() == crop_type.lower()
        ]

        if len(cohort_yields) < _MIN_COHORT_SIZE:
            return None

        cohort_yields = sorted(cohort_yields)
        rank = sum(1 for y in cohort_yields if y < farmer_yield)
        percentile = (rank / len(cohort_yields)) * 100 if cohort_yields else 0

        # Nearest peers
        diffs = [(abs(y - farmer_yield), y) for y in cohort_yields]
        diffs.sort()
        nearest = [y for _, y in diffs[:5]]

        return {
            "farmer_yield": round(farmer_yield, 2),
            "region": region,
            "crop_type": crop_type,
            "cohort_size": len(cohort_yields),
            "percentile": round(percentile, 1),
            "rank": rank + 1,
            "nearest_peers": [round(y, 2) for y in nearest],
            "cohort_mean": round(np.mean(cohort_yields), 2),
            "cohort_median": round(np.median(cohort_yields), 2),
        }

    # -------------------------------------------------------------------------
    # SIGNIFICANCE TESTS
    # -------------------------------------------------------------------------

    def significance_test(self, db, farmer_yield: float, region: str, crop_type: str) -> Optional[dict]:
        """Run t-test and Mann-Whitney U test against regional baseline."""
        records = _fetch_regional_data(db, region)
        cohort_yields = [
            r["yield_proxy"]
            for r in records
            if r["crop_type"].lower() == crop_type.lower()
        ]

        if len(cohort_yields) < _MIN_COHORT_SIZE:
            return None

        cohort_arr = np.array(cohort_yields)
        farmer_arr = np.array([farmer_yield])

        # One-sample t-test: farmer vs cohort mean
        t_stat, t_pvalue = stats.ttest_1samp(cohort_arr, farmer_yield)

        # Mann-Whitney U: farmer vs cohort
        try:
            u_stat, u_pvalue = stats.mannwhitneyu(farmer_arr, cohort_arr, alternative="two-sided")
        except ValueError:
            u_stat, u_pvalue = 0, 1.0

        significant = t_pvalue < _SIGNIFICANCE_LEVEL or u_pvalue < _SIGNIFICANCE_LEVEL

        interpretation = "not_significant"
        if significant:
            if farmer_yield > np.mean(cohort_arr):
                interpretation = "significantly_above"
            else:
                interpretation = "significantly_below"

        return {
            "farmer_yield": round(farmer_yield, 2),
            "region": region,
            "crop_type": crop_type,
            "cohort_size": len(cohort_yields),
            "cohort_mean": round(float(np.mean(cohort_arr)), 2),
            "t_test": {
                "statistic": round(float(t_stat), 4),
                "p_value": round(float(t_pvalue), 6),
                "significant": t_pvalue < _SIGNIFICANCE_LEVEL,
            },
            "mann_whitney_u": {
                "statistic": round(float(u_stat), 4),
                "p_value": round(float(u_pvalue), 6),
                "significant": u_pvalue < _SIGNIFICANCE_LEVEL,
            },
            "overall_significant": significant,
            "interpretation": interpretation,
            "alpha": _SIGNIFICANCE_LEVEL,
        }

    # -------------------------------------------------------------------------
    # TOP PERFORMERS
    # -------------------------------------------------------------------------

    def top_performers(self, db, region: str, crop_type: str, top_n: int = 10) -> List[dict]:
        """Identify top-performing farm clusters and extract patterns."""
        records = _fetch_regional_data(db, region)
        crop_records = [r for r in records if r["crop_type"].lower() == crop_type.lower()]

        if len(crop_records) < _MIN_COHORT_SIZE:
            return []

        # Sort by yield proxy
        sorted_records = sorted(crop_records, key=lambda r: r["yield_proxy"], reverse=True)
        top_20_pct = max(1, len(sorted_records) // 5)
        top_records = sorted_records[:top_20_pct]

        # Extract common patterns
        seasons = {}
        for r in top_records:
            s = r.get("season", "Unknown")
            seasons[s] = seasons.get(s, 0) + 1

        top_season = max(seasons, key=seasons.get) if seasons else "Unknown"

        return [
            {
                "rank": i + 1,
                "yield_proxy": round(r["yield_proxy"], 2),
                "season": r.get("season", "Unknown"),
                "region": r["region"],
            }
            for i, r in enumerate(top_records[:top_n])
        ]

    # -------------------------------------------------------------------------
    # REPORT GENERATION
    # -------------------------------------------------------------------------

    def generate_report(self, db, farmer_uid: str, farmer_yield: float, region: str, crop_type: str) -> Optional[dict]:
        """Generate all analytics for a farmer and return structured data for PDF."""
        percentile = self.compute_percentile(db, farmer_yield, region, crop_type)
        significance = self.significance_test(db, farmer_yield, region, crop_type)
        aggregates = self.aggregate_by_region_crop(db, region)
        top = self.top_performers(db, region, crop_type, top_n=5)

        report_id = f"RPT-{farmer_uid[:8]}-{_dt.now().strftime('%Y%m%d%H%M%S')}"

        report_data = {
            "report_id": report_id,
            "generated_at": _dt.utcnow().isoformat(),
            "farmer_yield": round(farmer_yield, 2),
            "region": region,
            "crop_type": crop_type,
            "percentile": percentile,
            "significance_test": significance,
            "regional_aggregates": aggregates.get("cohorts", {}).get(f"{region}::{crop_type}", {}),
            "top_performers": top,
        }

        # Persist
        try:
            path = _REPORTS_DIR / f"{report_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
        except Exception as exc:
            logger.warning("Failed persisting report: %s", exc)

        return report_data


# =============================================================================
# SINGLETON
# =============================================================================

_analytics: Optional[RegionalAnalytics] = None


def get_regional_analytics() -> RegionalAnalytics:
    global _analytics
    if _analytics is None:
        _analytics = RegionalAnalytics()
    return _analytics