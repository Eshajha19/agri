"""
Farmer Segmentation Engine
==========================
K-Means clustering on Firestore farmer profiles with historical yield pattern analysis.
Clusters auto-refresh when new farmer data is added.
"""

import json
import logging
import os
from datetime import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

_CLUSTERS_PATH = Path("farmer_clusters.json")
_MAX_FARMERS = 5000  # Hard cap to prevent memory exhaustion
_DEFAULT_N_CLUSTERS = 5


# =============================================================================
# FEATURE ENCODING
# =============================================================================

def _encode_crop_type(crop_type: Optional[str]) -> int:
    """Map crop type to agronomic family for clustering."""
    mapping = {
        "rice": 1, "wheat": 2, "cotton": 3, "sugarcane": 4,
        "maize": 5, "soybean": 6, "potato": 7, "onion": 8,
        "tomato": 9, "vegetables": 10, "fruits": 11, "other": 0,
    }
    return mapping.get((crop_type or "").lower(), 0)


def _encode_language(lang: Optional[str]) -> int:
    """Encode language to regional group."""
    north = {"hi", "pa", "ur", "en"}
    west = {"gu", "mr", "en"}
    south = {"ta", "te", "kn", "ml", "en"}
    east = {"bn", "or", "as", "en"}
    lang = (lang or "en").lower()
    if lang in north:
        return 1
    if lang in west:
        return 2
    if lang in south:
        return 3
    if lang in east:
        return 4
    return 0


def _extract_location_features(location: Optional[dict]) -> Tuple[float, float]:
    """Extract lat/lng from Firestore GeoPoint or dict."""
    if location is None:
        return 20.5937, 78.9629  # Center of India
    if hasattr(location, "latitude"):
        return float(location.latitude), float(location.longitude)
    lat = float(location.get("lat", 20.5937)) if isinstance(location, dict) else 20.5937
    lng = float(location.get("lng", 78.9629)) if isinstance(location, dict) else 78.9629
    return lat, lng


def _compute_yield_stats(history: List[dict]) -> Tuple[float, float, float]:
    """Compute mean, std, trend from farm intelligence history."""
    if not history:
        return 0.0, 0.0, 0.0

    yields = []
    for h in history:
        scores = h.get("scores", {})
        if isinstance(scores, dict):
            # Use composite score as yield proxy if actual yield not stored
            pest = scores.get("pest_risk", 0)
            irr = scores.get("irrigation", 0)
            mkt = scores.get("market", 0)
            yields.append(100 - (pest + irr) / 2 + mkt / 2)
        else:
            yields.append(0.0)

    if not yields:
        return 0.0, 0.0, 0.0

    mean_y = sum(yields) / len(yields)
    std_y = np.std(yields) if len(yields) > 1 else 0.0
    # Simple trend: last 3 vs first 3
    if len(yields) >= 6:
        trend = (sum(yields[-3:]) / 3) - (sum(yields[:3]) / 3)
    else:
        trend = 0.0

    return mean_y, std_y, trend


# =============================================================================
# SEGMENTATION ENGINE
# =============================================================================

class FarmerSegmentation:
    """
    K-Means clustering on farmer profiles with yield history integration.
    """

    def __init__(self, n_clusters: int = _DEFAULT_N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_profiles: Dict[int, dict] = {}
        self.farmer_assignments: Dict[str, int] = {}
        self._last_refresh: Optional[str] = None

    # -------------------------------------------------------------------------
    # DATA INGESTION
    # -------------------------------------------------------------------------

    def _fetch_farmer_profiles(self, db) -> List[dict]:
        """Fetch all farmer profiles from Firestore with yield history."""
        farmers = []
        try:
            docs = db.collection("users").limit(_MAX_FARMERS).stream()
            for doc in docs:
                data = doc.to_dict() or {}
                if not data.get("profileCompleted"):
                    continue

                uid = doc.id
                crop_type = data.get("cropType", "other")
                language = data.get("language", "en")
                reputation = float(data.get("reputation", 0))
                location = data.get("location")
                lat, lng = _extract_location_features(location)

                # Fetch yield history from subcollection
                history = []
                try:
                    hist_docs = (
                        db.collection("users")
                        .document(uid)
                        .collection("farm_intelligence_history")
                        .order_by("createdAt", direction="DESCENDING")
                        .limit(20)
                        .stream()
                    )
                    for hdoc in hist_docs:
                        history.append(hdoc.to_dict() or {})
                except Exception:
                    pass

                mean_y, std_y, trend = _compute_yield_stats(history)

                farmers.append({
                    "uid": uid,
                    "crop_type": crop_type,
                    "language": language,
                    "reputation": reputation,
                    "lat": lat,
                    "lng": lng,
                    "mean_yield_proxy": mean_y,
                    "yield_std": std_y,
                    "yield_trend": trend,
                    "history_count": len(history),
                    "display_name": data.get("displayName", "Farmer"),
                    "address": data.get("address", ""),
                })

        except Exception as exc:
            logger.error("Failed fetching farmer profiles: %s", exc)

        return farmers

    # -------------------------------------------------------------------------
    # CLUSTERING
    # -------------------------------------------------------------------------

    def fit(self, db) -> dict:
        """
        Run K-Means clustering on all farmer profiles. Returns cluster summary.
        """
        farmers = self._fetch_farmer_profiles(db)
        if len(farmers) < self.n_clusters:
            logger.warning(
                "Only %d farmers available, reducing clusters to %d",
                len(farmers), max(2, len(farmers)),
            )
            self.n_clusters = max(2, len(farmers))

        if len(farmers) < 2:
            return {"status": "insufficient_data", "farmers_count": len(farmers)}

        # Build feature matrix
        features = []
        for f in farmers:
            features.append([
                _encode_crop_type(f["crop_type"]),
                _encode_language(f["language"]),
                f["reputation"],
                f["lat"],
                f["lng"],
                f["mean_yield_proxy"],
                f["yield_std"],
                f["yield_trend"],
                f["history_count"],
            ])

        X = np.array(features, dtype=np.float64)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(X_scaled)

        # Assign clusters
        for i, f in enumerate(farmers):
            self.farmer_assignments[f["uid"]] = int(labels[i])

        # Build cluster profiles
        self.cluster_profiles = {}
        for cluster_id in range(self.n_clusters):
            cluster_farmers = [f for f in farmers if self.farmer_assignments[f["uid"]] == cluster_id]
            if not cluster_farmers:
                continue

            crops = {}
            for f in cluster_farmers:
                c = f["crop_type"]
                crops[c] = crops.get(c, 0) + 1

            top_crop = max(crops, key=crops.get) if crops else "mixed"
            mean_yield = sum(f["mean_yield_proxy"] for f in cluster_farmers) / len(cluster_farmers)
            mean_reputation = sum(f["reputation"] for f in cluster_farmers) / len(cluster_farmers)

            self.cluster_profiles[cluster_id] = {
                "size": len(cluster_farmers),
                "top_crop": top_crop,
                "crop_distribution": crops,
                "mean_yield_proxy": round(mean_yield, 2),
                "mean_reputation": round(mean_reputation, 2),
                "mean_yield_trend": round(
                    sum(f["yield_trend"] for f in cluster_farmers) / len(cluster_farmers), 2
                ),
                "farmers": [
                    {
                        "uid": f["uid"],
                        "display_name": f["display_name"],
                        "crop_type": f["crop_type"],
                        "mean_yield_proxy": round(f["mean_yield_proxy"], 2),
                        "yield_trend": round(f["yield_trend"], 2),
                    }
                    for f in cluster_farmers
                ],
            }

        self._last_refresh = _dt.utcnow().isoformat()
        self._persist()

        return {
            "status": "success",
            "n_clusters": self.n_clusters,
            "farmers_count": len(farmers),
            "clusters": self.cluster_profiles,
            "refreshed_at": self._last_refresh,
        }

    # -------------------------------------------------------------------------
    # PREDICTION / ASSIGNMENT
    # -------------------------------------------------------------------------

    def predict_cluster(self, farmer_features: dict) -> Optional[int]:
        """Predict cluster for a single farmer (used for new farmers)."""
        if self.kmeans is None or self.scaler is None:
            return None

        features = np.array([[
            _encode_crop_type(farmer_features.get("cropType")),
            _encode_language(farmer_features.get("language")),
            float(farmer_features.get("reputation", 0)),
            farmer_features.get("lat", 20.5937),
            farmer_features.get("lng", 78.9629),
            farmer_features.get("mean_yield_proxy", 0),
            farmer_features.get("yield_std", 0),
            farmer_features.get("yield_trend", 0),
            farmer_features.get("history_count", 0),
        ]], dtype=np.float64)

        features_scaled = self.scaler.transform(features)
        return int(self.kmeans.predict(features_scaled)[0])

    def get_farmer_cluster(self, uid: str) -> Optional[int]:
        return self.farmer_assignments.get(uid)

    def get_cluster_profile(self, cluster_id: int) -> Optional[dict]:
        return self.cluster_profiles.get(cluster_id)

    def get_peer_benchmark(self, uid: str) -> Optional[dict]:
        """Return cluster peers and benchmark stats for a farmer."""
        cluster_id = self.get_farmer_cluster(uid)
        if cluster_id is None:
            return None

        profile = self.cluster_profiles.get(cluster_id)
        if not profile:
            return None

        peers = [f for f in profile.get("farmers", []) if f["uid"] != uid]
        farmer_entry = next(
            (f for f in profile.get("farmers", []) if f["uid"] == uid), None
        )

        return {
            "cluster_id": cluster_id,
            "cluster_size": profile["size"],
            "top_crop": profile["top_crop"],
            "cluster_mean_yield": profile["mean_yield_proxy"],
            "cluster_mean_reputation": profile["mean_reputation"],
            "cluster_mean_trend": profile["mean_yield_trend"],
            "peers": peers[:10],  # Cap peers returned
            "my_rank": self._compute_rank(farmer_entry, profile.get("farmers", [])) if farmer_entry else None,
        }

    @staticmethod
    def _compute_rank(farmer: dict, all_farmers: List[dict]) -> dict:
        """Compute percentile rank within cluster."""
        if not farmer or not all_farmers:
            return {}

        sorted_by_yield = sorted(all_farmers, key=lambda f: f["mean_yield_proxy"], reverse=True)
        rank = next((i for i, f in enumerate(sorted_by_yield) if f["uid"] == farmer["uid"]), len(sorted_by_yield))
        percentile = (1 - (rank / len(sorted_by_yield))) * 100 if sorted_by_yield else 0

        return {
            "yield_percentile": round(percentile, 1),
            "rank": rank + 1,
            "total": len(sorted_by_yield),
        }

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    def _persist(self):
        """Save cluster state to disk for warm restarts."""
        try:
            record = {
                "n_clusters": self.n_clusters,
                "assignments": self.farmer_assignments,
                "cluster_profiles": {
                    k: {
                        **v,
                        "farmers": [
                            {fk: fv for fk, fv in f.items() if fk != "display_name"}
                            for f in v.get("farmers", [])
                        ],
                    }
                    for k, v in self.cluster_profiles.items()
                },
                "last_refresh": self._last_refresh,
            }
            tmp = _CLUSTERS_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            os.replace(tmp, _CLUSTERS_PATH)
        except Exception as exc:
            logger.warning("Failed persisting clusters: %s", exc)

    def load(self):
        """Load cluster state from disk."""
        if not _CLUSTERS_PATH.exists():
            return False
        try:
            data = json.loads(_CLUSTERS_PATH.read_text(encoding="utf-8"))
            self.n_clusters = data.get("n_clusters", _DEFAULT_N_CLUSTERS)
            self.farmer_assignments = data.get("assignments", {})
            self.cluster_profiles = data.get("cluster_profiles", {})
            self._last_refresh = data.get("last_refresh")
            return True
        except Exception as exc:
            logger.warning("Failed loading clusters: %s", exc)
            return False

    # -------------------------------------------------------------------------
    # GAP ANALYSIS
    # -------------------------------------------------------------------------

    def gap_analysis(self, uid: str, db) -> Optional[dict]:
        """
        Compare farmer's recent performance against cluster high-performers.
        """
        benchmark = self.get_peer_benchmark(uid)
        if not benchmark:
            return None

        cluster_id = benchmark["cluster_id"]
        profile = self.cluster_profiles.get(cluster_id)
        if not profile:
            return None

        # Find high-performers (top 20% of cluster)
        farmers = profile.get("farmers", [])
        sorted_farmers = sorted(farmers, key=lambda f: f["mean_yield_proxy"], reverse=True)
        top_20_count = max(1, len(sorted_farmers) // 5)
        top_performers = sorted_farmers[:top_20_count]

        top_mean = sum(f["mean_yield_proxy"] for f in top_performers) / len(top_performers)
        farmer_entry = next((f for f in farmers if f["uid"] == uid), None)
        farmer_yield = farmer_entry["mean_yield_proxy"] if farmer_entry else 0

        gap = top_mean - farmer_yield
        significant = gap > (top_mean * 0.15)  # 15% gap threshold

        actions = []
        if significant:
            if gap > top_mean * 0.3:
                actions.append({
                    "priority": "high",
                    "action": "Your yield is significantly below cluster peers. Review irrigation timing and fertilizer schedule.",
                    "impact": "Potential 20-30% yield improvement",
                })
            else:
                actions.append({
                    "priority": "medium",
                    "action": "Your yield is below cluster average. Consider adjusting crop variety or pest management.",
                    "impact": "Potential 10-15% yield improvement",
                })

        if farmer_entry and farmer_entry.get("yield_trend", 0) < -5:
            actions.append({
                "priority": "high",
                "action": "Yield trend is declining. Immediate soil testing and nutrient gap analysis recommended.",
                "impact": "Prevent further decline",
            })

        return {
            "cluster_id": cluster_id,
            "farmer_yield": round(farmer_yield, 2),
            "cluster_top_20_mean": round(top_mean, 2),
            "gap": round(gap, 2),
            "significant": significant,
            "actions": actions,
            "peer_count": len(top_performers),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_segmentation: Optional[FarmerSegmentation] = None


def get_segmentation() -> FarmerSegmentation:
    global _segmentation
    if _segmentation is None:
        _segmentation = FarmerSegmentation()
        _segmentation.load()
    return _segmentation