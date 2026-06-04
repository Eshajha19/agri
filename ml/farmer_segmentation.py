"""
Farmer Segmentation Engine
==========================
K-Means clustering on Firestore farmer profiles with historical yield pattern analysis.
Clusters auto-refresh when new farmer data is added.
"""

import json
import logging
import os
import time
from datetime import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Incremental refresh policy
_INCREMENTAL_THRESHOLD_PCT = 0.10  # Recompute if >10% new farmers
_INCREMENTAL_MAX_AGE_SECONDS = 3600  # Or if >1 hour since last refresh
_MINIBATCH_SIZE = 100  # Batch size for partial_fit

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

class _SegmentationState:
    """Serializable centroid + scaler state for warm-start restarts."""

    def __init__(self):
        self.centroids: Optional[np.ndarray] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.n_samples_seen: int = 0


class FarmerSegmentation:
    """
    K-Means clustering on farmer profiles with yield history integration.
    Supports incremental MiniBatchKMeans updates and warm-start fallback.
    """

    def __init__(self, n_clusters: int = _DEFAULT_N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_profiles: Dict[int, dict] = {}
        self.farmer_assignments: Dict[str, int] = {}
        self._last_refresh: Optional[str] = None
        self._state = _SegmentationState()
        self._refresh_duration_ms: Optional[float] = None
        self._incremental: bool = False

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

    def _build_feature_matrix(self, farmers: List[dict]) -> np.ndarray:
        """Convert farmer dicts to feature matrix."""
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
        return np.array(features, dtype=np.float64)

    def _build_cluster_profiles(self, farmers: List[dict], labels: np.ndarray) -> Dict[int, dict]:
        """Build cluster profiles from farmer list and label array."""
        profiles = {}
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

            profiles[cluster_id] = {
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
        return profiles

    def _should_full_recompute(self, farmers: List[dict]) -> bool:
        """Determine if full KMeans recompute is needed vs incremental update."""
        if self.kmeans is None or self.scaler is None:
            return True
        if not self._last_refresh:
            return True

        # Time-based staleness
        try:
            last_dt = _dt.fromisoformat(self._last_refresh)
            age = (_dt.utcnow() - last_dt).total_seconds()
            if age > _INCREMENTAL_MAX_AGE_SECONDS:
                return True
        except Exception:
            return True

        # Delta-based: >threshold % new farmers
        prev_count = self._state.n_samples_seen
        if prev_count == 0:
            return True

        new_ratio = abs(len(farmers) - prev_count) / max(prev_count, 1)
        return new_ratio > _INCREMENTAL_THRESHOLD_PCT

    def fit(self, db, force_full: bool = False) -> dict:
        """
        Run clustering on farmer profiles. Uses incremental update when possible,
        full KMeans recompute only when threshold exceeded or forced.
        """
        start = time.time()
        farmers = self._fetch_farmer_profiles(db)

        if len(farmers) < self.n_clusters:
            logger.warning(
                "Only %d farmers available, reducing clusters to %d",
                len(farmers), max(2, len(farmers)),
            )
            self.n_clusters = max(2, len(farmers))

        if len(farmers) < 2:
            return {"status": "insufficient_data", "farmers_count": len(farmers)}

        X = self._build_feature_matrix(farmers)
        needs_full = force_full or self._should_full_recompute(farmers)

        if needs_full:
            logger.info("Running full KMeans recompute on %d farmers", len(farmers))
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            init_centroids = None
            if self._state.centroids is not None and self._state.centroids.shape[0] == self.n_clusters:
                init_centroids = self._state.centroids

            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                init=init_centroids if init_centroids is not None else "k-means++",
            )
            labels = self.kmeans.fit_predict(X_scaled)
            self._incremental = False
        else:
            logger.info("Running incremental MiniBatchKMeans update on %d farmers", len(farmers))
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)

            # Use MiniBatchKMeans with previous centroids for warm partial_fit
            mbk = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                init=self._state.centroids if self._state.centroids is not None else "k-means++",
                batch_size=min(_MINIBATCH_SIZE, len(farmers)),
                n_init=1,
            )
            mbk.fit(X_scaled)
            labels = mbk.predict(X_scaled)
            self.kmeans = mbk
            self._incremental = True

        # Update state for next warm-start
        self._state.centroids = self.kmeans.cluster_centers_.copy()
        self._state.scaler_mean = self.scaler.mean_.copy() if hasattr(self.scaler, "mean_") else None
        self._state.scaler_scale = self.scaler.scale_.copy() if hasattr(self.scaler, "scale_") else None
        self._state.n_samples_seen = len(farmers)

        # Assign clusters
        for i, f in enumerate(farmers):
            self.farmer_assignments[f["uid"]] = int(labels[i])

        self.cluster_profiles = self._build_cluster_profiles(farmers, labels)

        self._last_refresh = _dt.utcnow().isoformat()
        self._refresh_duration_ms = round((time.time() - start) * 1000, 2)
        self._persist()

        return {
            "status": "success",
            "n_clusters": self.n_clusters,
            "farmers_count": len(farmers),
            "clusters": self.cluster_profiles,
            "refreshed_at": self._last_refresh,
            "incremental": self._incremental,
            "duration_ms": self._refresh_duration_ms,
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
        """Save cluster state + centroid metadata to disk for warm restarts."""
        try:
            # Serialize centroids as lists for JSON
            centroids = None
            if self._state.centroids is not None:
                centroids = self._state.centroids.tolist()

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
                "incremental": self._incremental,
                "duration_ms": self._refresh_duration_ms,
                "state": {
                    "centroids": centroids,
                    "scaler_mean": self._state.scaler_mean.tolist() if self._state.scaler_mean is not None else None,
                    "scaler_scale": self._state.scaler_scale.tolist() if self._state.scaler_scale is not None else None,
                    "n_samples_seen": self._state.n_samples_seen,
                },
            }
            tmp = _CLUSTERS_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            os.replace(tmp, _CLUSTERS_PATH)
        except Exception as exc:
            logger.warning("Failed persisting clusters: %s", exc)

    def load(self):
        """Load cluster state + centroid metadata from disk."""
        if not _CLUSTERS_PATH.exists():
            return False
        try:
            data = json.loads(_CLUSTERS_PATH.read_text(encoding="utf-8"))
            self.n_clusters = data.get("n_clusters", _DEFAULT_N_CLUSTERS)
            self.farmer_assignments = data.get("assignments", {})
            self.cluster_profiles = data.get("cluster_profiles", {})
            self._last_refresh = data.get("last_refresh")
            self._incremental = data.get("incremental", False)
            self._refresh_duration_ms = data.get("duration_ms")

            state = data.get("state", {})
            if state.get("centroids"):
                self._state.centroids = np.array(state["centroids"], dtype=np.float64)
            if state.get("scaler_mean"):
                self._state.scaler_mean = np.array(state["scaler_mean"], dtype=np.float64)
            if state.get("scaler_scale"):
                self._state.scaler_scale = np.array(state["scaler_scale"], dtype=np.float64)
            self._state.n_samples_seen = state.get("n_samples_seen", 0)

            # Reconstruct scaler if state available
            if self._state.scaler_mean is not None and self._state.scaler_scale is not None:
                self.scaler = StandardScaler()
                self.scaler.mean_ = self._state.scaler_mean
                self.scaler.scale_ = self._state.scaler_scale
                # Dummy fit to set n_features_in_ and feature_names_in_
                self.scaler.n_features_in_ = len(self._state.scaler_mean)

            # Reconstruct kmeans placeholder for warm-start
            if self._state.centroids is not None:
                self.kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=1,
                )
                self.kmeans.cluster_centers_ = self._state.centroids

            return True
        except Exception as exc:
            logger.warning("Failed loading clusters: %s", exc)
            return False

    # -------------------------------------------------------------------------
    # GAP ANALYSIS
    # -------------------------------------------------------------------------

    def health(self) -> dict:
        """Return segmentation health and update metrics for monitoring."""
        return {
            "ready": self.kmeans is not None,
            "n_clusters": self.n_clusters,
            "farmers_assigned": len(self.farmer_assignments),
            "last_refresh": self._last_refresh,
            "incremental": self._incremental,
            "refresh_duration_ms": self._refresh_duration_ms,
            "centroids_persisted": self._state.centroids is not None,
            "scaler_persisted": self._state.scaler_mean is not None,
            "timestamp": _dt.utcnow().isoformat(),
        }

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