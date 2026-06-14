"""
Personalized Advisory Router
============================
Farmer segmentation, peer benchmarking, and dynamic recommendation generation.

Endpoints
---------
POST /api/advisory/personalized          — generate personalized recommendations
GET  /api/advisory/cluster-profile/{uid} — farmer's cluster + peer stats
GET  /api/advisory/segments              — all cluster summaries
POST /api/advisory/segments/refresh      — trigger cluster recomputation
"""

import json
import logging
import os
from datetime import datetime as _dt
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/advisory", tags=["personalized advisory"])

_DB = None


def init_personalized_advisory(db_client):
    global _DB
    _DB = db_client


def _get_db():
    if _DB is not None:
        return _DB
    # Fallback: try to get from main module state
    try:
        import firebase_admin
        from firebase_admin import firestore
        if firebase_admin._apps:
            return firestore.client()
    except Exception:
        pass
    return None


@router.post("/personalized")
async def generate_personalized_recommendations(request: Request):
    """
    Generate dynamic, context-aware recommendations for the authenticated farmer.
    Uses cluster profile, peer benchmarks, and gap analysis.
    """
    try:
        from main import verify_role, _get_firestore_user_profile

        token_data = await verify_role(
            request,
            required_roles=["farmer"],          
            required_tenant_id=tenant_id        # comes from path param or request context
        )
        uid = token_data.get("uid")


        profile = _get_firestore_user_profile(uid)
        if not profile:
            raise HTTPException(status_code=404, detail="Farmer profile not found")

        from ml.farmer_segmentation import get_segmentation

        segmentation = get_segmentation()

        # Ensure clusters are fresh (lazy refresh if stale)
        if not segmentation._last_refresh:
            db = _get_db()
            if db:
                segmentation.fit(db)

        cluster_id = segmentation.get_farmer_cluster(uid)
        if cluster_id is None:
            # New farmer — assign to cluster
            db = _get_db()
            if db:
                lat, lng = 20.5937, 78.9629
                loc = profile.get("location")
                if loc and isinstance(loc, dict):
                    lat = float(loc.get("lat", 20.5937))
                    lng = float(loc.get("lng", 78.9629))

                cluster_id = segmentation.predict_cluster({
                    "cropType": profile.get("cropType", "other"),
                    "language": profile.get("language", "en"),
                    "reputation": profile.get("reputation", 0),
                    "lat": lat,
                    "lng": lng,
                    "mean_yield_proxy": 0,
                    "yield_std": 0,
                    "yield_trend": 0,
                    "history_count": 0,
                })

        benchmark = segmentation.get_peer_benchmark(uid) if cluster_id is not None else None
        gap = segmentation.gap_analysis(uid, _get_db()) if cluster_id is not None else None

        # Generate recommendations
        recommendations = []

        # Cluster-based crop recommendation
        if benchmark:
            top_crop = benchmark.get("top_crop")
            farmer_crop = profile.get("cropType", "other")
            if top_crop and top_crop.lower() != farmer_crop.lower():
                recommendations.append({
                    "category": "crop",
                    "priority": "medium",
                    "title": f"Consider {top_crop}",
                    "text": f"Top performers in your cluster grow {top_crop}. Your current crop ({farmer_crop}) may be less optimal for your region and conditions.",
                    "impact": "Potential yield improvement by aligning with cluster peers",
                    "confidence": "moderate",
                })

        # Gap-based actions
        if gap and gap.get("significant"):
            for action in gap.get("actions", []):
                recommendations.append({
                    "category": "performance",
                    "priority": action.get("priority", "medium"),
                    "title": action.get("action", "Improve yield"),
                    "text": action.get("action", ""),
                    "impact": action.get("impact", ""),
                    "confidence": "high" if action.get("priority") == "high" else "medium",
                })

        # Seasonal recommendation
        from advisory_rules import generate_advisories
        seasonal = generate_advisories(
            weather={},
            soil={},
            crop_type=profile.get("cropType"),
        )
        for adv in seasonal[:2]:
            recommendations.append({
                "category": "seasonal",
                "priority": "low",
                "title": adv.get("type", "General"),
                "text": adv.get("message", ""),
                "impact": "General best practice",
                "confidence": "low",
            })

        # Rank by predicted yield impact using XGBoost (batched)
        batch_inputs = []
        batch_indices = []

        for idx, rec in enumerate(recommendations):
            pred_input = {
                "Crop": profile.get("cropType", "Wheat"),
                "CropCoveredArea": 2.0,
                "CHeight": 100,
                "CNext": "Rice",
                "CLast": "Wheat",
                "CTransp": "Medium",
                "IrriType": "Drip",
                "IrriSource": "Groundwater",
                "IrriCount": 3,
                "WaterCov": 80,
                "Season": "Rabi",
            }
            batch_inputs.append(pred_input)
            batch_indices.append(idx)

        batch_preds = [None] * len(recommendations)
        if batch_inputs:
            try:
                from ml.registry import ModelRegistry
                xgb = ModelRegistry.get("xgboost")
                if xgb and hasattr(xgb, "predict_batch"):
                    raw_preds = xgb.predict_batch(batch_inputs)
                    for bi, pred in zip(batch_indices, raw_preds):
                        if pred is not None:
                            batch_preds[bi] = round(pred, 2)
                elif xgb:
                    # Fallback for adapters without predict_batch
                    for bi, inp in zip(batch_indices, batch_inputs):
                        try:
                            batch_preds[bi] = round(xgb.predict(inp), 2)
                        except Exception:
                            batch_preds[bi] = None
            except Exception as exc:
                logger.warning("Batch yield impact scoring failed: %s", exc)

        for idx, pred in enumerate(batch_preds):
            recommendations[idx]["predicted_yield_impact"] = pred

        # Sort by priority then predicted impact
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: (
            priority_order.get(r.get("priority", "low"), 2),
            -(r.get("predicted_yield_impact") or 0),
        ))

        # Log advisory generation
        log_path = Path("advisory_generation_log.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "uid": uid,
                    "cluster_id": cluster_id,
                    "recommendations_count": len(recommendations),
                    "timestamp": _dt.utcnow().isoformat(),
                }) + "\n")
        except Exception:
            pass

        # Batch efficiency metrics
        batch_metrics = {
            "recommendations_scored": sum(1 for p in batch_preds if p is not None),
            "recommendations_total": len(recommendations),
            "batch_size": len(batch_inputs),
            "batch_efficiency": round(len(batch_inputs) / max(len(recommendations), 1), 4),
        }

        return {
            "success": True,
            "uid": uid,
            "cluster_id": cluster_id,
            "benchmark": benchmark,
            "gap_analysis": gap,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "_batch_metrics": batch_metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Personalized advisory generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cluster-profile/{uid}")
async def get_cluster_profile(uid: str, request: Request):
    """
    Return a farmer's cluster assignment and peer benchmark stats.
    """
    try:
        from main import verify_role

        await verify_role(request)

        from ml.farmer_segmentation import get_segmentation

        segmentation = get_segmentation()
        if not segmentation._last_refresh:
            db = _get_db()
            if db:
                segmentation.fit(db)

        cluster_id = segmentation.get_farmer_cluster(uid)
        if cluster_id is None:
            raise HTTPException(status_code=404, detail="Farmer not found in any cluster")

        benchmark = segmentation.get_peer_benchmark(uid)
        gap = segmentation.gap_analysis(uid, _get_db())

        return {
            "success": True,
            "uid": uid,
            "cluster_id": cluster_id,
            "cluster_profile": segmentation.get_cluster_profile(cluster_id),
            "benchmark": benchmark,
            "gap_analysis": gap,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get cluster profile: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/segments")
async def get_all_segments():
    """
    Return summaries of all clusters and their sizes.
    """
    try:
        from ml.farmer_segmentation import get_segmentation

        segmentation = get_segmentation()
        if not segmentation._last_refresh:
            db = _get_db()
            if db:
                segmentation.fit(db)

        clusters = []
        for cid, profile in segmentation.cluster_profiles.items():
            clusters.append({
                "cluster_id": cid,
                "size": profile.get("size", 0),
                "top_crop": profile.get("top_crop", "mixed"),
                "mean_yield_proxy": profile.get("mean_yield_proxy", 0),
                "mean_reputation": profile.get("mean_reputation", 0),
                "mean_yield_trend": profile.get("mean_yield_trend", 0),
            })

        return {
            "success": True,
            "n_clusters": segmentation.n_clusters,
            "refreshed_at": segmentation._last_refresh,
            "clusters": clusters,
        }

    except Exception as e:
        logger.error("Failed to get segments: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segments/refresh")
async def refresh_segments(request: Request):
    """
    Trigger cluster recomputation. Admin/expert only.
    """
    try:
        from main import verify_role

        await verify_role(request, required_roles=["admin", "expert"])

        from celery_worker import refresh_farmer_segments_task
        task = refresh_farmer_segments_task.delay()

        return {
            "success": True,
            "task_id": task.id,
            "message": "Cluster refresh queued. Check status via Celery.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to queue segment refresh: %s", e)
        raise HTTPException(status_code=500, detail=str(e))