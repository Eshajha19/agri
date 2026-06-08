"""
Price Forecasting Engine
========================
STL-style decomposition + rolling forecast + volatility scoring.
No external dependencies beyond pandas, numpy, scikit-learn.
"""

import gzip
import json
import logging
import logging.handlers
import math
import os
import shutil
from datetime import datetime as _dt, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Rotation config (env overrideable)
_LOG_ROTATION_MAX_BYTES = int(os.getenv("LOG_ROTATION_MAX_BYTES", "104857600"))  # 100MB
_LOG_ROTATION_BACKUP_COUNT = int(os.getenv("LOG_ROTATION_BACKUP_COUNT", "5"))
_LOG_ARCHIVE_DAYS = int(os.getenv("LOG_ARCHIVE_DAYS", "7"))

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

_HISTORY_PATH = Path("price_history.json")
_FORECASTS_PATH = Path("price_forecasts.jsonl")
_VOLATILITY_PATH = Path("price_volatility.json")
_SEED_DAYS = 90  # Synthetic history length for new crops

# Realistic base prices (₹/quintal) for major Indian crops
_CROP_BASE_PRICES = {
    "Wheat": 2200,
    "Rice": 1800,
    "Cotton": 5500,
    "Sugarcane": 290,
    "Maize": 1850,
    "Soybean": 3900,
    "Potato": 1200,
    "Onion": 1500,
    "Tomato": 1800,
    "Vegetables": 2000,
    "Fruits": 3500,
}

_CROP_SEASONALITY = {
    "Wheat": 0.08,   # 8% seasonal swing
    "Rice": 0.12,
    "Cotton": 0.15,
    "Sugarcane": 0.05,
    "Maize": 0.10,
    "Soybean": 0.11,
    "Potato": 0.20,
    "Onion": 0.25,
    "Tomato": 0.22,
    "Vegetables": 0.18,
    "Fruits": 0.14,
}


# =============================================================================
# SYNTHETIC SEED DATA
# =============================================================================

def _generate_seed_history(crop: str, days: int = _SEED_DAYS) -> pd.DataFrame:
    """Generate realistic synthetic price history for a crop."""
    base = _CROP_BASE_PRICES.get(crop, 2000)
    seasonal_amp = _CROP_SEASONALITY.get(crop, 0.10)
    np.random.seed(42)

    dates = pd.date_range(end=_dt.now().date(), periods=days, freq="D")
    trend = np.linspace(base * 0.95, base * 1.05, days)
    seasonal = base * seasonal_amp * np.sin(2 * np.pi * np.arange(days) / 365.25 * 7)
    noise = np.random.normal(0, base * 0.03, days)

    prices = trend + seasonal + noise
    prices = np.maximum(prices, base * 0.5)  # Floor at 50% of base

    return pd.DataFrame({
        "date": dates,
        "price": prices.round(2),
    })


# =============================================================================
# GZIP ROTATING HANDLER
# =============================================================================

class _GzRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    RotatingFileHandler that compresses backups with gzip on rotation.
    Backups older than LOG_ARCHIVE_DAYS are deleted during rotation.
    """

    def __init__(self, filename, maxBytes=0, backupCount=0, archive_days=7, **kwargs):
        self.archive_days = archive_days
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, **kwargs)

    def namer(self, default_name: str) -> str:
        return default_name + ".gz"

    def rotator(self, source: str, dest: str):
        # Compress rotated file
        with open(source, "rb") as f_in:
            with gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)
        # Cleanup old archives
        self._cleanup_old_archives()

    def _cleanup_old_archives(self):
        base = self.baseFilename
        cutoff = _dt.now() - timedelta(days=self.archive_days)
        dir_path = Path(base).parent
        if not dir_path.exists():
            return
        for f in dir_path.glob("*.jsonl.*.gz"):
            try:
                mtime = _dt.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    logger.info("Deleted old forecast archive: %s", f.name)
            except Exception:
                pass


# =============================================================================
# FORECASTER
# =============================================================================

class PriceForecaster:
    """
    STL-style decomposition + rolling mean forecast + volatility scoring.
    """

    def __init__(self):
        self.history: Dict[str, pd.DataFrame] = {}
        self._load_history()

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    def _load_history(self):
        if not _HISTORY_PATH.exists():
            return
        try:
            with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for crop, records in data.items():
                self.history[crop] = pd.DataFrame(records)
                self.history[crop]["date"] = pd.to_datetime(self.history[crop]["date"])
        except Exception as exc:
            logger.warning("Failed loading price history: %s", exc)

    def _save_history(self):
        try:
            record = {}
            for crop, df in self.history.items():
                record[crop] = df.to_dict("records")
            tmp = _HISTORY_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
            os.replace(tmp, _HISTORY_PATH)
        except Exception as exc:
            logger.warning("Failed saving price history: %s", exc)

    def _log_forecast(self, crop: str, forecast_df: pd.DataFrame, volatility: dict):
        try:
            entry = {
                "crop": crop,
                "forecast": forecast_df.to_dict("records"),
                "volatility": volatility,
                "generated_at": _dt.utcnow().isoformat(),
            }
            # Use rotating handler for bounded disk usage
            handler = _GzRotatingFileHandler(
                _FORECASTS_PATH,
                maxBytes=_LOG_ROTATION_MAX_BYTES,
                backupCount=_LOG_ROTATION_BACKUP_COUNT,
                archive_days=_LOG_ARCHIVE_DAYS,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            record = logging.LogRecord(
                name="price_forecaster",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=json.dumps(entry, default=str),
                args=(),
                exc_info=None,
            )
            handler.emit(record)
            handler.close()
        except Exception as exc:
            logger.warning("Failed logging forecast: %s", exc)

    # -------------------------------------------------------------------------
    # DATA MANAGEMENT
    # -------------------------------------------------------------------------

    def _ensure_history(self, crop: str):
        if crop not in self.history or self.history[crop].empty:
            self.history[crop] = _generate_seed_history(crop)
            self._save_history()

    def add_price(self, crop: str, price: float, date: Optional[str] = None):
        """Add a real observed price to history."""
        self._ensure_history(crop)
        date = date or _dt.now().strftime("%Y-%m-%d")
        new_row = pd.DataFrame({"date": [pd.to_datetime(date)], "price": [float(price)]})
        self.history[crop] = pd.concat([self.history[crop], new_row], ignore_index=True)
        self.history[crop] = self.history[crop].drop_duplicates(subset=["date"], keep="last")
        self.history[crop] = self.history[crop].sort_values("date").reset_index(drop=True)
        self._save_history()

    # -------------------------------------------------------------------------
    # STL DECOMPOSITION (manual — no statsmodels)
    # -------------------------------------------------------------------------

    def stl_decompose(self, crop: str) -> dict:
        """
        Decompose price series into trend, seasonal, residual.
        """
        self._ensure_history(crop)
        df = self.history[crop].copy()

        if len(df) < 14:
            return {"error": "Insufficient data for decomposition (need 14+ days)"}

        # Trend: 7-day rolling mean (centered)
        df["trend"] = df["price"].rolling(window=7, min_periods=1, center=True).mean()

        # Detrended
        df["detrended"] = df["price"] - df["trend"]

        # Seasonal: day-of-year average (repeating annual pattern)
        df["doy"] = df["date"].dt.dayofyear
        seasonal_map = df.groupby("doy")["detrended"].mean().to_dict()
        df["seasonal"] = df["doy"].map(seasonal_map)

        # Residual
        df["residual"] = df["detrended"] - df["seasonal"]

        return {
            "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "price": df["price"].round(2).tolist(),
            "trend": df["trend"].round(2).tolist(),
            "seasonal": df["seasonal"].round(2).tolist(),
            "residual": df["residual"].round(2).tolist(),
        }

    # -------------------------------------------------------------------------
    # FORECAST
    # -------------------------------------------------------------------------

    def forecast(self, crop: str, days: int = 14) -> dict:
        """
        Generate price forecast with confidence intervals.
        """
        self._ensure_history(crop)
        df = self.history[crop].copy()

        if len(df) < 7:
            return {"error": "Insufficient data for forecast"}

        # Trend: extend rolling mean
        last_prices = df["price"].tail(7).values
        trend_slope = (last_prices[-1] - last_prices[0]) / 6 if len(last_prices) >= 6 else 0
        last_trend = df["price"].rolling(window=7, min_periods=1).mean().iloc[-1]

        # Seasonal: day-of-year pattern
        df["doy"] = df["date"].dt.dayofyear
        seasonal_map = df.groupby("doy")["price"].mean().to_dict()

        # Residual std for confidence intervals
        df["trend"] = df["price"].rolling(window=7, min_periods=1, center=True).mean()
        df["detrended"] = df["price"] - df["trend"]
        df["seasonal"] = df["doy"].map(seasonal_map)
        residual_std = df["price"].tail(14).std() if len(df) >= 14 else df["price"].std()

        # Generate future dates
        last_date = df["date"].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]

        forecasts = []
        for i, date in enumerate(future_dates):
            # Trend projection
            projected_trend = last_trend + trend_slope * (i + 1)

            # Seasonal component
            doy = date.timetuple().tm_yday
            seasonal = seasonal_map.get(doy, 0)

            # Point estimate
            point = projected_trend + seasonal

            # Confidence interval (±1.5 std)
            lower = point - 1.5 * residual_std
            upper = point + 1.5 * residual_std

            forecasts.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(point, 2),
                "lower_bound": round(max(lower, point * 0.7), 2),
                "upper_bound": round(upper, 2),
            })

        # Best sell date: highest price in forecast
        best = max(forecasts, key=lambda x: x["price"])

        # Recommendation
        current_price = df["price"].iloc[-1]
        price_change = (best["price"] - current_price) / current_price if current_price > 0 else 0

        if price_change > 0.05:
            recommendation = f"Prices expected to rise {price_change * 100:.1f}%. Hold for better rates."
        elif price_change < -0.05:
            recommendation = f"Prices expected to fall {abs(price_change) * 100:.1f}%. Consider selling soon."
        else:
            recommendation = "Price trend is stable. Sell when convenient."

        volatility = self.volatility_score(crop)

        result = {
            "crop": crop,
            "current_price": round(current_price, 2),
            "forecast": forecasts,
            "best_sell_date": best["date"],
            "best_sell_price": best["price"],
            "recommendation": recommendation,
            "volatility": volatility,
            "confidence_interval_width": round(3 * residual_std, 2),
            "model_type": "STL-Rolling",
            "generated_at": _dt.utcnow().isoformat(),
        }

        self._log_forecast(crop, pd.DataFrame(forecasts), volatility)
        return result

    # -------------------------------------------------------------------------
    # VOLATILITY
    # -------------------------------------------------------------------------

    def volatility_score(self, crop: str) -> dict:
        """
        Compute volatility score based on recent residual variance.
        """
        self._ensure_history(crop)
        df = self.history[crop].copy()

        if len(df) < 14:
            return {"score": 0, "classification": "unknown", "coefficient_of_variation": 0}

        recent = df["price"].tail(14)
        mean_price = recent.mean()
        std_price = recent.std()

        cv = (std_price / mean_price) if mean_price > 0 else 0

        # Score 0-100
        score = min(100, cv * 500)

        if score < 20:
            classification = "low"
        elif score < 50:
            classification = "moderate"
        elif score < 80:
            classification = "high"
        else:
            classification = "extreme"

        # Persist
        try:
            vol_data = json.loads(_VOLATILITY_PATH.read_text()) if _VOLATILITY_PATH.exists() else {}
            vol_data[crop] = {
                "score": round(score, 2),
                "classification": classification,
                "cv": round(cv, 4),
                "updated_at": _dt.utcnow().isoformat(),
            }
            with open(_VOLATILITY_PATH, "w", encoding="utf-8") as f:
                json.dump(vol_data, f, indent=2)
        except Exception:
            pass

        return {
            "score": round(score, 2),
            "classification": classification,
            "coefficient_of_variation": round(cv, 4),
            "mean_price": round(mean_price, 2),
            "std_price": round(std_price, 2),
        }

    # -------------------------------------------------------------------------
    # ALERT EVALUATION
    # -------------------------------------------------------------------------

    def disk_health(self) -> dict:
        """
        Return disk usage metrics for the forecasts log directory.
        """
        try:
            forecasts_mb = round(_FORECASTS_PATH.stat().st_size / (1024 * 1024), 2) if _FORECASTS_PATH.exists() else 0.0
        except Exception:
            forecasts_mb = 0.0

        # Sum archived .gz files
        archive_mb = 0.0
        try:
            dir_path = _FORECASTS_PATH.parent
            for f in dir_path.glob("*.jsonl.*.gz"):
                archive_mb += f.stat().st_size / (1024 * 1024)
            archive_mb = round(archive_mb, 2)
        except Exception:
            pass

        # Disk usage percent (best effort via shutil)
        disk_percent = None
        try:
            usage = shutil.disk_usage(_FORECASTS_PATH.parent)
            disk_percent = round((usage.used / usage.total) * 100, 1)
        except Exception:
            pass

        total_forecast_mb = round(forecasts_mb + archive_mb, 2)

        return {
            "forecasts_log_size_mb": forecasts_mb,
            "archive_size_mb": archive_mb,
            "total_forecast_storage_mb": total_forecast_mb,
            "disk_usage_percent": disk_percent,
            "rotation_max_bytes": _LOG_ROTATION_MAX_BYTES,
            "rotation_backup_count": _LOG_ROTATION_BACKUP_COUNT,
            "archive_retention_days": _LOG_ARCHIVE_DAYS,
            "healthy": disk_percent is not None and disk_percent < 90,
            "timestamp": _dt.utcnow().isoformat(),
        }

    def get_seasonal_demand_signal(self) -> float:
        """
        Return a demand multiplier (0.5–3.0) based on the current month
        versus the Indian harvest calendar. Higher values indicate predicted
        traffic spikes during harvest seasons when farmers check prices and
        request yield predictions most frequently.
        """
        month = _dt.now().month

        # Peak harvest months = highest API traffic
        peak = {3: 2.5, 4: 2.5, 9: 3.0, 10: 3.0}   # Rabi + Kharif
        high = {2: 1.8, 5: 1.5, 6: 2.0, 7: 2.0, 8: 1.8, 11: 1.5}

        if month in peak:
            return peak[month]
        elif month in high:
            return high[month]
        else:
            return 1.0  # Dec, Jan — off-season trough

    def check_alerts(self, db, send_fn) -> List[dict]:
        """
        Evaluate all farmer price alerts against current forecasts.
        Send WhatsApp alerts for triggered thresholds.
        """
        triggered = []

        if db is None:
            return triggered

        try:
            # Fetch all farmers with price alerts
            users = db.collection("users").stream()
            for user_doc in users:
                uid = user_doc.id
                user_data = user_doc.to_dict() or {}
                phone = user_data.get("phoneNumber") or user_data.get("phone_number") or user_data.get("phone")

                # Get price alerts subcollection
                alert_docs = db.collection("users").document(uid).collection("price_alerts").stream()
                for alert_doc in alert_docs:
                    alert = alert_doc.to_dict() or {}
                    crop = alert.get("crop")
                    threshold_type = alert.get("threshold_type")  # "above", "below", "volatility"
                    threshold_value = alert.get("threshold_value")

                    if not crop or threshold_type not in {"above", "below", "volatility"}:
                        continue

                    # Get latest forecast
                    forecast = self.forecast(crop, days=7)
                    if "error" in forecast:
                        continue

                    latest_price = forecast["forecast"][0]["price"] if forecast["forecast"] else None
                    volatility = forecast.get("volatility", {})

                    triggered_alert = None

                    if threshold_type == "above" and latest_price and latest_price >= threshold_value:
                        triggered_alert = {
                            "type": "price_above",
                            "crop": crop,
                            "current_price": latest_price,
                            "threshold": threshold_value,
                            "message": f"📈 {crop} price has crossed ₹{threshold_value}/qtl! Current: ₹{latest_price}/qtl. Consider selling.",
                        }
                    elif threshold_type == "below" and latest_price and latest_price <= threshold_value:
                        triggered_alert = {
                            "type": "price_below",
                            "crop": crop,
                            "current_price": latest_price,
                            "threshold": threshold_value,
                            "message": f"📉 {crop} price has dropped below ₹{threshold_value}/qtl! Current: ₹{latest_price}/qtl. Consider buying or hedging.",
                        }
                    elif threshold_type == "volatility" and volatility.get("score", 0) >= threshold_value:
                        triggered_alert = {
                            "type": "volatility_high",
                            "crop": crop,
                            "volatility_score": volatility.get("score"),
                            "threshold": threshold_value,
                            "message": f"⚠️ {crop} market volatility is high (score: {volatility['score']:.1f}). Expect price swings. Consider staggered selling.",
                        }

                    if triggered_alert and phone:
                        try:
                            send_fn(phone, triggered_alert["message"])
                            triggered.append({
                                "uid": uid,
                                "phone": phone[-4:],
                                "alert": triggered_alert,
                                "sent_at": _dt.utcnow().isoformat(),
                            })
                        except Exception as exc:
                            logger.warning("Failed sending price alert to %s: %s", phone[-4:], exc)

        except Exception as exc:
            logger.error("Alert check failed: %s", exc)

        return triggered


# =============================================================================
# SINGLETON
# =============================================================================

_forecaster: Optional[PriceForecaster] = None


def get_price_forecaster() -> PriceForecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = PriceForecaster()
    return _forecaster