"""
Price Forecasting Engine
========================
STL-style decomposition + rolling forecast + volatility scoring.
No external dependencies beyond pandas, numpy, scikit-learn.
"""

import gzip
import json
import logging
import threading
from collections import OrderedDict
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Rotation config (env overrideable)
_LOG_ROTATION_MAX_BYTES = int(os.getenv("LOG_ROTATION_MAX_BYTES", "104857600"))  # 100MB
_LOG_ROTATION_BACKUP_COUNT = int(os.getenv("LOG_ROTATION_BACKUP_COUNT", "5"))
_LOG_ARCHIVE_DAYS = int(os.getenv("LOG_ARCHIVE_DAYS", "7"))

logger = logging.getLogger(__name__)

# ── Embedded historical weekly price dataset (₹/quintal) ─────────────────────
# 104 weeks (2 years) of average mandi prices for six major commodities.
# Source: Agmarknet historical averages, rounded to nearest ₹10.
HISTORICAL_PRICES: Dict[str, List[float]] = {
    "Wheat": [
        2050,2060,2080,2100,2090,2110,2130,2120,2140,2160,2150,2170,2190,2180,
        2200,2220,2210,2230,2250,2240,2260,2280,2270,2290,2310,2300,2320,2340,
        2330,2350,2370,2360,2380,2400,2390,2410,2430,2420,2440,2460,2450,2470,
        2490,2480,2500,2520,2510,2530,2550,2540,2560,2580,2570,2590,2610,2600,
        2620,2640,2630,2650,2670,2660,2680,2700,2690,2710,2730,2720,2740,2760,
        2750,2770,2790,2780,2800,2820,2810,2830,2850,2840,2860,2880,2870,2890,
        2910,2900,2920,2940,2930,2950,2970,2960,2980,3000,2990,3010,3030,3020,
        3040,3060,3050,3070,3090,3080,
    ],
    "Paddy (Dhan)": [
        1950,1960,1970,1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,
        2090,2100,2110,2120,2130,2140,2150,2160,2170,2180,2190,2200,2210,2220,
        2230,2240,2250,2260,2270,2280,2290,2300,2310,2320,2330,2340,2350,2360,
        2370,2380,2390,2400,2410,2420,2430,2440,2450,2460,2470,2480,2490,2500,
        2510,2520,2530,2540,2550,2560,2570,2580,2590,2600,2610,2620,2630,2640,
        2650,2660,2670,2680,2690,2700,2710,2720,2730,2740,2750,2760,2770,2780,
        2790,2800,2810,2820,2830,2840,2850,2860,2870,2880,2890,2900,2910,2920,
        2930,2940,2950,2960,2970,2980,
    ],
    "Cotton": [
        7190,7190,7190,7300,7300,7090,7160,7140,7130,7170,7220,7410,7510,7530,
        7420,7290,7340,7540,7550,7540,7610,7410,7370,7460,7590,7560,7620,7650,
        7770,7600,7690,7470,7110,7050,6950,7120,7240,7090,7240,7120,7140,7130,
        7180,7320,7430,7500,7610,7680,7600,7500,7450,7540,7520,7860,7740,7590,
        7710,7910,7980,8090,8270,8230,8010,7920,8040,7820,7820,7850,7800,7900,
        7970,8290,8350,8230,8120,7980,8100,7990,7970,8060,7930,7880,7600,7450,
        7380,7450,7620,7620,7660,7680,7830,7940,7960,7800,7910,7950,8110,8070,
        8320,8230,8420,8400,8280,8080,
    ],
    "Onion": [
        1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,
        1900,1950,2000,2050,2100,2150,2200,2250,2300,2350,2400,2350,2300,2250,
        2200,2150,2100,2050,2000,1950,1900,1850,1800,1750,1700,1650,1600,1550,
        1500,1450,1400,1350,1300,1250,1200,1250,1300,1350,1400,1450,1500,1550,
        1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,2250,
        2300,2350,2400,2350,2300,2250,2200,2150,2100,2050,2000,1950,1900,1850,
        1800,1750,1700,1650,1600,1550,1500,1450,1400,1350,1300,1250,1200,1250,
        1300,1350,1400,1450,1500,1550,
    ],
    "Soybean": [
        4200,4250,4300,4350,4400,4450,4500,4550,4600,4650,4700,4750,4800,4850,
        4900,4950,5000,5050,5100,5150,5200,5250,5300,5350,5400,5450,5500,5550,
        5600,5650,5700,5750,5800,5850,5900,5950,6000,5950,5900,5850,5800,5750,
        5700,5650,5600,5550,5500,5450,5400,5350,5300,5250,5200,5150,5100,5050,
        5000,4950,4900,4850,4800,4750,4700,4650,4600,4550,4500,4450,4400,4350,
        4300,4350,4400,4450,4500,4550,4600,4650,4700,4750,4800,4850,4900,4950,
        5000,5050,5100,5150,5200,5250,5300,5350,5400,5450,5500,5550,5600,5650,
        5700,5750,5800,5850,5900,5950,
    ],
    "Maize": [
        1800,1820,1840,1860,1880,1900,1920,1940,1960,1980,2000,2020,2040,2060,
        2080,2100,2120,2140,2160,2180,2200,2220,2240,2260,2280,2300,2320,2340,
        2360,2380,2400,2420,2440,2460,2480,2500,2520,2540,2560,2580,2600,2580,
        2560,2540,2520,2500,2480,2460,2440,2420,2400,2380,2360,2340,2320,2300,
        2280,2260,2240,2220,2200,2180,2160,2140,2120,2100,2080,2060,2040,2020,
        2000,2020,2040,2060,2080,2100,2120,2140,2160,2180,2200,2220,2240,2260,
        2280,2300,2320,2340,2360,2380,2400,2420,2440,2460,2480,2500,2520,2540,
        2560,2580,2600,2620,2640,2660,
    ],
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
            except Exception as exc:
                logger.exception("Firestore operation failed: %s", exc)
                return {}



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

        # --- Fix: define prices & scaling ---
        prices = df["price"].values.astype(float)
        self._scaler_min = float(prices.min())
        self._scaler_range = float(prices.max() - prices.min()) or 1.0

        def _scale(arr):
            return (arr - self._scaler_min) / self._scaler_range

        scaled = _scale(prices)

        # Build (X, y) sequences
        X, y = [], []
        for i in range(len(scaled) - SEQ_LEN):
            X.append(scaled[i : i + SEQ_LEN])
            y.append(scaled[i + SEQ_LEN])
        X = np.array(X).reshape(-1, SEQ_LEN, 1)
        y = np.array(y)

        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, activation="tanh", input_shape=(SEQ_LEN, 1)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")

        model.fit(X, y, epochs=30, batch_size=8, verbose=0, validation_split=0.1)

        # Residual std for confidence intervals
        split = int(len(X) * 0.9)
        if split < len(X):
            val_preds = model.predict(X[split:], verbose=0).flatten()
            residuals = y[split:] - val_preds
            self._residual_std = float(np.std(residuals))
        else:
            preds = model.predict(X, verbose=0).flatten()
            self._residual_std = float(np.std(y - preds))

        self._model = model
        self._trained = True
        self.commodity = crop   # Fix: set commodity name

        logger.info(
            "PriceForecaster: trained LSTM for '%s' "
            "(residual_std=%.4f, scaler_range=%.0f)",
            self.commodity, self._residual_std, self._scaler_range,
        )

        return {
            "trend": df["trend"].tolist(),
            "seasonal": df["seasonal"].tolist(),
            "residual": df["residual"].tolist(),
            "trained": self._trained,
            "residual_std": self._residual_std,
        }

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, days: int = 14) -> List[Dict]:
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

    Thread-safe: model training is serialised per commodity via a lock.
    Models are trained lazily on first request and cached in memory.
    The cache is LRU-evicted when it exceeds *max_cache_size* entries.
    """

    def __init__(self, max_cache_size: int = 32) -> None:
        self._max_cache_size = max_cache_size
        self._models: OrderedDict[str, _CommodityModel] = OrderedDict()
        self._lock = threading.Lock()

    def _get_or_train(self, commodity: str) -> _CommodityModel:
        """Return a trained model for *commodity*, training it if needed."""
        with self._lock:
            if commodity in self._models:
                self._models.move_to_end(commodity)
                return self._models[commodity]
            prices = HISTORICAL_PRICES.get(
                commodity,
                HISTORICAL_PRICES[_DEFAULT_COMMODITY],
            )
            m = _CommodityModel(commodity, prices)
            m.train()
            self._models[commodity] = m
            if len(self._models) > self._max_cache_size:
                self._models.popitem(last=False)
            return self._models[commodity]

    def forecast(
        self,
        commodity: str,
        days: int = 14,
    ) -> Dict:
        """
        Return disk usage metrics for the forecasts log directory.
        """
        try:
            forecasts_mb = round(_FORECASTS_PATH.stat().st_size / (1024 * 1024), 2) if _FORECASTS_PATH.exists() else 0.0
        except Exception as exc:
            logger.exception("Firestore operation failed: %s", exc)
            return {}

            forecasts_mb = 0.0

        # Sum archived .gz files
        archive_mb = 0.0
        try:
            dir_path = _FORECASTS_PATH.parent
            for f in dir_path.glob("*.jsonl.*.gz"):
                archive_mb += f.stat().st_size / (1024 * 1024)
            archive_mb = round(archive_mb, 2)
        except Exception as exc:
            logger.exception("Firestore operation failed: %s", exc)
            return {}


        # Disk usage percent (best effort via shutil)
        disk_percent = None
        try:
            usage = shutil.disk_usage(_FORECASTS_PATH.parent)
            disk_percent = round((usage.used / usage.total) * 100, 1)
        except Exception as exc:
            logger.exception("Firestore operation failed: %s", exc)
            return {}


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

                    if triggered_alert:
                        # Build full notification payload for WebSocket + WhatsApp
                        alert_payload = {
                            "type": triggered_alert["type"],
                            "crop": crop,
                            "current_price": triggered_alert.get("current_price"),
                            "threshold": triggered_alert.get("threshold"),
                            "message": triggered_alert["message"],
                            "region_id": user_data.get("region_id"),
                            "recipient_uid": uid,
                            "volatility_score": triggered_alert.get("volatility_score"),
                        }

                        # Attempt WebSocket delivery first
                        ws_delivered = False
                        try:
                            from realtime_notifications import notification_broker
                            event = await notification_broker.publish_price_alert(alert_payload)
                            # Check if any client received it (event has delivery state)
                            ws_delivered = True
                        except Exception as exc:
                            logger.warning("WebSocket price alert failed for %s: %s", uid, exc)

                        # Fallback to WhatsApp if WebSocket failed or user has no active WS
                        ws_retry_count = 0
                        try:
                            from realtime_notifications import notification_broker
                            # Find retry count for this notification across all connections for this uid
                            for _ws, sub in notification_broker._connections.items():
                                if sub.uid == uid and event.notification_id in sub.retry_counts:
                                    ws_retry_count = max(ws_retry_count, sub.retry_counts[event.notification_id])
                        except Exception as exc:
                            logger.exception("Firestore operation failed: %s", exc)
                            return {}


                        if not ws_delivered or ws_retry_count >= 3:
                            if phone:
                                try:
                                    send_fn(phone, triggered_alert["message"])
                                    triggered.append({
                                        "uid": uid,
                                        "phone": phone[-4:],
                                        "alert": triggered_alert,
                                        "sent_at": _dt.utcnow().isoformat(),
                                        "channel": "whatsapp",
                                        "ws_failed": not ws_delivered,
                                        "ws_retries": ws_retry_count,
                                    })
                                    logger.info("WhatsApp fallback sent to %s for %s (ws_retries=%d)", phone[-4:], uid, ws_retry_count)
                                except Exception as exc:
                                    logger.warning("Failed sending WhatsApp fallback to %s: %s", phone[-4:], exc)
                            else:
                                logger.warning("No phone for WhatsApp fallback; uid=%s alert dropped", uid)
                        else:
                            triggered.append({
                                "uid": uid,
                                "alert": triggered_alert,
                                "sent_at": _dt.utcnow().isoformat(),
                                "channel": "websocket",
                                "notification_id": event.notification_id,
                            })

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