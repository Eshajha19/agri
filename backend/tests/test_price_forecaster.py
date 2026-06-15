import pandas as pd
from backend.price_forecaster import PriceForecaster

def test_stl_decompose_with_seed_data():
    pf = PriceForecaster()
    # Seed 20 days of fake prices
    dates = pd.date_range("2026-05-01", periods=20, freq="D")
    for i, d in enumerate(dates):
        pf.add_price("wheat", price=200 + i, date=str(d.date()))

    result = pf.stl_decompose("wheat")
    assert "trend" in result
    assert "seasonal" in result
    assert "residual" in result
    assert result["trained"] is True
    assert result["residual_std"] >= 0
