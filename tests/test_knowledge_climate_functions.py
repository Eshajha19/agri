from backend.routers import knowledge


def test_compute_impact_score_baseline_is_100():
    profile = {"temp_coeff": -0.04, "rain_coeff": 0.02}

    score = knowledge._compute_impact_score(
        sim_temp=30.0,
        sim_rain=200.0,
        profile=profile,
        base_temp=30.0,
        base_rain=200.0,
    )

    assert score == 100.0


def test_compute_impact_score_clamps_to_upper_bound():
    profile = {"temp_coeff": -0.04, "rain_coeff": 0.02}

    score = knowledge._compute_impact_score(
        sim_temp=10.0,
        sim_rain=600.0,
        profile=profile,
        base_temp=30.0,
        base_rain=200.0,
    )

    assert score == 150.0


def test_compute_impact_score_clamps_to_lower_bound():
    profile = {"temp_coeff": -0.04, "rain_coeff": 0.02}

    score = knowledge._compute_impact_score(
        sim_temp=80.0,
        sim_rain=0.0,
        profile=profile,
        base_temp=30.0,
        base_rain=200.0,
    )

    assert score == 0.0


def test_build_recommendations_includes_heat_drought_and_significant_impact_guidance():
    profile = {"opt_temp": (15, 25), "opt_rain": (40, 80)}

    recs = knowledge._build_recommendations(
        sim_temp=40.0,
        sim_rain=10.0,
        profile=profile,
        impact_score=50.0,
        crop_type="wheat",
        season="kharif",
    )

    assert any("High heat stress expected" in rec for rec in recs)
    assert any("Severe drought risk" in rec for rec in recs)
    assert any("Significant yield reduction expected" in rec for rec in recs)


def test_build_recommendations_favourable_conditions_path():
    profile = {"opt_temp": (20, 30), "opt_rain": (80, 120)}

    recs = knowledge._build_recommendations(
        sim_temp=25.0,
        sim_rain=100.0,
        profile=profile,
        impact_score=120.0,
        crop_type="tomato",
        season="zaid",
    )

    assert len(recs) == 1
    assert "Favourable conditions projected" in recs[0]


def test_build_recommendations_standard_practices_path():
    profile = {"opt_temp": (25, 35), "opt_rain": (150, 300)}

    recs = knowledge._build_recommendations(
        sim_temp=30.0,
        sim_rain=200.0,
        profile=profile,
        impact_score=100.0,
        crop_type="rice",
        season="kharif",
    )

    assert len(recs) == 1
    assert "Conditions are within the acceptable range" in recs[0]
