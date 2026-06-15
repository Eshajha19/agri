from farm_finance_ai import FarmFinanceAI


def test_financial_analysis_produces_score_and_recommendations():
    engine = FarmFinanceAI()
    result = engine.analyze_financial_profile(
        {
            "farmer_name": "Asha",
            "crop_type": "wheat",
            "acreage": 12,
            "annual_revenue": 900000,
            "annual_operating_cost": 540000,
            "existing_debt": 120000,
            "emergency_fund": 140000,
            "credit_score": 742,
            "requested_loan_amount": 250000,
            "loan_tenure_months": 48,
        }
    )

    assert result["financial_health_score"] >= 60
    assert result["risk_level"] in {"Low", "Moderate"}
    assert result["recommended_loan_amount"] > 0
    assert result["lender_matches"]
    assert result["required_documents"]


def test_high_debt_profile_is_flagged_as_risky():
    engine = FarmFinanceAI()
    result = engine.analyze_financial_profile(
        {
            "farmer_name": "Ravi",
            "crop_type": "tomato",
            "acreage": 4,
            "annual_revenue": 300000,
            "annual_operating_cost": 340000,
            "existing_debt": 280000,
            "emergency_fund": 5000,
            "credit_score": 560,
            "requested_loan_amount": 260000,
            "loan_tenure_months": 24,
        }
    )

    assert result["financial_health_score"] < 60
    assert result["risk_level"] in {"High", "Critical"}
    assert result["max_affordable_emi"] == 0


def test_create_application_persists_and_returns_status():
    engine = FarmFinanceAI()
    application = engine.create_application(
        {
            "farmer_name": "Meera",
            "crop_type": "rice",
            "acreage": 8,
            "annual_revenue": 1200000,
            "annual_operating_cost": 700000,
            "existing_debt": 90000,
            "emergency_fund": 200000,
            "credit_score": 770,
            "requested_loan_amount": 320000,
            "loan_tenure_months": 36,
        }
    )

    stored = engine.get_application(application["application_id"], owner_uid=None)

    assert application["status"] in {"pre_approved", "under_review", "needs_documents"}
    assert stored is not None
    assert stored["application_id"] == application["application_id"]
    assert stored["selected_lender"]


def test_marketplace_lists_multiple_products():
    engine = FarmFinanceAI()
    marketplace = engine.list_marketplace()

    assert len(marketplace) >= 3
    assert {item["lender_name"] for item in marketplace}


def test_zero_revenue_does_not_crash():
    engine = FarmFinanceAI()
    result = engine.analyze_financial_profile(
        {
            "farmer_name": "Zero",
            "crop_type": "wheat",
            "acreage": 10,
            "annual_revenue": 0,
            "annual_operating_cost": 50000,
            "existing_debt": 20000,
            "emergency_fund": 10000,
            "credit_score": 650,
            "requested_loan_amount": 100000,
            "loan_tenure_months": 36,
        }
    )
    assert result["financial_health_score"] == 0
    assert result["risk_level"] == "Critical"
