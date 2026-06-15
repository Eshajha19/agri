from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

# Maximum number of loan applications to keep in memory
# Prevents unbounded memory growth when using in-memory storage
MAX_IN_MEMORY_APPLICATIONS = 10000


@dataclass(frozen=True)
class LoanProduct:
    lender_name: str
    product_name: str
    min_credit_score: int
    max_debt_ratio: float
    max_amount_factor: float
    annual_interest_rate: float
    tenure_months: int
    description: str
    requires_collateral: bool = False


@dataclass
class FinanceApplication:
    application_id: str
    farmer_name: str
    crop_type: str
    requested_amount: float
    recommended_amount: float
    selected_lender: str
    status: str
    created_at: str
    assessment_score: float
    risk_level: str
    required_documents: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # owner_uid ties the application to the Firebase UID of the farmer who
    # created it. Used by get_application() to enforce ownership and prevent
    # IDOR — a farmer must not be able to read another farmer's loan profile
    # by guessing or enumerating application IDs.
    owner_uid: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        """Normalise owner_uid so ownership checks behave consistently.

        If owner_uid is an empty string (which could happen if the dataclass
        previously had owner_uid: str = "" before the field was changed to
        Optional[str]) we set it to None so that downstream checks of the
        form 'owner_uid is not None' work predictably.
        """
        if self.owner_uid is not None and self.owner_uid.strip() == "":
            self.owner_uid = None


class FarmFinanceAI:
    """Deterministic finance-planning engine for farm loan recommendations."""

    def __init__(self, repository: Any = None) -> None:
        """
        Initialize FarmFinanceAI with optional persistent repository.
        
        Parameters
        ----------
        repository : FinanceApplicationRepository, optional
            Persistent repository for storing applications. If None, uses in-memory storage only.
        """
        self.loan_products: List[LoanProduct] = [
            LoanProduct(
                lender_name="Regional Cooperative Bank",
                product_name="Crop Growth Loan",
                min_credit_score=600,
                max_debt_ratio=0.35,
                max_amount_factor=0.65,
                annual_interest_rate=7.5,
                tenure_months=48,
                description="Low-cost working capital for seasonal crop cycles.",
            ),
            LoanProduct(
                lender_name="National Agriculture Bank",
                product_name="Kisan Working Capital Loan",
                min_credit_score=650,
                max_debt_ratio=0.45,
                max_amount_factor=0.9,
                annual_interest_rate=8.9,
                tenure_months=60,
                description="Flexible repayment loan for input purchases and expansion.",
            ),
            LoanProduct(
                lender_name="Agri NBFC",
                product_name="Climate Resilience Credit",
                min_credit_score=580,
                max_debt_ratio=0.55,
                max_amount_factor=0.55,
                annual_interest_rate=10.8,
                tenure_months=36,
                description="Faster approval for farms investing in resilience upgrades.",
            ),
            LoanProduct(
                lender_name="Farmer Co-op Finance",
                product_name="Warehouse Receipt Loan",
                min_credit_score=620,
                max_debt_ratio=0.4,
                max_amount_factor=0.7,
                annual_interest_rate=8.1,
                tenure_months=24,
                description="Short-term credit backed by stored produce and receipts.",
                requires_collateral=True,
            ),
        ]
        self.applications: OrderedDict[str, FinanceApplication] = OrderedDict()
        self.repository = repository
        logger.info("FarmFinanceAI initialized with %s", "persistent repository" if repository else "in-memory storage")

    def analyze_financial_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._normalize_payload(payload)
        annual_revenue = data["annual_revenue"]
        annual_operating_cost = data["annual_operating_cost"]
        existing_debt = data["existing_debt"]
        emergency_fund = data["emergency_fund"]
        credit_score = data["credit_score"]
        requested_amount = data["requested_loan_amount"]
        tenure_months = data["loan_tenure_months"]
        crop_type = data["crop_type"]

        annual_profit = annual_revenue - annual_operating_cost
        # Profit margin is undefined/negative when revenue <= 0
        if annual_revenue > 0:
            profit_margin = annual_profit / annual_revenue
            debt_ratio = existing_debt / annual_revenue
        else:
            profit_margin = -1.0  # Indicates invalid/negative revenue scenario
            debt_ratio = 1.0      # Maximum risk
        monthly_surplus = annual_profit / 12 if annual_profit > 0 else 0.0
        emergency_cover_months = emergency_fund / (annual_operating_cost / 12 if annual_operating_cost else 1.0)
        crop_risk = self._crop_risk_factor(crop_type)

        score = 100.0
        score -= crop_risk * 12

        if profit_margin < 0:
            score -= 30
        elif profit_margin < 0.15:
            score -= 15
        elif profit_margin < 0.25:
            score -= 6

        if debt_ratio > 0.6:
            score -= 25
        elif debt_ratio > 0.35:
            score -= 15
        elif debt_ratio > 0.2:
            score -= 8

        if credit_score < 600:
            score -= 18
        elif credit_score < 700:
            score -= 8
        elif credit_score >= 780:
            score += 4

        if emergency_cover_months < 1:
            score -= 15
        elif emergency_cover_months < 3:
            score -= 8

        if requested_amount > annual_revenue * 0.8 > 0:
            score -= 10

        if annual_revenue <= 0:
            score -= 10
            _no_revenue = True
        else:
            _no_revenue = False

        score = max(0, min(100, round(score, 1)))
        risk_level = self._risk_level(score)
        best_product = self._best_product(data, score, debt_ratio)
        lender_matches = self._lender_matches(data, score, debt_ratio)

        max_affordable_emi = round(max(monthly_surplus * 0.35, 0.0), 2)
        recommended_loan_amount = self._recommended_loan_amount(
            annual_revenue=annual_revenue,
            requested_amount=requested_amount,
            max_affordable_emi=max_affordable_emi,
            rate=best_product.annual_interest_rate,
            tenure_months=tenure_months or best_product.tenure_months,
            no_revenue=_no_revenue,
        )

        estimated_emi = round(
            self._calculate_emi(
                principal=recommended_loan_amount,
                annual_interest_rate=best_product.annual_interest_rate,
                tenure_months=tenure_months or best_product.tenure_months,
            ),
            2,
        )

        recommended_documents = [
            "Government-issued photo ID",
            "Land ownership or lease documents",
            "Last 12 months of bank statements",
            "Crop income and expense summary",
            "Kisan / farmer registration proof",
        ]

        action_plan = self._action_plan(score, debt_ratio, emergency_cover_months)

        return {
            "farmer_name": data["farmer_name"],
            "crop_type": crop_type,
            "acreage": data["acreage"],
            "annual_revenue": round(annual_revenue, 2),
            "annual_operating_cost": round(annual_operating_cost, 2),
            "annual_profit": round(annual_profit, 2),
            "profit_margin_pct": round(profit_margin * 100, 1),
            "debt_ratio_pct": round(debt_ratio * 100, 1),
            "credit_score": credit_score,
            "financial_health_score": score,
            "risk_level": risk_level,
            "crop_risk_factor": round(crop_risk, 2),
            "emergency_cover_months": round(emergency_cover_months, 2),
            "max_affordable_emi": max_affordable_emi,
            "recommended_loan_amount": recommended_loan_amount,
            "estimated_emi": estimated_emi,
            "tenure_months": tenure_months or best_product.tenure_months,
            "selected_product": self._product_payload(best_product),
            "lender_matches": lender_matches,
            "required_documents": recommended_documents,
            "action_plan": action_plan,
            "marketplace": [self._product_payload(product) for product in self.loan_products],
        }

    def create_application(self, payload: Dict[str, Any], owner_uid: Optional[str] = None) -> Dict[str, Any]:
        if not owner_uid or not owner_uid.strip():
            raise ValueError("owner_uid is required to create an application")
        analysis = self.analyze_financial_profile(payload)
        requested_lender = (payload.get("selected_lender") or "").strip()
        selected_product = self._select_product(requested_lender, analysis["lender_matches"], analysis["selected_product"])
        application_id = f"LOAN-{uuid4().hex[:10].upper()}"
        status = "pre_approved" if analysis["financial_health_score"] >= 80 else "under_review"
        if analysis["financial_health_score"] < 45:
            status = "needs_documents"

        # Enforce in-memory application limit when no persistent repository is configured
        if self.repository is None and len(self.applications) >= MAX_IN_MEMORY_APPLICATIONS:
            raise RuntimeError(f"In-memory application limit ({MAX_IN_MEMORY_APPLICATIONS}) reached. Configure a persistent repository to continue.")
        
        application = FinanceApplication(
            application_id=application_id,
            farmer_name=analysis["farmer_name"],
            crop_type=analysis["crop_type"],
            requested_amount=float(payload.get("requested_loan_amount") or analysis["recommended_loan_amount"]),
            recommended_amount=analysis["recommended_loan_amount"],
            selected_lender=selected_product["lender_name"],
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
            assessment_score=analysis["financial_health_score"],
            risk_level=analysis["risk_level"],
            required_documents=analysis["required_documents"],
            notes=analysis["action_plan"],
            owner_uid=owner_uid,
        )
        if self.repository:
            app_dict = {
                "application_id": application.application_id,
                "farmer_name": application.farmer_name,
                "crop_type": application.crop_type,
                "requested_amount": application.requested_amount,
                "recommended_amount": application.recommended_amount,
                "selected_lender": application.selected_lender,
                "status": application.status,
                "created_at": application.created_at,
                "assessment_score": application.assessment_score,
                "risk_level": application.risk_level,
                "owner_uid": application.owner_uid,
                "required_documents": application.required_documents,
                "notes": application.notes,
            }
            self.repository.create(app_dict)
            logger.info("Application %s persisted to repository.", application_id)

        self._store_application(application_id, application)

        return {
            "application_id": application.application_id,
            "farmer_name": application.farmer_name,
            "crop_type": application.crop_type,
            "requested_amount": round(application.requested_amount, 2),
            "recommended_amount": round(application.recommended_amount, 2),
            "selected_lender": application.selected_lender,
            "status": application.status,
            "created_at": application.created_at,
            "assessment_score": application.assessment_score,
            "risk_level": application.risk_level,
            "required_documents": application.required_documents,
            "notes": application.notes,
            "estimated_emi": analysis["estimated_emi"],
        }

    def get_application(self, application_id: str, owner_uid: Any = _OWNER_UID_NOT_PROVIDED) -> Optional[Dict[str, Any]]:
        """
        Retrieve a finance application by ID.

        Parameters
        ----------
        application_id : str
            The application ID to look up.
        owner_uid : str, optional
            When provided, the application is only returned if its stored
            owner_uid matches — preventing IDOR where a farmer reads another
            farmer's loan profile by guessing an application ID.
            Pass None to bypass the ownership check (admins / experts).
            When omitted, access is denied by default.
        """
        if owner_uid is _OWNER_UID_NOT_PROVIDED:
            return None
        # Try repository first
        if self.repository:
            try:
                app_dict = self.repository.get(application_id)
                if app_dict:
                    stored_uid = app_dict.get("owner_uid")
                    # Reject records without an owner (orphaned).
                    if not stored_uid:
                        return None
                    # Enforce ownership when a uid filter is supplied.
                    if owner_uid is not None and stored_uid != owner_uid:
                        return None
                    return {
                        "application_id": app_dict.get("application_id"),
                        "farmer_name": app_dict.get("farmer_name"),
                        "crop_type": app_dict.get("crop_type"),
                        "requested_amount": round(app_dict.get("requested_amount", 0), 2),
                        "recommended_amount": round(app_dict.get("recommended_amount", 0), 2),
                        "selected_lender": app_dict.get("selected_lender"),
                        "status": app_dict.get("status"),
                        "created_at": app_dict.get("created_at"),
                        "assessment_score": app_dict.get("assessment_score"),
                        "risk_level": app_dict.get("risk_level"),
                        "owner_uid": app_dict.get("owner_uid", ""),
                        "required_documents": app_dict.get("required_documents", []),
                        "notes": app_dict.get("notes", []),
                    }
            except Exception as exc:
                logger.warning("Failed to retrieve application from repository: %s", exc)

        # Fall back to in-memory storage
        application = self.applications.get(application_id)
        if not application:
            return None

        # Enforce ownership on in-memory records too
        # Reject orphaned records (None owner_uid) entirely.
        if not application.owner_uid:
            return None
        if owner_uid is not None and application.owner_uid != owner_uid:
            return None

        return {
            "application_id": application.application_id,
            "farmer_name": application.farmer_name,
            "crop_type": application.crop_type,
            "requested_amount": round(application.requested_amount, 2),
            "recommended_amount": round(application.recommended_amount, 2),
            "selected_lender": application.selected_lender,
            "status": application.status,
            "created_at": application.created_at,
            "assessment_score": application.assessment_score,
            "risk_level": application.risk_level,
            "owner_uid": application.owner_uid,
            "required_documents": application.required_documents,
            "notes": application.notes,
        }

    def _store_application(self, application_id: str, application: FinanceApplication) -> None:
        if MAX_IN_MEMORY_APPLICATIONS <= 0:
            return
        if application_id in self.applications:
            del self.applications[application_id]
        while len(self.applications) >= MAX_IN_MEMORY_APPLICATIONS:
            self.applications.popitem(last=False)
        self.applications[application_id] = application

    def delete_application(self, application_id: str) -> bool:
        """Delete a finance application from both the repository and the
        in-memory cache.

        The previous design had no delete method on FarmFinanceAI.  Callers
        that needed to remove a record could only call
        ``self.repository.delete(application_id)`` directly, which deleted
        the record from Firestore but left the stale ``FinanceApplication``
        object in ``self.applications``.  A subsequent call to
        ``get_application`` would find nothing in the repository (correct)
        but then fall through to the in-memory dict and return the deleted
        record — silently serving data that should no longer exist.

        This method is the single deletion entry point.  It:
        1. Removes the entry from ``self.applications`` first so the
           in-memory cache is immediately consistent.
        2. Delegates to the repository for durable deletion.
        3. Returns True only when the record existed in at least one of the
           two stores and was successfully removed.
        """
        deleted_from_memory = self.applications.pop(application_id, None) is not None
        deleted_from_repo = False

        if self.repository:
            try:
                deleted_from_repo = self.repository.delete(application_id)
            except Exception as exc:
                logger.error(
                    "Failed to delete application %s from repository: %s",
                    application_id,
                    exc,
                )

        deleted = deleted_from_memory or deleted_from_repo
        if deleted:
            logger.info("Application %s deleted (memory=%s, repo=%s).",
                        application_id, deleted_from_memory, deleted_from_repo)
        else:
            logger.warning("delete_application: application %s not found.", application_id)

        return deleted

    def list_marketplace(self) -> List[Dict[str, Any]]:
        return [self._product_payload(product) for product in self.loan_products]

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def to_float(value: Any, default: float = 0.0) -> float:
            try:
                if value in (None, ""):
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        def to_int(value: Any, default: int = 0) -> int:
            try:
                if value in (None, ""):
                    return default
                return int(float(value))
            except (TypeError, ValueError, OverflowError):
                return default

        return {
            "farmer_name": str(payload.get("farmer_name") or "Farmer").strip(),
            "crop_type": str(payload.get("crop_type") or "Mixed Crops").strip(),
            "acreage": to_float(payload.get("acreage"), 0.0),
            "annual_revenue": to_float(payload.get("annual_revenue"), 0.0),
            "annual_operating_cost": to_float(payload.get("annual_operating_cost"), 0.0),
            "existing_debt": to_float(payload.get("existing_debt"), 0.0),
            "emergency_fund": to_float(payload.get("emergency_fund"), 0.0),
            "credit_score": max(300, min(900, to_int(payload.get("credit_score"), 650))),
            "requested_loan_amount": to_float(payload.get("requested_loan_amount"), 0.0),
            "loan_tenure_months": min(120, max(6, to_int(payload.get("loan_tenure_months"), 36))),
        }

    def _crop_risk_factor(self, crop_type: str) -> float:
        crop = crop_type.lower()
        crop_risks = {
            "rice": 0.35,
            "wheat": 0.22,
            "maize": 0.3,
            "cotton": 0.33,
            "sugarcane": 0.2,
            "vegetables": 0.4,
            "fruits": 0.38,
            "tomato": 0.42,
            "potato": 0.24,
        }
        return crop_risks.get(crop, 0.28)

    def _risk_level(self, score: float) -> str:
        if score >= 80:
            return "Low"
        if score >= 60:
            return "Moderate"
        if score >= 40:
            return "High"
        return "Critical"

    def _best_product(self, payload: Dict[str, Any], score: float, debt_ratio: float) -> LoanProduct:
        matches = self._candidate_products(payload, score, debt_ratio)
        if matches:
            return matches[0]
        return self.loan_products[0]

    def _candidate_products(self, payload: Dict[str, Any], score: float, debt_ratio: float) -> List[LoanProduct]:
        annual_revenue = payload["annual_revenue"]
        credit_score = payload["credit_score"]
        recommended_amount = max(payload["requested_loan_amount"], annual_revenue * 0.25)
        valid_products = []
        for product in self.loan_products:
            if credit_score < product.min_credit_score:
                continue
            if debt_ratio > product.max_debt_ratio:
                continue
            if annual_revenue > 0 and recommended_amount > annual_revenue * product.max_amount_factor:
                continue
            valid_products.append(product)
        return sorted(
            valid_products,
            key=lambda item: (item.annual_interest_rate, -item.max_amount_factor),
        )

    def _lender_matches(self, payload: Dict[str, Any], score: float, debt_ratio: float) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        annual_revenue = payload["annual_revenue"]
        requested_amount = max(payload["requested_loan_amount"], annual_revenue * 0.25)
        for product in self.loan_products:
            fit_score = 100
            if payload["credit_score"] < product.min_credit_score:
                fit_score -= 35
            if debt_ratio > product.max_debt_ratio:
                fit_score -= 25
            if annual_revenue > 0 and requested_amount > annual_revenue * product.max_amount_factor:
                fit_score -= 20
            fit_score -= max(0, 80 - score) * 0.4
            if fit_score < 40:
                continue
            matches.append({
                **self._product_payload(product),
                "fit_score": round(max(0, min(100, fit_score)), 1),
            })
        matches.sort(key=lambda item: (-item["fit_score"], item["annual_interest_rate"]))
        return matches

    def _select_product(self, requested_lender: str, lender_matches: List[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
        if requested_lender:
            for lender in lender_matches:
                if lender["lender_name"].lower() == requested_lender.lower():
                    return lender
            for product in self.loan_products:
                if product.lender_name.lower() == requested_lender.lower():
                    return self._product_payload(product)
        if lender_matches:
            return lender_matches[0]
        return fallback

    def _product_payload(self, product: LoanProduct) -> Dict[str, Any]:
        return {
            "lender_name": product.lender_name,
            "product_name": product.product_name,
            "min_credit_score": product.min_credit_score,
            "max_debt_ratio": product.max_debt_ratio,
            "max_amount_factor": product.max_amount_factor,
            "annual_interest_rate": product.annual_interest_rate,
            "tenure_months": product.tenure_months,
            "description": product.description,
            "requires_collateral": product.requires_collateral,
        }

    def _recommended_loan_amount(
        self,
        annual_revenue: float,
        requested_amount: float,
        max_affordable_emi: float,
        rate: float,
        tenure_months: int,
        no_revenue: bool = False,
    ) -> float:
        candidate = requested_amount or annual_revenue * 0.3
        if no_revenue:
            candidate = max(candidate, 25000.0)
        affordable_principal = self._principal_from_emi(max_affordable_emi, rate, tenure_months)
        revenue_cap = annual_revenue * 0.75 if annual_revenue else candidate
        cap = affordable_principal if affordable_principal > 0 else candidate
        return max(0.0, round(min(candidate, cap, revenue_cap), 2))

    def _principal_from_emi(self, monthly_emi: float, annual_interest_rate: float, tenure_months: int) -> float:
        monthly_rate = annual_interest_rate / 12 / 100
        if monthly_emi <= 0 or tenure_months <= 0:
            return 0.0
        if tenure_months > MAX_LOAN_TENURE_MONTHS:
            tenure_months = MAX_LOAN_TENURE_MONTHS
        if monthly_rate == 0:
            return monthly_emi * tenure_months
        try:
            growth = (1 + monthly_rate) ** tenure_months
        except OverflowError:
            return monthly_emi / monthly_rate
        return monthly_emi * ((growth - 1) / (monthly_rate * growth))

    def _calculate_emi(self, principal: float, annual_interest_rate: float, tenure_months: int) -> float:
        if principal <= 0 or tenure_months <= 0:
            return 0.0
        monthly_rate = annual_interest_rate / 12 / 100
        if tenure_months > MAX_LOAN_TENURE_MONTHS:
            tenure_months = MAX_LOAN_TENURE_MONTHS
        if monthly_rate == 0:
            return principal / tenure_months
        try:
            growth = (1 + monthly_rate) ** tenure_months
        except OverflowError:
            return principal * monthly_rate
        return principal * monthly_rate * growth / (growth - 1)

    def _action_plan(self, score: float, debt_ratio: float, emergency_cover_months: float) -> List[str]:
        plan = []
        if score >= 80:
            plan.append("Profile is strong enough for standard agricultural working-capital products.")
        elif score >= 60:
            plan.append("Improve input-output records to unlock better pricing and faster approval.")
        else:
            plan.append("Reduce revolving debt before applying to improve approval chances.")

        if debt_ratio > 0.35:
            plan.append("Rework liabilities and avoid a request above current repayment capacity.")
        if emergency_cover_months < 2:
            plan.append("Build a small emergency reserve for weather and price shocks.")
        if len(plan) < 3:
            plan.append("Submit the required documents early to speed up review.")
        return plan[:3]
