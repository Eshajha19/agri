"""
Role-Based Access Control (RBAC) Enforcement Layer
Provides fine-grained access control across all API routes.

Authorization model
-------------------
* **Authoritative role source (API):** Firestore ``users/{uid}.role``.
* **JWT custom claim ``role``:** Mirror of Firestore, set via ``role_sync.sync_role_claim``.
  The API rejects requests when the claim is present but disagrees with Firestore
  (stale token after demotion or role change).
* **Firestore security rules:** Read ``request.auth.token.role`` only (no ``get()`` on
  users). Claims must be kept in sync by the backend; rules do not default missing
  claims to elevated roles.
* **Unauthenticated callers:** Treated as ``Role.GUEST`` only when no Bearer token is
  supplied. Valid tokens without a user profile fail closed (403), never guest.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Callable, Dict, List, Optional

import firebase_admin
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

__all__ = [
    "RBACManager",
    "Permission",
    "RBACMiddleware",
    "RBACMatrix",
    "AuthContext",
    "Role",
    "require_permission",
    "print_rbac_matrix",
]

from firebase_admin import auth as firebase_auth, firestore

logger = logging.getLogger(__name__)

# Must match role_sync.VALID_ROLES and firestore.rules role strings.
KNOWN_ROLES = frozenset({"admin", "expert", "farmer", "vendor", "system", "guest"})
DEFAULT_PROFILE_ROLE = "farmer"
ROLE_PRECEDENCE = ("admin", "expert", "system", "vendor", "farmer", "guest")
TENANT_FIELD_CANDIDATES = ("tenant_id", "tenantId", "organization_id", "org_id")
STALE_TOKEN_DETAIL = (
    "Authorization token is stale. Sign out and sign in again to refresh your session."
)


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Resolved identity for an authenticated API request."""

    uid: str
    role: str
    roles: tuple[str, ...] = ()
    tenant_id: Optional[str] = None


class Role(Enum):
    """Application roles."""
    ADMIN = "admin"
    EXPERT = "expert"
    FARMER = "farmer"
    VENDOR = "vendor"
    SYSTEM = "system"
    GUEST = "guest"


class Permission(Enum):
    """Fine-grained permissions."""
    # Finance
    FINANCE_CREATE = "finance:create"
    FINANCE_READ_OWN = "finance:read:own"
    FINANCE_READ_ALL = "finance:read:all"
    FINANCE_UPDATE_OWN = "finance:update:own"
    FINANCE_UPDATE_ALL = "finance:update:all"
    FINANCE_DELETE = "finance:delete"

    # Supply Chain
    SUPPLY_CHAIN_CREATE = "supply_chain:create"
    SUPPLY_CHAIN_READ = "supply_chain:read"
    SUPPLY_CHAIN_UPDATE = "supply_chain:update"
    SUPPLY_CHAIN_DELETE = "supply_chain:delete"

    # Notifications
    NOTIFICATIONS_READ = "notifications:read"
    NOTIFICATIONS_CREATE = "notifications:create"
    NOTIFICATIONS_DELETE = "notifications:delete"

    # Reports
    REPORTS_CREATE = "reports:create"
    REPORTS_READ_OWN = "reports:read:own"
    REPORTS_READ_ALL = "reports:read:all"
    REPORTS_DELETE = "reports:delete"

    # Quality Grading
    QUALITY_ASSESS = "quality:assess"
    QUALITY_READ = "quality:read"

    # Seeds
    SEEDS_VERIFY = "seeds:verify"
    SEEDS_READ = "seeds:read"

    # WhatsApp
    WHATSAPP_SUBSCRIBE = "whatsapp:subscribe"
    WHATSAPP_TRIGGER = "whatsapp:trigger"
    WHATSAPP_WEBHOOK = "whatsapp:webhook"

    # System
    SYSTEM_LOG = "system:log"
    SYSTEM_ADMIN = "system:admin"
    RAG_QUERY = "rag:query"
    CLIMATE_SIMULATE = "climate:simulate"


class RBACMatrix:
    """
    Role-based access control matrix.
    Maps roles to their permissions.
    """

    ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
        Role.ADMIN: [
            # Admin can do everything
            Permission.FINANCE_CREATE,
            Permission.FINANCE_READ_ALL,
            Permission.FINANCE_UPDATE_ALL,
            Permission.FINANCE_DELETE,
            Permission.SUPPLY_CHAIN_CREATE,
            Permission.SUPPLY_CHAIN_READ,
            Permission.SUPPLY_CHAIN_UPDATE,
            Permission.SUPPLY_CHAIN_DELETE,
            Permission.NOTIFICATIONS_READ,
            Permission.NOTIFICATIONS_CREATE,
            Permission.NOTIFICATIONS_DELETE,
            Permission.REPORTS_READ_ALL,
            Permission.REPORTS_DELETE,
            Permission.QUALITY_ASSESS,
            Permission.QUALITY_READ,
            Permission.SEEDS_VERIFY,
            Permission.SEEDS_READ,
            Permission.WHATSAPP_SUBSCRIBE,
            Permission.WHATSAPP_TRIGGER,
            Permission.WHATSAPP_WEBHOOK,
            Permission.SYSTEM_LOG,
            Permission.SYSTEM_ADMIN,
            Permission.RAG_QUERY,
            Permission.CLIMATE_SIMULATE,
            Permission.REPORTS_CREATE,
            Permission.FINANCE_UPDATE_OWN,
            Permission.FINANCE_READ_OWN,
        ],

        Role.EXPERT: [
            # Expert: Read finance/supply chain, assess quality, verify seeds
            Permission.FINANCE_READ_ALL,
            Permission.SUPPLY_CHAIN_READ,
            Permission.NOTIFICATIONS_READ,
            Permission.REPORTS_READ_ALL,
            Permission.REPORTS_CREATE,
            Permission.QUALITY_ASSESS,
            Permission.QUALITY_READ,
            Permission.SEEDS_VERIFY,
            Permission.SEEDS_READ,
            Permission.RAG_QUERY,
            Permission.CLIMATE_SIMULATE,
        ],

        Role.FARMER: [
            # Farmer: Read own finance, create supply chain, quality checks
            Permission.FINANCE_CREATE,
            Permission.FINANCE_READ_OWN,
            Permission.FINANCE_UPDATE_OWN,
            Permission.SUPPLY_CHAIN_CREATE,
            Permission.SUPPLY_CHAIN_READ,
            Permission.SUPPLY_CHAIN_UPDATE,
            Permission.NOTIFICATIONS_READ,
            Permission.REPORTS_CREATE,
            Permission.REPORTS_READ_OWN,
            Permission.QUALITY_ASSESS,
            Permission.QUALITY_READ,
            Permission.SEEDS_READ,
            Permission.WHATSAPP_SUBSCRIBE,
            Permission.RAG_QUERY,
            Permission.CLIMATE_SIMULATE,
        ],

        Role.VENDOR: [
            # Vendor: Read supply chain, manage marketplace
            Permission.SUPPLY_CHAIN_READ,
            Permission.SUPPLY_CHAIN_CREATE,
            Permission.SUPPLY_CHAIN_UPDATE,
            Permission.NOTIFICATIONS_READ,
            Permission.QUALITY_READ,
            Permission.SEEDS_READ,
            Permission.WHATSAPP_SUBSCRIBE,
            Permission.RAG_QUERY,
            Permission.CLIMATE_SIMULATE,
        ],

        Role.SYSTEM: [
            # System: All permissions (for internal processes)
            Permission.FINANCE_CREATE,
            Permission.FINANCE_READ_ALL,
            Permission.FINANCE_UPDATE_ALL,
            Permission.SUPPLY_CHAIN_CREATE,
            Permission.SUPPLY_CHAIN_READ,
            Permission.SUPPLY_CHAIN_UPDATE,
            Permission.NOTIFICATIONS_CREATE,
            Permission.SYSTEM_ADMIN,
            Permission.SYSTEM_LOG,
            Permission.WHATSAPP_WEBHOOK,
        ],

        Role.GUEST: [
            # Guest: Read-only public data
            Permission.RAG_QUERY,
            Permission.CLIMATE_SIMULATE,
            Permission.SEEDS_READ,
        ],
    }

    @classmethod
    def has_permission(cls, role: Role, permission: Permission) -> bool:
        """Check if role has permission."""
        permissions = cls.ROLE_PERMISSIONS.get(role, [])
        return permission in permissions

    @classmethod
    def has_any_permission(cls, role: Role, permissions: List[Permission]) -> bool:
        """Check if role has any of the given permissions."""
        return any(cls.has_permission(role, perm) for perm in permissions)

    @classmethod
    def has_all_permissions(cls, role: Role, permissions: List[Permission]) -> bool:
        """Check if role has all given permissions."""
        return all(cls.has_permission(role, perm) for perm in permissions)


class RBACManager:
    """Manager for authentication and authorization."""

    @staticmethod
    def _normalize_roles(profile: Dict) -> List[str]:
        """Return normalized roles from user profile, falling back to `role`."""
        raw_roles = profile.get("roles")
        normalized: List[str] = []

        if isinstance(raw_roles, list):
            for item in raw_roles:
                role = str(item).strip().lower()
                if role in KNOWN_ROLES and role not in normalized:
                    normalized.append(role)

        if not normalized:
            role_str = profile.get("role", DEFAULT_PROFILE_ROLE)
            role = str(role_str).strip().lower()
            if role not in KNOWN_ROLES:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid role assigned to user profile",
                )
            normalized.append(role)

        return normalized

    @staticmethod
    def _effective_role(roles: List[str]) -> str:
        """Pick the most privileged role for compatibility with role-based checks."""
        for preferred in ROLE_PRECEDENCE:
            if preferred in roles:
                return preferred
        return DEFAULT_PROFILE_ROLE

    @staticmethod
    def _extract_tenant(value: Dict) -> Optional[str]:
        for field in TENANT_FIELD_CANDIDATES:
            tenant = value.get(field)
            if isinstance(tenant, str) and tenant.strip():
                return tenant.strip()
        return None

    @staticmethod
    def can_admin_or_expert_override(
        ctx: AuthContext,
        *,
        resource_owner_uid: Optional[str] = None,
        resource_tenant_id: Optional[str] = None,
        allow_cross_tenant: bool = False,
    ) -> bool:
        """Return True if caller may override ownership constraints."""
        if ctx.role not in ("admin", "expert"):
            return False

        if resource_owner_uid and resource_owner_uid == ctx.uid:
            return True

        if resource_tenant_id and ctx.tenant_id and not allow_cross_tenant:
            return resource_tenant_id == ctx.tenant_id

        if resource_tenant_id and ctx.tenant_id is None and not allow_cross_tenant:
            return False

        return True

    @staticmethod
    def assert_tenant_scope(
        ctx: AuthContext,
        resource_tenant_id: Optional[str],
        *,
        allow_cross_tenant_admin: bool = False,
    ) -> None:
        """Raise 403 when caller crosses tenant boundary without explicit allowance."""
        if not resource_tenant_id:
            return
        if not ctx.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context missing for authenticated user",
            )

        if ctx.tenant_id == resource_tenant_id:
            return

        if allow_cross_tenant_admin and ctx.role == "admin":
            return

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: cross-tenant request not permitted",
        )

    @staticmethod
    def get_db():
        """Get Firestore client."""
        try:
            return firestore.client()
        except Exception:
            return None

    @staticmethod
    def _parse_bearer_token(request: Request) -> Optional[str]:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        token = auth_header[7:].strip()
        return token or None

    @staticmethod
    async def resolve_auth_context(
        request: Request,
        *,
        allow_unauthenticated: bool = False,
    ) -> Optional[AuthContext]:
        """
        Resolve the caller's UID and role from Firestore (authoritative).

        When ``allow_unauthenticated`` is True and no Bearer token is present,
        returns None. Otherwise fail-closed with 401/403/503 matching ``verify_role``.
        """
        token = RBACManager._parse_bearer_token(request)
        if token is None:
            if allow_unauthenticated:
                return None
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authentication token",
            )

        loop = asyncio.get_running_loop()

        try:
            decoded_token = firebase_auth.verify_id_token(token, check_revoked=True)
        except firebase_auth.RevokedIdTokenError as exc:
            logger.error("Token revoked: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session revoked. Please sign in again.",
            ) from exc
        except Exception as exc:
            logger.error("Token verification failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authorization token",
            ) from exc

        uid = decoded_token.get("uid")
        if not uid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authorization token",
            )

        db = RBACManager.get_db()
        if not db:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication database unavailable",
            )

        try:
            # Firestore's .get() is a blocking network call; run it off the
            # event loop to avoid stalling concurrent coroutines.
            user_doc = await loop.run_in_executor(
                None, db.collection("users").document(uid).get
            )
        except Exception as exc:
            logger.error("Firestore query failed for user %s: %s", uid, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication database lookup failed",
            ) from exc

        claim_tenant = RBACManager._extract_tenant(decoded_token)
        if claim_tenant and tenant_id and claim_tenant != tenant_id:
            logger.warning(
                "Stale JWT tenant for uid=%s: claim=%s firestore=%s",
                uid,
                claim_tenant,
                tenant_id,
            )







            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=STALE_TOKEN_DETAIL,
            )

        return AuthContext(uid=uid, role=role_str, roles=tuple(roles), tenant_id=tenant_id)

    @staticmethod
    async def get_user_role(request: Request) -> Role:
        """
        Return the caller's role.

        Unauthenticated requests (no Bearer token) map to ``Role.GUEST``.
        Authenticated requests use Firestore and fail closed on missing profiles
        or stale JWT claims.
        """
        ctx = await RBACManager.resolve_auth_context(
            request,
            allow_unauthenticated=True,
        )
        if ctx is None:
            return Role.GUEST
        return Role(ctx.role)

    @staticmethod
    async def verify_permission(
        request: Request,
        required_permissions: List[Permission],
        require_all: bool = False,
    ) -> bool:
        """
        Verify user has required permissions.
        
        Parameters
        ----------
        request : Request
            FastAPI request object
        required_permissions : list of Permission
            Permissions to check
        require_all : bool
            If True, user must have ALL permissions (AND logic)
            If False, user must have ANY permission (OR logic)
        
        Returns
        -------
        bool
            True if user has required permissions
        """
        ctx = await RBACManager.resolve_auth_context(
            request,
            allow_unauthenticated=False,
        )
        user_role = Role(ctx.role)

        if require_all:
            return RBACMatrix.has_all_permissions(user_role, required_permissions)
        return RBACMatrix.has_any_permission(user_role, required_permissions)

    @staticmethod
    async def raise_if_unauthorized(
        request: Request,
        required_permissions: List[Permission],
        require_all: bool = False,
        detail: str = "Insufficient permissions",
    ) -> None:
        """
        Raise HTTPException if user lacks permissions.
        
        Parameters
        ----------
        request : Request
            FastAPI request object
        required_permissions : list of Permission
            Permissions to check
        require_all : bool
            If True, user must have ALL permissions (AND logic)
        detail : str
            Error message
        
        Raises
        ------
        HTTPException
            If user lacks required permissions
        """
        has_permission = await RBACManager.verify_permission(
            request, required_permissions, require_all=require_all
        )

        if not has_permission:
            try:
                ctx = await RBACManager.resolve_auth_context(
                    request,
                    allow_unauthenticated=False,
                )
                role_label = ctx.role
            except HTTPException:
                role_label = "unknown"
            logger.warning(
                "Unauthorized access attempt with role: %s, required: %s",
                role_label,
                [p.value for p in required_permissions],
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=detail,
            )


def require_permission(*permissions: Permission, require_all: bool = False):
    """
    Decorator to enforce permission requirements on FastAPI endpoints.
    
    Parameters
    ----------
    *permissions : Permission
        One or more permissions required
    require_all : bool
        If True, user must have ALL permissions (AND logic)
        If False, user must have ANY permission (OR logic)
    
    Returns
    -------
    Callable
        Decorated function with permission check
    
    Examples
    --------
    @app.post("/api/finance/applications")
    @require_permission(Permission.FINANCE_CREATE)
    async def create_application(request: Request, payload: Dict):
        ...
    
    @app.delete("/api/applications/{id}")
    @require_permission(Permission.FINANCE_DELETE, require_all=True)
    async def delete_application(id: str, request: Request):
        ...
    """
    required_perms = list(permissions)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs):
            # Try to find request in args or kwargs
            if request is None:
                # For dependency injection, request might be in kwargs
                for key, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break

            if request is None:
                logger.error("Request object not found in function parameters")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                )

            # Check permissions
            await RBACManager.raise_if_unauthorized(
                request,
                required_perms,
                require_all=require_all,
                detail=f"Required permissions: {', '.join(p.value for p in required_perms)}",
            )

            # Avoid passing duplicate request argument
            if "request" in kwargs:
                return await func(*args, **kwargs)

            return await func(*args, request=request, **kwargs)

        return wrapper

    return decorator


class RBACMiddleware(BaseHTTPMiddleware):
    """
    RBAC logging middleware for tracking access attempts.
    Skips Firebase/Firestore verification for public endpoints to
    avoid unnecessary latency and Firebase API calls.
    """

    PUBLIC_PATH_PREFIXES = frozenset({"/health", "/metrics", "/favicon"})

    def __init__(self, app):
        super().__init__(app)

    async def __call__(self, request: Request, call_next):
        """Log all API requests."""
        path = request.url.path
        response = await call_next(request)
        logger.info(
            "API Response - Method: %s, Path: %s, Status: %s",
            request.method,
            path,
            response.status_code,
        )
        return response


def print_rbac_matrix() -> str:
    """Generate human-readable RBAC matrix."""
    lines = [
        "\n" + "=" * 100,
        "RBAC ENFORCEMENT MATRIX",
        "=" * 100,
    ]

    for role in Role:
        permissions = RBACMatrix.ROLE_PERMISSIONS.get(role, [])
        lines.append(f"\n{role.value.upper()} ({len(permissions)} permissions):")
        lines.append("-" * 50)

        # Group permissions by category
        categories = {}
        for perm in permissions:
            category = perm.value.split(":")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(perm.value)

        for category in sorted(categories.keys()):
            perms = sorted(categories[category])
            lines.append(f"  {category}:")
            for perm in perms:
                lines.append(f"    ✓ {perm}")

    lines.append("\n" + "=" * 100 + "\n")
    return "\n".join(lines)

from fastapi import HTTPException

async def verify_role(
    token: str,
    required_roles: list[str] | None = None,
    require_all: bool = False,
) -> dict:
    """
    Verify Firebase token and enforce RBAC role checks.
    Returns dict with uid and roles if authorized.
    """
    try:
        user = await RBACManager.decode_token(token)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    except FirestoreUnavailable:
        raise HTTPException(status_code=503, detail="Authorization service unavailable")

    roles = user.get("roles", [])
    uid = user.get("uid")

    if required_roles:
        if require_all:
            has_access = all(role in roles for role in required_roles)
        else:
            has_access = any(role in roles for role in required_roles)

        if not has_access:
            raise HTTPException(status_code=403, detail="Insufficient permissions")

    return {"uid": uid, "roles": roles}
