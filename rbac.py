"""
Role-Based Access Control (RBAC) Enforcement Layer
Provides fine-grained access control across all API routes.
"""

import logging
from typing import List, Dict, Optional, Callable
from enum import Enum
from functools import wraps

from fastapi import HTTPException, status, Request
import firebase_admin
from firebase_admin import auth as firebase_auth, firestore

logger = logging.getLogger(__name__)


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
    def get_db():
        """Get Firestore client."""
        try:
            return firestore.client()
        except Exception:
            return None

    @staticmethod
    async def get_user_role(request: Request) -> Optional[Role]:
        """
        Extract user role from Firebase token and Firestore.
        
        Returns
        -------
        Role or None
            User role, or None if not authenticated
        """
        try:
            # Get token from Authorization header
            auth_header = request.headers.get("Authorization", "")
            if not auth_header:
                return Role.GUEST
            if not auth_header.startswith("Bearer "):
                logger.warning("Missing or invalid Authorization header format")
                return Role.GUEST

            token = auth_header.split(" ")[1]

            # Verify Firebase token
            try:
                decoded_token = firebase_auth.verify_id_token(token)
                uid = decoded_token.get("uid")
            except Exception as exc:
                logger.error("Token verification failed: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired authorization token"
                )

            # Get user role from Firestore
            db = RBACManager.get_db()
            if db is None:
                logger.error("Firestore not available; cannot retrieve user role")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Authentication database service temporarily unavailable"
                )

            try:
                user_doc = db.collection("users").document(uid).get()
            except Exception as exc:
                logger.error("Firestore query failed for user %s: %s", uid, exc)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Authentication database lookup failed"
                )

            if not user_doc.exists:
                logger.warning("User %s not found in Firestore, defaulting to farmer", uid)
                return Role.FARMER

            role_str = user_doc.get("role", "farmer").lower()
            try:
                return Role(role_str)
            except ValueError:
                logger.warning("Invalid role for user %s: %s, defaulting to farmer", uid, role_str)
                return Role.FARMER

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Unexpected error getting user role: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during authorization verification"
            )

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
        user_role = await RBACManager.get_user_role(request)

        if require_all:
            return RBACMatrix.has_all_permissions(user_role, required_permissions)
        else:
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
            user_role = await RBACManager.get_user_role(request)
            logger.warning(
                "Unauthorized access attempt with role: %s, required: %s",
                user_role.value if user_role else "unknown",
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

            # Call original function
            return await func(*args, request=request, **kwargs)

        return wrapper

    return decorator


class RBACMiddleware:
    """
    RBAC logging middleware for tracking access attempts.
    Skips Firebase/Firestore verification for public endpoints to
    avoid unnecessary latency and Firebase API calls.
    """

    PUBLIC_PATH_PREFIXES = frozenset({"/", "/health", "/metrics", "/favicon"})

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        """Log all API requests with user role."""
        path = request.url.path
        if any(path.startswith(prefix) for prefix in self.PUBLIC_PATH_PREFIXES):
            user_role = Role.GUEST
        else:
            user_role = await RBACManager.get_user_role(request)

        # Log the access attempt
        logger.info(
            "API Request - Method: %s, Path: %s, Role: %s",
            request.method,
            path,
            user_role.value if user_role else "unknown",
        )

        response = await call_next(request)
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
