import logging

logger = logging.getLogger(__name__)

ROUTER_DIAGNOSTICS = {
    "duplicate_prefixes": [],
    "missing_routers": [],
    "registered_routers": [],
    "warnings": [],
    "conflicts": [],
    "health_status": "unknown",
    "severity_counts": {
        "critical": 0,
        "warning": 0,
    },
    "startup_summary": {},
}

FAIL_FAST_ON_ROUTER_CONFLICTS = False

def register_routers(
    app,
    ml,
    governance,
    finance,
    quality,
    blockchain,
    reports,
    marketplace,
    knowledge,
    community,
    referrals,
    platform,
    advisory,
    alerts,
    flags_router,
    lms,
    voice_assistant_router,


):
    logger.info("Starting router registration")

    registered_prefixes = {}

    def register(router_obj, prefix="", tags=None, router_name="Unknown"):
        if router_obj is None:
            logger.warning(
                "Skipping router registration: %s is None",
                router_name,
            )

            ROUTER_DIAGNOSTICS["missing_routers"].append(
                router_name
            )

            return

        if prefix in registered_prefixes:

            conflict = {
                "severity": "critical",
                "type": "duplicate_prefix",
                "prefix": prefix,
                "existing_router": registered_prefixes[prefix],
                "incoming_router": router_name,
                "remediation": (
                    "Assign a unique prefix or merge endpoint ownership before deployment."
                ),
            }

            ROUTER_DIAGNOSTICS["duplicate_prefixes"].append(conflict)
            ROUTER_DIAGNOSTICS["conflicts"].append(conflict)

            ROUTER_DIAGNOSTICS["severity_counts"]["critical"] += 1

            logger.error(
                "ROUTER_CONFLICT %s",
                conflict,
            )

            if FAIL_FAST_ON_ROUTER_CONFLICTS:
                raise RuntimeError(
                    f"Critical router conflict detected for prefix {prefix}"
                )
        else:
            registered_prefixes[prefix] = router_name

        logger.info(
            "Registering router: %s | prefix=%s | tags=%s",
            router_name,
            prefix,
            tags,
        )

        if not router_name.strip():
            ROUTER_DIAGNOSTICS["warnings"].append(
                "Router with empty name detected"
            )

        if tags is None or len(tags) == 0:
            ROUTER_DIAGNOSTICS["warnings"].append(
                f"{router_name} has no tags configured"
            )

        app.include_router(
            router_obj,
            prefix=prefix,
            tags=tags or [],
        )

        ROUTER_DIAGNOSTICS["registered_routers"].append({
            "name": router_name,
            "prefix": prefix,
            "tags": tags or [],
        })

    register(
        ml.router,
        prefix="/api/yield",
        tags=["ML Prediction"],
        router_name="ML Prediction",
    )

    register(
        governance.router,
        prefix="/api/ml-governance",
        tags=["ML Governance"],
        router_name="ML Governance",
    )

    register(
        finance.router,
        prefix="/api/farm-finance",
        tags=["Finance"],
        router_name="Finance",
    )

    register(
        finance.router,
        prefix="/api/finance",
        tags=["Finance Legacy"],
        router_name="Finance Legacy",
    )

    register(
        quality.router,
        prefix="/api/crop-quality",
        tags=["Quality"],
        router_name="Quality",
    )

    register(
        blockchain.router,
        prefix="/api/supply-chain",
        tags=["Blockchain"],
        router_name="Blockchain",
    )

    register(
        reports.router,
        prefix="/api/admin",
        tags=["Reports"],
        router_name="Reports",
    )

    register(
        marketplace.router,
        prefix="/api/marketplace",
        tags=["Marketplace"],
        router_name="Marketplace",
    )

    register(
        knowledge.router,
        prefix="/api/knowledge",
        tags=["Knowledge"],
        router_name="Knowledge",
    )

    register(
        community.router,
        prefix="/api/community",
        tags=["Community"],
        router_name="Community",
    )

    register(
        voice_assistant_router.router if voice_assistant_router else None,
        prefix="/api/voice",
        tags=["Voice Assistant"],
        router_name="Voice Assistant",
    )

    register(
        referrals.router,
        prefix="/api/referrals",
        tags=["Referrals"],
        router_name="Referrals",
    )

    register(
        platform.router,
        prefix="/api",
        tags=["Platform"],
        router_name="Platform",
    )

    register(
        advisory.router,
        prefix="/api",
        tags=["Advisory"],
        router_name="Advisory",
    )

    register(
        alerts.router,
        prefix="/api/notifications",
        tags=["Alerts"],
        router_name="Alerts",
    )

    register(
        flags_router,
        tags=["Feature Flags"],
        router_name="Feature Flags",
    )

    register(
        lms.router,
        prefix="/api",
        tags=["LMS"],
        router_name="LMS",
    )

    expected_routers = {
        "ML Prediction",
        "ML Governance",
        "Finance",
        "Finance Legacy",
        "Quality",
        "Blockchain",
        "Reports",
        "Marketplace",
        "Knowledge",
        "Community",
        "Referrals",
        "Platform",
        "Advisory",
        "Alerts",
        "Feature Flags",
        "LMS",
    }

    registered_names = {
        item["name"]
        for item in ROUTER_DIAGNOSTICS["registered_routers"]
    }

    missing = expected_routers - registered_names

    ROUTER_DIAGNOSTICS["missing_routers"] = sorted(
        list(missing)
    )

    for router_name in missing:

        ROUTER_DIAGNOSTICS["conflicts"].append({
            "severity": "warning",
            "type": "missing_router",
            "router": router_name,
            "remediation": (
                "Verify dependency initialization and registration order."
            ),
        })

        ROUTER_DIAGNOSTICS["severity_counts"]["warning"] += 1

    if ROUTER_DIAGNOSTICS["severity_counts"]["critical"] > 0:
        ROUTER_DIAGNOSTICS["health_status"] = "critical"

    elif ROUTER_DIAGNOSTICS["severity_counts"]["warning"] > 0:
        ROUTER_DIAGNOSTICS["health_status"] = "degraded"

    else:
        ROUTER_DIAGNOSTICS["health_status"] = "healthy"

    ROUTER_DIAGNOSTICS["startup_summary"] = {
        "registered_router_count": len(
            ROUTER_DIAGNOSTICS["registered_routers"]
        ),
        "missing_router_count": len(
            ROUTER_DIAGNOSTICS["missing_routers"]
        ),
        "conflict_count": len(
            ROUTER_DIAGNOSTICS["conflicts"]
        ),
        "health_status": ROUTER_DIAGNOSTICS["health_status"],
    }

    @app.get("/system/router-diagnostics")
    async def router_diagnostics():
        return {
            "success": True,
            "validation_passed":
            ROUTER_DIAGNOSTICS["health_status"] == "healthy",
            **ROUTER_DIAGNOSTICS,
        }
    
    logger.info(
        "Router validation summary | "
        "registered=%d | duplicates=%d | "
        "missing=%d | warnings=%d",
        len(ROUTER_DIAGNOSTICS["registered_routers"]),
        len(ROUTER_DIAGNOSTICS["duplicate_prefixes"]),
        len(ROUTER_DIAGNOSTICS["missing_routers"]),
        len(ROUTER_DIAGNOSTICS["warnings"]),
    )

    logger.info(
        "Router registration completed successfully. Registered prefixes: %d",
        len(registered_prefixes),
    )