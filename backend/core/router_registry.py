import logging

logger = logging.getLogger(__name__)

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
            return

        if prefix in registered_prefixes:
            logger.warning(
                "Duplicate router prefix detected: %s (%s and %s)",
                prefix,
                registered_prefixes[prefix],
                router_name,
            )
        else:
            registered_prefixes[prefix] = router_name

        logger.info(
            "Registering router: %s | prefix=%s | tags=%s",
            router_name,
            prefix,
            tags,
        )

        app.include_router(
            router_obj,
            prefix=prefix,
            tags=tags or [],
        )

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

    if voice_assistant_router is not None:
        register(
            voice_assistant_router.router,
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

    logger.info(
        "Router registration completed successfully. Registered prefixes: %d",
        len(registered_prefixes),
    )