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
    app.include_router(
        ml.router,
        prefix="/api/yield",
        tags=["ML Prediction"],
    )

    app.include_router(
        governance.router,
        prefix="/api/ml-governance",
        tags=["ML Governance"],
    )

    app.include_router(
        finance.router,
        prefix="/api/farm-finance",
        tags=["Finance"],
    )

    app.include_router(
        finance.router,
        prefix="/api/finance",
        tags=["Finance Legacy"],
    )

    app.include_router(
        quality.router,
        prefix="/api/crop-quality",
        tags=["Quality"],
    )

    app.include_router(
        blockchain.router,
        prefix="/api/supply-chain",
        tags=["Blockchain"],
    )

    app.include_router(
        reports.router,
        prefix="/api/admin",
        tags=["Reports"],
    )

    app.include_router(
        marketplace.router,
        prefix="/api/marketplace",
        tags=["Marketplace"],
    )

    app.include_router(
        knowledge.router,
        prefix="/api/knowledge",
        tags=["Knowledge"],
    )

    app.include_router(
        community.router,
        prefix="/api/community",
        tags=["Community"],
    )

    if voice_assistant_router is not None:
        app.include_router(
            voice_assistant_router.router,
            prefix="/api/voice",
            tags=["Voice Assistant"],
        )

    app.include_router(
        referrals.router,
        prefix="/api/referrals",
        tags=["Referrals"],
    )

    app.include_router(
        platform.router,
        prefix="/api",
        tags=["Platform"],
    )

    app.include_router(
        advisory.router,
        prefix="/api",
        tags=["Advisory"],
    )

    app.include_router(
        alerts.router,
        prefix="/api/notifications",
        tags=["Alerts"],
    )

    app.include_router(
        flags_router,
        tags=["Feature Flags"],
    )

    app.include_router(
        lms.router,
        prefix="/api",
        tags=["LMS"],
    )