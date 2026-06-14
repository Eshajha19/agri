import os

from fastapi import HTTPException, Request, Response

OBSERVABILITY_STATUS = {
    "tracing": False,
    "prometheus": False,
    "fallback_logging": False,
    "last_error": None,
    "active_components": [],
    "degraded_components": [],
}


def _calculate_observability_mode():
    tracing = OBSERVABILITY_STATUS["tracing"]
    prometheus = OBSERVABILITY_STATUS["prometheus"]
    fallback = OBSERVABILITY_STATUS["fallback_logging"]

    if tracing and prometheus and not fallback:
        return "fully_operational"

    if fallback and not tracing and not prometheus:
        return "fallback_only"

    return "degraded"

COMPONENT_WEIGHTS = {
    "distributed_tracing": 60,
    "prometheus_metrics": 40,
}


def calculate_health_score():
    score = 0

    if OBSERVABILITY_STATUS["tracing"]:
        score += COMPONENT_WEIGHTS["distributed_tracing"]

    if OBSERVABILITY_STATUS["prometheus"]:
        score += COMPONENT_WEIGHTS["prometheus_metrics"]

    return score


def calculate_degradation_severity():
    score = calculate_health_score()

    if score >= 90:
        return "minimal"

    if score >= 60:
        return "moderate"

    if score >= 30:
        return "high"

    return "critical"


def generate_remediation_recommendations():
    recommendations = []

    if not OBSERVABILITY_STATUS["tracing"]:
        recommendations.append(
            "Verify OpenTelemetry exporter configuration and tracing initialization."
        )

    if not OBSERVABILITY_STATUS["prometheus"]:
        recommendations.append(
            "Verify Prometheus instrumentation and metrics endpoint configuration."
        )

    if OBSERVABILITY_STATUS["fallback_logging"]:
        recommendations.append(
            "Investigate degraded observability components and restore telemetry."
        )

    return recommendations


def build_component_diagnostics():
    diagnostics = []

    diagnostics.append({
        "component": "distributed_tracing",
        "status": (
            "healthy"
            if OBSERVABILITY_STATUS["tracing"]
            else "degraded"
        ),
        "weight": COMPONENT_WEIGHTS["distributed_tracing"],
    })

    diagnostics.append({
        "component": "prometheus_metrics",
        "status": (
            "healthy"
            if OBSERVABILITY_STATUS["prometheus"]
            else "degraded"
        ),
        "weight": COMPONENT_WEIGHTS["prometheus_metrics"],
    })

    return diagnostics

def build_recovery_progress():
    return {
        "healthy_components":
            len(OBSERVABILITY_STATUS["active_components"]),
        "degraded_components":
            len(OBSERVABILITY_STATUS["degraded_components"]),
        "recovery_percent":
            calculate_health_score(),
    }

def setup_observability(app, verify_role, logger):
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        service_name = os.environ.get(
            "OTEL_SERVICE_NAME",
            "fasal-saathi-backend"
        )

        resource = Resource.create({
            "service.name": service_name
        })

        provider = TracerProvider(resource=resource)

        trace.set_tracer_provider(provider)

        otlp_endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )

        if otlp_endpoint and otlp_endpoint.startswith(
            ("http://", "https://")
        ):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=otlp_endpoint
                    )
                )
            )

        else:
            provider.add_span_processor(
                SimpleSpanProcessor(
                    ConsoleSpanExporter()
                )
            )

        FastAPIInstrumentor().instrument_app(app)

        OBSERVABILITY_STATUS["tracing"] = True
        if "distributed_tracing" not in OBSERVABILITY_STATUS["active_components"]:
            OBSERVABILITY_STATUS["active_components"].append(
                "distributed_tracing"
            )

        logger.info(
            "Observability validation passed for tracing subsystem"
        )

        logger.info(
            "Tracing initialized successfully (service=%s)",
            service_name,
        )

    except Exception as exc:
        OBSERVABILITY_STATUS["fallback_logging"] = True
        if "distributed_tracing" not in OBSERVABILITY_STATUS["degraded_components"]:
            OBSERVABILITY_STATUS["degraded_components"].append(
                "distributed_tracing"
            )
        OBSERVABILITY_STATUS["last_error"] = str(exc)

        logger.warning(
            "Tracing setup skipped. Falling back to application logging. Error: %s",
            exc,
        )

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app)

        OBSERVABILITY_STATUS["prometheus"] = True
        if "prometheus_metrics" not in OBSERVABILITY_STATUS["active_components"]:
            OBSERVABILITY_STATUS["active_components"].append(
                "prometheus_metrics"
            )

        logger.info(
            "Observability validation passed for metrics subsystem"
        )

        logger.info(
            "Prometheus instrumentation initialized successfully"
        )

        @app.get("/metrics")
        async def metrics(request: Request):
            if verify_role is None:
                raise HTTPException(
                    status_code=500,
                    detail="Auth service not initialized"
                )

            await verify_role(
                request,
                required_roles=["admin"]
            )

            from prometheus_client import (
                generate_latest,
                CONTENT_TYPE_LATEST,
            )

            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

        @app.get("/observability/status")
        async def observability_status(request: Request):
            if verify_role is None:
                raise HTTPException(
                    status_code=500,
                    detail="Auth service not initialized"
                )

            await verify_role(
                request,
                required_roles=["admin"]
            )

            health_score = calculate_health_score()
            severity = calculate_degradation_severity()
            recommendations = generate_remediation_recommendations()

            return {
                "success": True,
                "tracing_enabled": OBSERVABILITY_STATUS["tracing"],
                "prometheus_enabled": OBSERVABILITY_STATUS["prometheus"],
                "fallback_logging": OBSERVABILITY_STATUS["fallback_logging"],
                "last_error": OBSERVABILITY_STATUS["last_error"],
                "otlp_endpoint_configured": bool(
                    os.environ.get(
                        "OTEL_EXPORTER_OTLP_ENDPOINT"
                    )
                ),
                "observability_mode": _calculate_observability_mode(),

                "health_score": health_score,
                "health_score_percent": health_score,
                "degradation_severity": severity,
                "remediation_recommendations": recommendations,

                "telemetry_available": (
                    OBSERVABILITY_STATUS["tracing"]
                    or OBSERVABILITY_STATUS["prometheus"]
                ),
                "telemetry_coverage_percent": round(
                    (
                        int(OBSERVABILITY_STATUS["tracing"])
                        + int(OBSERVABILITY_STATUS["prometheus"])
                    ) / 2 * 100,
                    0,
                ),
                "active_components":
                    OBSERVABILITY_STATUS["active_components"],
                "degraded_components":
                    OBSERVABILITY_STATUS["degraded_components"],
                "status_reason": {
                    "fully_operational": (
                        "Tracing and metrics are fully operational."
                    ),
                    "degraded": (
                        "One or more observability components are unavailable."
                    ),
                    "fallback_only": (
                        "Application is running using fallback logging only."
                    ),
                }.get(_calculate_observability_mode()),
            }

        @app.get("/observability/diagnostics")
        async def observability_diagnostics(request: Request):
            if verify_role is None:
                raise HTTPException(
                    status_code=500,
                    detail="Auth service not initialized"
                )

            await verify_role(
                request,
                required_roles=["admin"]
            )

            health_score = calculate_health_score()
            severity = calculate_degradation_severity()

            return {
                "success": True,
                "status": OBSERVABILITY_STATUS,
                "service_name": os.environ.get(
                    "OTEL_SERVICE_NAME",
                    "fasal-saathi-backend"
                ),
                "otlp_configured": bool(
                    os.environ.get(
                        "OTEL_EXPORTER_OTLP_ENDPOINT"
                    )
                ),
                "observability_mode": _calculate_observability_mode(),

                "health_score": health_score,
                "degradation_severity": severity,
                "component_diagnostics": build_component_diagnostics(),
                "remediation_recommendations":
                    generate_remediation_recommendations(),
                "recovery_progress":
                    build_recovery_progress(),

                "component_summary": {
                    "active":
                        OBSERVABILITY_STATUS["active_components"],
                    "degraded":
                        OBSERVABILITY_STATUS["degraded_components"],
                },
            }

    except Exception as exc:
        OBSERVABILITY_STATUS["last_error"] = str(exc)
        if "prometheus_metrics" not in OBSERVABILITY_STATUS["degraded_components"]:
            OBSERVABILITY_STATUS["degraded_components"].append(
                "prometheus_metrics"
            )
        logger.warning(
            "Prometheus setup skipped. Metrics endpoint unavailable. Error: %s",
            exc,
        )