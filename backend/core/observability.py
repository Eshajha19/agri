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