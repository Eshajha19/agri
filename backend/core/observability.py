import os

from fastapi import HTTPException, Request, Response

OBSERVABILITY_STATUS = {
    "tracing": False,
    "prometheus": False,
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

        logger.info(
            "Tracing initialized successfully (service=%s)",
            service_name,
        )

    except Exception as exc:
        logger.warning(
            "Tracing setup skipped. Falling back to application logging. Error: %s",
            exc,
        )

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app)

        OBSERVABILITY_STATUS["prometheus"] = True

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
                "otlp_endpoint_configured": bool(
                    os.environ.get(
                        "OTEL_EXPORTER_OTLP_ENDPOINT"
                    )
                ),
            }

    except Exception as exc:
        logger.warning(
            "Prometheus setup skipped. Metrics endpoint unavailable. Error: %s",
            exc,
        )