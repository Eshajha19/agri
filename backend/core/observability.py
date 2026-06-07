import os

from fastapi import HTTPException, Request, Response


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

        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=otlp_endpoint)
                )
            )

        else:
            provider.add_span_processor(
                SimpleSpanProcessor(
                    ConsoleSpanExporter()
                )
            )

        FastAPIInstrumentor().instrument_app(app)

    except Exception as exc:
        logger.warning("Tracing setup skipped: %s", exc)

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app)

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

    except Exception as exc:
        logger.warning("Prometheus setup skipped: %s", exc)