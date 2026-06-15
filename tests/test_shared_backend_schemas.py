from backend.routers import alerts, knowledge, platform
from backend.schemas import AlertTriggerRequest, RAGQuery


def test_alert_trigger_request_is_shared_schema():
    assert alerts.AlertTriggerRequest is AlertTriggerRequest
    assert platform.AlertTriggerRequest is AlertTriggerRequest


def test_rag_query_is_shared_schema():
    assert knowledge.RAGQuery is RAGQuery
    assert platform.RAGQuery is RAGQuery
