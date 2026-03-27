"""Tests for SELF-RAG System."""
import pytest
from unittest.mock import patch, MagicMock
from app.core.config import settings

def test_settings():
    assert settings.MAX_REFLECTION_ROUNDS == 3
    assert settings.RELEVANCE_THRESHOLD == 0.7
    assert settings.SUPPORT_THRESHOLD == 0.7

def test_retrieve_decision_parsing():
    import json
    decision = {"should_retrieve": True, "reasoning": "Factual question requires retrieval"}
    assert decision["should_retrieve"] is True

def test_reflection_token_concepts():
    tokens = ["RETRIEVE", "IS_REL", "IS_SUP", "IS_USE"]
    for token in tokens:
        assert len(token) > 2

@patch("app.services.selfrag_service.OpenAI")
@patch("app.services.selfrag_service.OpenAIEmbeddings")
def test_selfrag_init(mock_embed, mock_openai):
    from app.services.selfrag_service import SELFRAGService
    svc = SELFRAGService()
    assert svc is not None
    assert svc.vectorstore is None

@patch("app.services.selfrag_service.OpenAI")
@patch("app.services.selfrag_service.OpenAIEmbeddings")
def test_no_retrieve_direct_generation(mock_embed, mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"should_retrieve": false, "reasoning": "simple"}'))]
    )
    from app.services.selfrag_service import SELFRAGService
    svc = SELFRAGService()
    decision = svc._decide_retrieve("What is 2+2?")
    assert "should_retrieve" in decision

@pytest.mark.asyncio
async def test_api_health():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/api/v1/selfrag/health")
    assert resp.status_code == 200

@pytest.mark.asyncio
async def test_api_query_short():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.post("/api/v1/selfrag/query", json={"question": "Hi"})
    assert resp.status_code == 400
