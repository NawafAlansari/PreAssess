from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from smc_agents.report_agent import SeattleReportAgent


@pytest.fixture()
def client(retriever, monkeypatch):
    monkeypatch.setattr(api_main, "_retriever", retriever)
    monkeypatch.setattr(api_main, "_agent", None)
    monkeypatch.setattr(api_main, "_report_hits", __import__("collections").defaultdict(__import__("collections").deque))
    return TestClient(api_main.app)


@pytest.fixture()
def client_with_agent(client, retriever, monkeypatch):
    agent = SeattleReportAgent(retriever=retriever, api_key="test-key")
    reply = "Setbacks per SMC 23.44.010. Permits per SMC 22.801.050. See SMC 99.99.999."
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = completion
    monkeypatch.setattr(api_main, "_agent", agent)
    return client


def test_health_reports_corpus(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["corpus"]["chunks"] == 4
    assert body["corpus"]["titles"] == [22, 23]


def test_health_includes_built_at_when_meta_present(client, tmp_path, monkeypatch):
    import json as _json

    meta = tmp_path / "corpus_meta.json"
    meta.write_text(_json.dumps({"built_at": "2026-07-08", "chunks": 4}))
    monkeypatch.setattr(api_main, "CORPUS_META_PATH", meta)
    body = client.get("/api/health").json()
    assert body["corpus"]["built_at"] == "2026-07-08"


def test_health_built_at_null_when_meta_absent(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "CORPUS_META_PATH", tmp_path / "nope.json")
    body = client.get("/api/health").json()
    assert body["corpus"]["built_at"] is None


def test_stats_counts_by_title_and_type(client):
    body = client.get("/api/stats").json()
    assert body["chunks_by_title"] == {"22": 1, "23": 3}
    assert body["chunks_by_type"]["section"] == 3


def test_search_returns_trimmed_hits(client):
    body = client.get("/api/search", params={"q": "setback dwelling"}).json()
    assert body["results"][0]["section_citation"] == "23.44.010"
    assert "text" in body["results"][0]


def test_search_rejects_empty_query(client):
    assert client.get("/api/search", params={"q": "  "}).status_code == 422


def test_search_title_filter(client):
    body = client.get("/api/search", params={"q": "permit", "title": 22}).json()
    assert all(r["title_number"] == 22 for r in body["results"])


def test_report_requires_input(client_with_agent):
    resp = client_with_agent.post("/api/report", json={})
    assert resp.status_code == 422


def test_report_returns_audit_and_evidence(client_with_agent):
    resp = client_with_agent.post(
        "/api/report",
        json={
            "address_profile": {"address": "1 Test Ave", "zoning": "SF 5000"},
            "project_description": "Build a dwelling with new setbacks",
            "questions": ["What permit submittal documents do I need?"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    statuses = {v["citation"]: v["status"] for v in body["citation_audit"]}
    assert statuses["23.44.010"] == "grounded"
    assert statuses["99.99.999"] == "unknown"
    assert 0.0 < body["grounded_ratio"] < 1.0
    assert "project" in body["evidence"]
    assert body["evidence"]["project"][0]["chunk_id"]


def test_report_503_without_server_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    resp = client.post("/api/report", json={"project_description": "setback"})
    assert resp.status_code == 503
    assert "LLM_API_KEY" in resp.json()["detail"]


def test_report_rate_limit(client_with_agent, monkeypatch):
    monkeypatch.setattr(api_main, "REPORT_RATE_LIMIT", 2)
    payload = {"project_description": "setback dwelling"}
    assert client_with_agent.post("/api/report", json=payload).status_code == 200
    assert client_with_agent.post("/api/report", json=payload).status_code == 200
    assert client_with_agent.post("/api/report", json=payload).status_code == 429


def test_citation_lookup_exact_and_subsection(client):
    body = client.get("/api/citation/23.44.010").json()
    assert body["results"][0]["citation"] == "SMC 23.44.010"
    deep = client.get("/api/citation/23.44.010.C.2").json()
    assert deep["results"][0]["citation"] == "SMC 23.44.010"
    none = client.get("/api/citation/99.99.999").json()
    assert none["results"] == []


def test_followup_endpoint(client_with_agent, monkeypatch):
    # reuse the mocked agent; its scripted reply cites 23.44.010 and 22.801.050
    resp = client_with_agent.post(
        "/api/followup",
        json={
            "question": "What about the setback?",
            "history": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and body["citation_audit"]
    assert "followup" in body["evidence"]


def test_followup_rejects_empty_question(client_with_agent):
    resp = client_with_agent.post("/api/followup", json={"question": "  "})
    assert resp.status_code == 422


def test_feedback_appends_jsonl(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    resp = client.post(
        "/api/feedback",
        json={"vote": "up", "comment": "good", "grounded_ratio": 0.8},
    )
    assert resp.status_code == 200
    import json as _json

    lines = (tmp_path / "feedback.jsonl").read_text().strip().splitlines()
    rec = _json.loads(lines[0])
    assert rec["vote"] == "up" and rec["grounded_ratio"] == 0.8 and rec["ts"]


def test_feedback_rejects_bad_vote(client):
    assert client.post("/api/feedback", json={"vote": "meh"}).status_code == 422
