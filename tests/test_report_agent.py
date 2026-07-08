from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from smc_agents.report_agent import EvidenceRequest, SeattleReportAgent


def make_agent(retriever, reply: str) -> SeattleReportAgent:
    agent = SeattleReportAgent(retriever=retriever, api_key="test-key")
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = completion
    return agent


def test_requires_api_key(retriever, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        SeattleReportAgent(retriever=retriever)


def test_model_env_override(retriever, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some-newer-model")
    import importlib

    from smc_agents import report_agent as module

    importlib.reload(module)
    agent = module.SeattleReportAgent(retriever=retriever, api_key="k")
    assert agent.model == "some-newer-model"
    monkeypatch.delenv("GROQ_MODEL")
    importlib.reload(module)


def test_gather_evidence_labels(retriever):
    agent = make_agent(retriever, "ok")
    evidence = agent.gather_evidence(
        [
            EvidenceRequest(label="zoning", query="setback dwelling", title_number=23),
            EvidenceRequest(label="permits", query="permit", title_number=22),
        ]
    )
    assert set(evidence) == {"zoning", "permits"}
    assert evidence["zoning"][0].chunk_id == "c-2344-010"
    assert evidence["permits"][0].chunk_id == "c-22801-050"


def test_prompt_includes_evidence_and_facts(retriever):
    agent = make_agent(retriever, "ok")
    evidence = agent.gather_evidence(
        [EvidenceRequest(label="zoning", query="setback", title_number=23)]
    )
    prompt = agent.build_prompt(
        address_profile={"address": "1 Test Ave", "zoning": "SF 5000"},
        user_inputs={"project": "Add an ADU"},
        evidence=evidence,
    )
    assert "1 Test Ave" in prompt
    assert "Add an ADU" in prompt
    assert "ZONING:" in prompt
    assert "SMC 23.44.010" in prompt


def test_prompt_handles_empty_evidence(retriever):
    agent = make_agent(retriever, "ok")
    prompt = agent.build_prompt(
        address_profile={}, user_inputs={}, evidence={"zoning": []}
    )
    assert "No evidence found." in prompt


def test_generate_report_bundle_includes_audit(retriever):
    reply = "Setbacks per SMC 23.44.010 apply. Also see SMC 99.99.999."
    agent = make_agent(retriever, reply)
    bundle = agent.generate_report(
        address_profile={"address": "1 Test Ave"},
        user_inputs={"project": "ADU"},
        evidence_requests=[
            EvidenceRequest(label="zoning", query="setback", title_number=23)
        ],
    )
    assert bundle["report"] == reply
    statuses = {v["citation"]: v["status"] for v in bundle["citation_audit"]}
    assert statuses["23.44.010"] == "grounded"
    assert statuses["99.99.999"] == "unknown"
    assert bundle["grounded_ratio"] == pytest.approx(0.5)
    assert "zoning" in bundle["evidence"]


def test_answer_followup_grounded_and_audited(retriever):
    reply = "The code limits ADU size per SMC 23.44.010. Also SMC 99.99.999."
    agent = make_agent(retriever, reply)
    bundle = agent.answer_followup(
        question="How big can the setback dwelling be?",
        history=[
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer citing SMC 23.44.010."},
        ],
        address_profile={"address": "1 Test Ave"},
    )
    statuses = {v["citation"]: v["status"] for v in bundle["citation_audit"]}
    assert statuses["23.44.010"] == "grounded"
    assert statuses["99.99.999"] == "unknown"
    # history + property context reached the model
    messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert any(m["role"] == "assistant" for m in messages)
    assert "1 Test Ave" in messages[-1]["content"]
    assert "Municipal code evidence" in messages[-1]["content"]
