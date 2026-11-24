from guardrails.results import GuardrailResult
from guardrails.aggregator import AggregatedResults
from guardrails.policy import PolicyEngine, Action

engine = PolicyEngine()


def _r(name, passed=True, error=False, transformed=None, reason=None):
    return GuardrailResult(guardrail=name, passed=passed, error=error, transformed_output=transformed, reason=reason)


def _decide(*results, candidate="original"):
    return engine.decide(AggregatedResults(results=list(results)), candidate)


def test_all_clean_allows():
    d = _decide(_r("schema"), _r("pii"))
    assert d.action == Action.ALLOW
    assert d.output == "original"


def test_schema_fail_blocks():
    d = _decide(_r("schema", passed=False, reason="bad json"))
    assert d.action == Action.BLOCK
    assert "bad json" in d.reason


def test_pii_proposal_modifies():
    d = _decide(_r("schema"), _r("pii", transformed='{"c": "[REDACTED_EMAIL]"}'))
    assert d.action == Action.MODIFY
    assert "[REDACTED_EMAIL]" in d.output


def test_error_fails_closed():
    d = _decide(_r("schema"), _r("pii", passed=False, error=True))
    assert d.action == Action.BLOCK
    assert "internal error" in d.reason


def test_error_beats_modify():
    # ek check crash + doosre ne transform propose kiya -> phir bhi block
    d = _decide(_r("schema", passed=False, error=True),
                _r("pii", transformed="clean"))
    assert d.action == Action.BLOCK


def test_original_never_modified_without_proposal():
    d = _decide(_r("schema"), _r("pii", transformed=None))
    assert d.action == Action.ALLOW
    assert d.output == "original"

def test_schema_fail_blocks_even_when_not_first():
    d = _decide(_r("pii"), _r("schema", passed=False, reason="bad json"))
    assert d.action == Action.BLOCK


def test_toxicity_fail_blocks_with_incident_id():
    d = _decide(_r("schema"), _r("pii"), _r("toxicity", passed=False, reason="score 0.95"))
    assert d.action == Action.BLOCK
    assert "toxic content detected" in d.reason
    assert "incident_id=" in d.reason


def test_toxicity_beats_pii_modify():
    # Even if PII produced a transformed output, toxicity BLOCK takes precedence
    d = _decide(_r("schema"), _r("pii", transformed="clean"), _r("toxicity", passed=False, reason="score 0.95"))
    assert d.action == Action.BLOCK
    assert d.output is None


def test_groundedness_fail_blocks_when_budget_exhausted():
    d = engine.decide(
        AggregatedResults(results=[_r("schema"), _r("groundedness", passed=False, reason="hallucinated claim")]),
        candidate="original",
        retries_left=0,
    )
    assert d.action == Action.BLOCK
    assert "cannot verify against source" in d.reason


def test_groundedness_fail_reasks_when_retries_available():
    d = engine.decide(
        AggregatedResults(results=[_r("schema"), _r("groundedness", passed=False, reason="hallucinated claim")]),
        candidate="original",
        retries_left=1,
    )
    assert d.action == Action.REASK
    assert "groundedness check failed" in d.reason


def test_toxicity_beats_groundedness_fail():
    # Toxicity BLOCK takes precedence over groundedness REASK
    d = engine.decide(
        AggregatedResults(results=[
            _r("schema"),
            _r("toxicity", passed=False, reason="threat"),
            _r("groundedness", passed=False, reason="unsupported"),
        ]),
        candidate="original",
        retries_left=2,
    )
    assert d.action == Action.BLOCK
    assert "toxic content detected" in d.reason


# --- Decision.metadata whitelist (point 9) -------------------------------------------

def test_modify_decision_never_contains_raw_email():
    from pydantic import BaseModel
    from guardrails.config import PolicyConfig
    from guardrails.engine import GuardrailEngine

    class Contact(BaseModel):
        email: str

    eng = GuardrailEngine(Contact, config=PolicyConfig(enabled=["schema", "pii"]))
    d = eng.validate('{"email": "ravi@example.com"}')

    assert d.action == Action.MODIFY
    assert "ravi@example.com" not in d.model_dump_json()
    assert d.metadata["checks"]["pii"]["counts"]["found"] == {"email": 1}


def test_decision_metadata_is_whitelisted_summary_only():
    meta = {
        "parsed": {"email": "ravi@example.com"},
        "found": {"email": 1},
        "ungrounded_indices": [0],
        "note": "raw text here",
        "score": 0.4,
        "missing_source": False,
    }
    r = GuardrailResult(guardrail="pii", passed=True, score=0.4, metadata=meta)
    d = engine.decide(AggregatedResults(results=[_r("schema"), r]), candidate="original")

    pii = d.metadata["checks"]["pii"]
    assert pii == {
        "passed": True,
        "error": False,
        "score": 0.4,
        "counts": {"found": {"email": 1}, "score": 0.4, "missing_source": False},
    }
    assert "ravi@example.com" not in d.model_dump_json()
    assert "raw text here" not in d.model_dump_json()


def test_toxicity_block_metadata_has_incident_id_only():
    d = _decide(_r("schema"), _r("toxicity", passed=False, reason="threat"))
    assert d.action == Action.BLOCK
    assert d.metadata["incident_id"] in d.reason
    assert set(d.metadata) == {"checks", "incident_id"}


# --- missing source (point 6) -----------------------------------------------------------

def test_missing_source_blocks_without_reask():
    g = GuardrailResult(guardrail="groundedness", passed=False, reason="no source provided",
                        metadata={"missing_source": True})
    d = engine.decide(AggregatedResults(results=[_r("schema"), g]), candidate="x", retries_left=2)
    assert d.action == Action.BLOCK
    assert "no source provided" in d.reason


def test_engine_groundedness_enabled_without_source_fails_closed():
    from pydantic import BaseModel
    from guardrails.config import PolicyConfig
    from guardrails.engine import GuardrailEngine

    class Answer(BaseModel):
        answer: str

    eng = GuardrailEngine(Answer, config=PolicyConfig(enabled=["schema", "groundedness"]))
    d = eng.validate('{"answer": "The Eiffel Tower was completed in 1898."}')
    assert d.action == Action.BLOCK
    assert "no source provided" in d.reason

    eng_skip = GuardrailEngine(
        Answer,
        config=PolicyConfig(enabled=["schema", "groundedness"], groundedness_on_missing_source="skip"),
    )
    assert eng_skip.validate('{"answer": "The Eiffel Tower was completed in 1898."}').action == Action.ALLOW
