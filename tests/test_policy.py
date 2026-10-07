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