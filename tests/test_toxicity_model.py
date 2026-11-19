"""Opt-in real-model tests for ToxicityScanner (unitary/toxic-bert).

Marked slow: skipped unless pytest is run with --run-slow.
Also skipped unless the ml extra is installed: pip install -e '.[ml]'
First run downloads the model from HuggingFace.
"""
import json

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from pydantic import BaseModel

from guardrails.checks.toxicity import ToxicityScanner
from guardrails.config import PolicyConfig
from guardrails.engine import GuardrailEngine
from guardrails.policy import Action

TOXIC = "You stupid idiot pay me right now"
CLEAN = "Thanks, have a great day"

# ~600 polite words (602), ~650 tokens: longer than the model's 512-token window
LONG_POLITE = ("Thank you so much for your patience and kind help with this order today. " * 43).strip()
LONG_TOXIC_TAIL = LONG_POLITE + " You stupid idiot pay me right now!"


class Customer(BaseModel):
    customer: str


@pytest.fixture(scope="module")
def scanner():
    return ToxicityScanner()


@pytest.fixture(scope="module")
def engine(scanner):
    return GuardrailEngine(
        Customer,
        config=PolicyConfig(enabled=["schema", "pii", "toxicity"]),
        toxicity_scanner=scanner,
    )


@pytest.mark.slow
def test_real_model_blocks_toxic_plain_text(scanner):
    result = scanner.check(TOXIC)
    assert result.error is False
    assert result.passed is False


@pytest.mark.slow
def test_real_model_blocks_toxic_inside_json(engine):
    d = engine.validate(json.dumps({"customer": TOXIC}))
    assert d.action == Action.BLOCK
    assert "toxic content detected" in d.reason


@pytest.mark.slow
def test_real_model_passes_clean_plain_text(scanner):
    result = scanner.check(CLEAN)
    assert result.error is False
    assert result.passed is True


@pytest.mark.slow
def test_real_model_allows_clean_inside_json(engine):
    d = engine.validate(json.dumps({"customer": CLEAN}))
    assert d.action == Action.ALLOW


# --- long input: insult after ~600 polite words (truncation / dilution bypass) ---

@pytest.mark.slow
def test_real_model_blocks_toxic_tail_after_long_polite_plain_text(scanner):
    result = scanner.check(LONG_TOXIC_TAIL)
    assert result.error is False
    assert result.passed is False


@pytest.mark.slow
def test_real_model_blocks_toxic_tail_after_long_polite_inside_json(engine):
    d = engine.validate(json.dumps({"customer": LONG_TOXIC_TAIL}))
    assert d.action == Action.BLOCK
    assert "toxic content detected" in d.reason


@pytest.mark.slow
def test_real_model_allows_long_polite_text_alone(scanner, engine):
    result = scanner.check(LONG_POLITE)
    assert result.error is False
    assert result.passed is True
    assert engine.validate(json.dumps({"customer": LONG_POLITE})).action == Action.ALLOW
