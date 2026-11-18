"""Opt-in real-model tests for ToxicityScanner (unitary/toxic-bert).

Skipped unless the ml extra is installed: pip install -e '.[ml]'
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


def test_real_model_blocks_toxic_plain_text(scanner):
    result = scanner.check(TOXIC)
    assert result.error is False
    assert result.passed is False


def test_real_model_blocks_toxic_inside_json(engine):
    d = engine.validate(json.dumps({"customer": TOXIC}))
    assert d.action == Action.BLOCK
    assert "toxic content detected" in d.reason


def test_real_model_passes_clean_plain_text(scanner):
    result = scanner.check(CLEAN)
    assert result.error is False
    assert result.passed is True


def test_real_model_allows_clean_inside_json(engine):
    d = engine.validate(json.dumps({"customer": CLEAN}))
    assert d.action == Action.ALLOW
