"""Opt-in real-model tests for GroundednessChecker (MiniLM bi-encoder + DeBERTa NLI).

Marked slow: skipped unless pytest is run with --run-slow.
Also skipped unless the ml extra is installed: pip install -e '.[ml]'
First run downloads the models from HuggingFace.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")

from guardrails.checks.groundedness import GroundednessChecker

pytestmark = pytest.mark.slow

SOURCE = "The Eiffel Tower was completed in 1889."


@pytest.fixture(scope="module")
def checker():
    return GroundednessChecker()


def test_supported_claim_passes(checker):
    res = checker.check("The Eiffel Tower was completed in 1889.", {"source": SOURCE})
    assert res.error is False
    assert res.passed is True


def test_negated_claim_fails(checker):
    res = checker.check("The Eiffel Tower was not completed in 1889.", {"source": SOURCE})
    assert res.error is False
    assert res.passed is False
    # high cosine, so this was decided by NLI, not rejected by cosine
    assert res.metadata["details"][0]["action"] == "evaluated_by_nli"


def test_wrong_year_claim_fails(checker):
    res = checker.check("The Eiffel Tower was completed in 1898.", {"source": SOURCE})
    assert res.error is False
    assert res.passed is False
    assert res.metadata["details"][0]["action"] == "evaluated_by_nli"
