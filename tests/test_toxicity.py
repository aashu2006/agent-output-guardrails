import pytest
from guardrails.checks.toxicity import ToxicityScanner, _extract_toxicity_score


def test_clean_text_passes():
    mock_classifier = lambda text: 0.05
    scanner = ToxicityScanner(threshold=0.8, classifier=mock_classifier)
    result = scanner.check("Have a wonderful and productive day!")
    assert result.passed is True
    assert result.score == 0.05
    assert result.reason is None
    assert result.metadata["threshold"] == 0.8


def test_toxic_text_fails():
    mock_classifier = lambda text: 0.92
    scanner = ToxicityScanner(threshold=0.8, classifier=mock_classifier)
    result = scanner.check("You are completely stupid and useless!")
    assert result.passed is False
    assert result.score == 0.92
    assert "exceeds threshold" in result.reason
    assert result.error is False


def test_empty_candidate_passes():
    mock_classifier = lambda text: 1.0  # Should never be called for empty
    scanner = ToxicityScanner(threshold=0.8, classifier=mock_classifier)
    result = scanner.check("   ")
    assert result.passed is True
    assert result.score == 0.0


def test_custom_threshold_respected():
    mock_classifier = lambda text: 0.65
    strict_scanner = ToxicityScanner(threshold=0.5, classifier=mock_classifier)
    lenient_scanner = ToxicityScanner(threshold=0.8, classifier=mock_classifier)

    assert strict_scanner.check("borderline rude text").passed is False
    assert lenient_scanner.check("borderline rude text").passed is True


def test_extract_toxicity_score_formats():
    # Direct number
    assert _extract_toxicity_score(0.42) == 0.42

    # HF pipeline label toxic
    hf_toxic = [{"label": "toxic", "score": 0.88}, {"label": "non-toxic", "score": 0.12}]
    assert _extract_toxicity_score(hf_toxic) == 0.88

    # HF pipeline nested list [[...]]
    assert _extract_toxicity_score([hf_toxic]) == 0.88

    # Binary LABEL_1 (toxic)
    assert _extract_toxicity_score([{"label": "LABEL_1", "score": 0.91}]) == 0.91

    # Binary non-toxic only (should invert: 1.0 - 0.9 = 0.1)
    assert round(_extract_toxicity_score([{"label": "non-toxic", "score": 0.9}]), 2) == 0.10

    # Empty or unrecognized
    assert _extract_toxicity_score([]) == 0.0


def test_crash_safety_returns_error_result():
    mock_classifier = lambda text: 1 / 0  # simulates internal crash
    scanner = ToxicityScanner(classifier=mock_classifier)
    result = scanner.check("trigger crash")
    assert result.passed is False
    assert result.error is True
    assert "ZeroDivisionError" in result.reason
