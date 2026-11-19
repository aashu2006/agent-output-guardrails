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


# --- multi-label (unitary/toxic-bert output shape) ---------------------------

def _toxic_bert(toxic: float, **others: float):
    """Fake classifier returning the toxic-bert top_k=None shape."""
    scores = {"toxic": toxic, "severe_toxic": 0.01, "obscene": 0.01,
              "threat": 0.01, "insult": 0.01, "identity_hate": 0.01, **others}
    out = [{"label": k, "score": v} for k, v in scores.items()]
    out.sort(key=lambda d: d["score"], reverse=True)
    return lambda text: out


def test_multilabel_toxic_high_fails():
    scanner = ToxicityScanner(threshold=0.8, classifier=_toxic_bert(0.97, insult=0.9))
    result = scanner.check("You stupid idiot")
    assert result.passed is False
    assert result.score == 0.97


def test_multilabel_toxic_low_passes():
    scanner = ToxicityScanner(threshold=0.8, classifier=_toxic_bert(0.02))
    result = scanner.check("Thanks, have a great day")
    assert result.passed is True
    assert result.score == 0.02


def test_non_toxic_label_never_matches_toxic():
    # "non-toxic" contains "toxic" as a substring; exact matching must not confuse them
    out = [{"label": "non-toxic", "score": 0.97}, {"label": "toxic", "score": 0.03}]
    scanner = ToxicityScanner(threshold=0.8, classifier=lambda text: out)
    result = scanner.check("Have a nice day")
    assert result.passed is True
    assert result.score == 0.03

    # safe label alone -> inverted, still low
    assert round(_extract_toxicity_score([{"label": "NON-TOXIC", "score": 0.97}]), 2) == 0.03


def test_parsed_context_scores_string_values_only():
    seen = []

    def recording_classifier(texts):
        seen.append(texts)
        return [[{"label": "toxic", "score": 0.01}] for _ in texts]

    scanner = ToxicityScanner(classifier=recording_classifier)
    raw = '{"customer": "hello there", "amount": 5, "notes": ["a", {"inner": "b"}], "paid": true}'
    parsed = {"customer": "hello there", "amount": 5, "notes": ["a", {"inner": "b"}], "paid": True}
    scanner.check(raw, context={"parsed": parsed})

    # one batch call, each string value scored separately, no keys/brackets/numbers
    assert seen == [["hello there", "a", "b"]]


def test_no_parsed_context_falls_back_to_raw_candidate():
    seen = []
    scanner = ToxicityScanner(classifier=lambda texts: seen.append(texts) or 0.0)
    scanner.check("plain text response", context={})
    assert seen == [["plain text response"]]


# --- truncation bypass / max over strings, chunks and labels -----------------

def _keyword_classifier(keyword: str = "idiot"):
    """Fake batch classifier: toxic-bert shape per text, toxic only if keyword present."""
    def classify(texts):
        return [
            [{"label": "toxic", "score": 0.97 if keyword in t else 0.01},
             {"label": "insult", "score": 0.01}]
            for t in texts
        ]
    return classify


POLITE_WORDS = ("thank you so much for your kind help today " * 75).strip()  # 600 words


def test_long_polite_field_then_toxic_field_fails():
    scanner = ToxicityScanner(threshold=0.8, classifier=_keyword_classifier())
    parsed = {"notes": POLITE_WORDS, "customer": "you stupid idiot pay me now"}
    result = scanner.check("<raw json unused>", context={"parsed": parsed})
    assert result.passed is False
    assert result.score == 0.97


def test_toxic_content_past_chunk_size_fails():
    seen = []
    classify = _keyword_classifier()

    def recording(texts):
        seen.append(texts)
        return classify(texts)

    scanner = ToxicityScanner(threshold=0.8, classifier=recording)
    result = scanner.check(POLITE_WORDS + " you stupid idiot pay me now")
    assert len(seen) == 1 and len(seen[0]) > 1  # one batch call, several chunks
    assert "idiot" not in seen[0][0]            # toxic part is past the first chunk
    assert result.passed is False


def test_max_over_toxic_labels_threat_beats_toxic():
    scanner = ToxicityScanner(threshold=0.8, classifier=_toxic_bert(0.75, threat=0.95))
    result = scanner.check("I know where you live")
    assert result.passed is False
    assert result.score == 0.95
