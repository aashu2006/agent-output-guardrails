import json

import pytest
from guardrails.checks.groundedness import (
    GroundednessChecker,
    _split_sentences,
    _is_claim,
    _extract_claim,
    _chunk_text,
)


# --- batched fake models -------------------------------------------------------------
# embedder(claims, chunks) -> sim matrix; nli(pairs) -> entailment per pair

def const_embedder(value):
    return lambda claims, chunks: [[value] * len(chunks) for _ in claims]


def const_nli(value):
    return lambda pairs: [value for _ in pairs]


def nli_by_hypothesis(fn):
    return lambda pairs: [fn(p, h) for p, h in pairs]


# --- no source (point 6) -------------------------------------------------------------

def test_no_source_fails_closed_by_default():
    scanner = GroundednessChecker(embedder=const_embedder(0.9), nli_classifier=const_nli(0.9))
    for ctx in ({}, None, {"source": "   "}):
        res = scanner.check("The Eiffel Tower is in Paris.", context=ctx)
        assert res.passed is False
        assert res.error is False
        assert "no source provided" in res.reason
        assert res.metadata["missing_source"] is True


def test_no_source_skip_option_passes():
    scanner = GroundednessChecker(on_missing_source="skip")
    res = scanner.check("The Eiffel Tower is in Paris.", context={})
    assert res.passed is True
    assert res.score == 1.0
    assert res.metadata["skipped"] == "no_source"


def test_invalid_on_missing_source_rejected():
    with pytest.raises(ValueError):
        GroundednessChecker(on_missing_source="maybe")


# --- splitting / claim filter (points 2, 3) --------------------------------------------

def test_split_sentences_preserves_abbreviations_and_numbers():
    text = "Dr. Smith bought 3.14 kg of apples e.g. fruit. He paid $10.50 in the U.S. Then he left."
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Dr. Smith bought 3.14 kg of apples e.g. fruit."
    assert sentences[1] == "He paid $10.50 in the U.S."
    assert sentences[2] == "Then he left."


def test_claim_filter_skips_greetings_questions_and_short_fragments():
    # Greetings & meta
    assert _is_claim("Hello there!") is False
    assert _is_claim("Good morning, user.") is False
    assert _is_claim("Thank you so much!") is False
    assert _is_claim("Sure, here is the answer:") is False

    # Questions
    assert _is_claim("What is the capital of France?") is False
    assert _is_claim("Can you help me?") is False

    # Short fragments (< 4 words)
    assert _is_claim("Yes indeed.") is False
    assert _is_claim("Not quite.") is False

    # Valid factual claims
    assert _is_claim("The Eiffel Tower is located in Paris.") is True
    assert _is_claim("Revenue grew by fifteen percent in 2023.") is True


@pytest.mark.parametrize("sentence", [
    "Let me know if you have any other questions.",
    "I hope this helps.",
    "Feel free to reach out anytime.",
    "Please don't hesitate to contact me again.",
    "If you have any further questions, I am here.",
    "Thank you for your patience with this request.",
    "You're welcome, happy to assist.",
    "Have a great day ahead!",
    "I'm happy to help with anything else.",
    "Is there anything else I can help with today.",
])
def test_pure_chatter_is_not_a_claim(sentence):
    assert _is_claim(sentence) is False


def test_leading_filler_is_stripped_and_rest_is_checked():
    assert _extract_claim("Based on the provided context, the tower was completed in 1898.") == \
        "The tower was completed in 1898."
    assert _extract_claim("Sure, here is the answer: the tower opened in 1898.") == \
        "The tower opened in 1898."
    assert _extract_claim("According to the document, sales fell by half in March.") == \
        "Sales fell by half in March."


def test_thanks_inside_a_real_claim_is_still_a_claim():
    assert _is_claim("Revenue grew 40% thanks to the new Mumbai office.") is True
    assert _is_claim("Thanks to the new office, revenue grew 40% last year.") is True


def test_filler_prefixed_hallucination_is_caught():
    """Point 2: a hallucination behind 'Based on the provided context,' must be checked and fail."""
    seen = []

    def nli(pairs):
        seen.extend(h for _, h in pairs)
        return [0.02 if "1898" in h else 0.97 for _, h in pairs]

    scanner = GroundednessChecker(embedder=const_embedder(0.9), nli_classifier=nli)
    res = scanner.check(
        "Based on the provided context, the tower was completed in 1898. "
        "Revenue grew 40% thanks to the new Mumbai office.",
        context={"source": "The tower was completed in 1889. Revenue grew 40% thanks to the Mumbai office."},
    )
    assert res.passed is False
    assert res.metadata["claims_evaluated"] == 2
    assert res.metadata["ungrounded_indices"] == [0]
    # filler stripped before NLI; the "thanks" sentence was evaluated, not skipped
    assert seen[0] == "The tower was completed in 1898."
    assert any("thanks to the new Mumbai office" in h for h in seen)


def test_chatter_does_not_fail_min_score():
    """Point 3: chatter is skipped, so it can't drag the min score down."""
    scanner = GroundednessChecker(
        embedder=const_embedder(0.9),
        nli_classifier=nli_by_hypothesis(lambda p, h: 0.95 if "Friday" in h else 0.01),
    )
    res = scanner.check(
        "The deadline moved to next Friday. Let me know if you have any other questions. "
        "I hope this helps. Feel free to reach out anytime.",
        context={"source": "The team moved the deadline to next Friday."},
    )
    assert res.passed is True
    assert res.metadata["claims_evaluated"] == 1
    assert res.metadata["sentences_skipped"] == 3
    assert res.metadata["skipped_sentence_indices"] == [1, 2, 3]


# --- metadata hygiene (point 1) --------------------------------------------------------

def test_metadata_never_contains_raw_claim_or_sentence_text():
    scanner = GroundednessChecker(embedder=const_embedder(0.9), nli_classifier=const_nli(0.1))
    claim = "Contact Ravi at ravi@example.com for the refund details."
    res = scanner.check(f"Hello there! {claim} Let me know if you need more.", {"source": "Refunds take 5 days."})

    dumped = json.dumps(res.metadata)
    assert "ravi@example.com" not in dumped
    assert "Contact Ravi" not in dumped
    assert "Hello" not in dumped and "Let me know" not in dumped
    assert res.metadata["ungrounded_indices"] == [1]
    assert res.metadata["skipped_sentence_indices"] == [0, 2]
    assert res.metadata["details"][0]["sentence_index"] == 1
    assert set(res.metadata["details"][0]) == {
        "sentence_index", "best_sim", "action", "score", "chunks_evaluated"
    }


def test_metadata_has_no_text_when_all_sentences_skipped():
    scanner = GroundednessChecker(embedder=const_embedder(0.9), nli_classifier=const_nli(0.9))
    res = scanner.check("Hello there! I hope this helps.", {"source": "anything"})
    assert res.passed is True
    assert "Hello" not in json.dumps(res.metadata)
    assert res.metadata["skipped_sentence_indices"] == [0, 1]


# --- cosine / NLI routing ---------------------------------------------------------------

def test_cosine_below_t_low_rejects_without_calling_nli():
    """Step 4: Cosine < T_low -> UNGROUNDED, skip NLI, score = 0.0"""
    nli_called = False

    def mock_nli(pairs):
        nonlocal nli_called
        nli_called = True
        return [1.0 for _ in pairs]

    scanner = GroundednessChecker(
        t_low=0.35,
        t_ground=0.70,
        embedder=const_embedder(0.15),  # Far below t_low=0.35
        nli_classifier=mock_nli,
    )

    candidate = "Martians discovered gold on the dark side of Neptune."
    source = "The meeting discussed Q3 financial earnings in Tokyo."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is False
    assert res.score == 0.0
    assert nli_called is False  # NLI was successfully skipped!
    assert res.metadata["ungrounded_indices"] == [0]
    assert res.metadata["details"][0]["action"] == "rejected_by_cosine"


def test_high_cosine_is_not_auto_accept():
    """Note 1: High cosine does NOT auto-accept. NLI decides every real claim."""
    scanner = GroundednessChecker(
        t_low=0.35,
        t_ground=0.70,
        embedder=const_embedder(0.98),  # Very high cosine similarity
        nli_classifier=const_nli(0.15),  # NLI catches the negation
    )

    candidate = "The company did not achieve profitability in 2022."
    source = "The company achieved strong profitability in 2022."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is False
    assert res.score == 0.15
    assert res.metadata["ungrounded_indices"] == [0]
    assert res.metadata["details"][0]["action"] == "evaluated_by_nli"


def test_min_claims_aggregation():
    """Step 6: score = min(claims), failing claims tracked by index"""
    def score(premise, hypothesis):
        if "Paris" in hypothesis:
            return 0.95
        if "Berlin" in hypothesis:
            return 0.20  # Failing claim
        return 0.85

    scanner = GroundednessChecker(
        t_low=0.35,
        t_ground=0.70,
        embedder=const_embedder(0.75),
        nli_classifier=nli_by_hypothesis(score),
    )

    candidate = "The capital of France is Paris. The capital of Spain is Berlin."
    source = "France's capital is Paris. Spain's capital is Madrid."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is False
    assert res.score == 0.20  # min(0.95, 0.20)
    assert res.metadata["ungrounded_count"] == 1
    assert res.metadata["ungrounded_indices"] == [1]


def test_fully_grounded_response_passes():
    scanner = GroundednessChecker(
        t_low=0.35,
        t_ground=0.70,
        embedder=const_embedder(0.85),
        nli_classifier=const_nli(0.92),
    )

    candidate = "Hello! The project deadline has been moved to next Friday. Thank you!"
    source = "The team decided to move the project deadline to next Friday."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is True
    assert res.score == 0.92
    assert res.reason is None
    assert res.metadata["ungrounded_count"] == 0
    # Hello and Thank you were filtered out, only the real claim was evaluated
    assert res.metadata["claims_evaluated"] == 1


def test_crash_safety_fails_closed():
    def mock_embedder(claims, chunks):
        raise RuntimeError("Vector database connection timed out")

    scanner = GroundednessChecker(embedder=mock_embedder)
    res = scanner.check("Some claim to verify here.", context={"source": "Some source."})

    assert res.passed is False
    assert res.error is True
    assert "RuntimeError" in res.reason


def test_missing_embedder_dependency_is_an_error_not_a_fallback(monkeypatch):
    """Point 4: no silent lexical fallback; an unloadable embedder becomes an error result."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    scanner = GroundednessChecker(nli_classifier=const_nli(0.99))
    res = scanner.check("The Eiffel Tower was completed in 1898.", {"source": "The Eiffel Tower was completed in 1889."})
    assert res.passed is False
    assert res.error is True
    assert "ImportError" in res.reason


def test_structured_json_extracts_clean_claims_without_syntax():
    """Gap 1: Ensure JSON brackets, keys, numbers are stripped from claims."""
    evaluated_claims = []

    def mock_embedder(claims, chunks):
        evaluated_claims.extend(claims)
        return [[0.85] * len(chunks) for _ in claims]

    scanner = GroundednessChecker(embedder=mock_embedder, nli_classifier=const_nli(0.95))

    raw_json = '{"summary": "Paris is the capital of France.", "stats": {"score": 99, "valid": true}}'
    parsed_dict = {"summary": "Paris is the capital of France.", "stats": {"score": 99, "valid": True}}

    res = scanner.check(raw_json, context={"source": "France's capital is Paris.", "parsed": parsed_dict})

    assert res.passed is True
    # Evaluated claim must be pure text, NOT {"summary": ...}
    assert evaluated_claims == ["Paris is the capital of France."]


def test_top_k_chunks_finds_evidence_in_secondary_chunk():
    """Gap 2: Ensure top-k chunks are evaluated and highest entailment is taken."""
    evaluated_pairs = []

    def mock_embedder(claims, chunks):
        # Chunk 0 has 0.80 similarity, others have 0.75
        return [[0.80 if i == 0 else 0.75 for i in range(len(chunks))] for _ in claims]

    def mock_nli(pairs):
        evaluated_pairs.extend(pairs)
        return [0.95 if "Proof that revenue reached 10M" in p else 0.20 for p, _ in pairs]

    scanner = GroundednessChecker(top_k=2, embedder=mock_embedder, nli_classifier=mock_nli)

    # Multi-chunk source (> DEFAULT_CHUNK_SIZE of 200 words)
    source = ("Chunk 1: Intro to company history. " * 45) + ("Chunk 2: Proof that revenue reached 10M. " * 45)
    candidate = "Revenue reached ten million dollars."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is True
    assert res.score == 0.95
    assert len(evaluated_pairs) == 2


# --- batching (point 5) ------------------------------------------------------------------

def test_embedder_and_nli_are_each_called_once_per_check():
    calls = {"embed": 0, "nli": 0, "pairs": 0}

    def embedder(claims, chunks):
        calls["embed"] += 1
        assert len(claims) == 5  # all claims batch-encoded together
        return [[0.9] * len(chunks) for _ in claims]

    def nli(pairs):
        calls["nli"] += 1
        calls["pairs"] += len(pairs)
        return [0.95 for _ in pairs]

    scanner = GroundednessChecker(top_k=1, embedder=embedder, nli_classifier=nli)
    answer = ("The tower was completed in 1889. It was built for the World's Fair. "
              "Gustave Eiffel's company designed the tower. The tower stands 330 metres tall. "
              "It sits on the Champ de Mars.")
    res = scanner.check(answer, {"source": "The Eiffel Tower facts. " * 10})
    assert res.passed is True
    assert calls == {"embed": 1, "nli": 1, "pairs": 5}


# --- source chunks (point 7) ---------------------------------------------------------------

def test_ranks_across_all_chunks_by_default():
    """Evidence only in the LAST chunk of a long source must still be found."""
    def embedder(claims, chunks):
        return [[0.9 if "EVIDENCE" in c else 0.4 for c in chunks] for _ in claims]

    def nli(pairs):
        return [0.97 if "EVIDENCE" in p else 0.05 for p, _ in pairs]

    source = ("filler words here. " * 3000) + ("EVIDENCE revenue reached ten million. " * 30)
    scanner = GroundednessChecker(top_k=1, embedder=embedder, nli_classifier=nli)
    res = scanner.check("Revenue reached ten million dollars.", {"source": source})

    assert len(_chunk_text(source)) > 50
    assert res.passed is True
    assert res.metadata["source_truncated"] is False
    assert res.metadata["source_chunks_ranked"] == res.metadata["source_chunks_total"]


def test_explicit_chunk_cap_records_truncation():
    source = "word " * 2000
    scanner = GroundednessChecker(max_source_chunks=2, embedder=const_embedder(0.9), nli_classifier=const_nli(0.9))
    res = scanner.check("The source has many words in it.", {"source": source})
    assert res.metadata["source_truncated"] is True
    assert res.metadata["source_chunks_ranked"] == 2
    assert res.metadata["source_chunks_total"] > 2
