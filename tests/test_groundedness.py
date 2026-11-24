import pytest
from guardrails.checks.groundedness import (
    GroundednessChecker,
    _split_sentences,
    _is_claim,
    _chunk_text,
)


def test_no_source_skips_check():
    scanner = GroundednessChecker()
    # No source in context
    res = scanner.check("The Eiffel Tower is in Paris.", context={})
    assert res.passed is True
    assert res.score == 1.0
    assert res.metadata["skipped"] == "no_source"

    # None context
    res2 = scanner.check("The Eiffel Tower is in Paris.", context=None)
    assert res2.passed is True
    assert res2.metadata["skipped"] == "no_source"


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


def test_cosine_below_t_low_rejects_without_calling_nli():
    """Step 4: Cosine < T_low -> UNGROUNDED, skip NLI, score = 0.0"""
    nli_called = False

    def mock_embedder(query, docs):
        return [0.15]  # Far below t_low=0.35

    def mock_nli(premise, hypothesis):
        nonlocal nli_called
        nli_called = True
        return 1.0

    scanner = GroundednessChecker(
        t_low=0.35,
        t_ground=0.70,
        embedder=mock_embedder,
        nli_classifier=mock_nli,
    )

    candidate = "Martians discovered gold on the dark side of Neptune."
    source = "The meeting discussed Q3 financial earnings in Tokyo."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is False
    assert res.score == 0.0
    assert nli_called is False  # NLI was successfully skipped!
    assert candidate in res.metadata["ungrounded"]
    assert res.metadata["details"][0]["action"] == "rejected_by_cosine"


def test_high_cosine_is_not_auto_accept():
    """Note 1: High cosine does NOT auto-accept. NLI decides every real claim."""
    def mock_embedder(query, docs):
        return [0.98]  # Very high cosine similarity

    def mock_nli(premise, hypothesis):
        # NLI catches the factual negation or wrong number!
        return 0.15  # Contradiction / low entailment

    scanner = GroundednessChecker(
        t_low=0.35,
        t_ground=0.70,
        embedder=mock_embedder,
        nli_classifier=mock_nli,
    )

    candidate = "The company did not achieve profitability in 2022."
    source = "The company achieved strong profitability in 2022."

    res = scanner.check(candidate, context={"source": source})

    # High cosine got routed to NLI, and NLI rejected it!
    assert res.passed is False
    assert res.score == 0.15
    assert candidate in res.metadata["ungrounded"]
    assert res.metadata["details"][0]["action"] == "evaluated_by_nli"


def test_min_claims_aggregation():
    """Step 6: score = min(claims), failing claims tracked in metadata.ungrounded"""
    def mock_embedder(query, docs):
        return [0.75]

    def mock_nli(premise, hypothesis):
        if "Paris" in hypothesis:
            return 0.95
        if "Berlin" in hypothesis:
            return 0.20  # Failing claim
        return 0.85

    scanner = GroundednessChecker(
        t_low=0.35,
        t_ground=0.70,
        embedder=mock_embedder,
        nli_classifier=mock_nli,
    )

    candidate = "The capital of France is Paris. The capital of Spain is Berlin."
    source = "France's capital is Paris. Spain's capital is Madrid."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is False
    assert res.score == 0.20  # min(0.95, 0.20)
    assert len(res.metadata["ungrounded"]) == 1
    assert "The capital of Spain is Berlin." in res.metadata["ungrounded"]


def test_fully_grounded_response_passes():
    def mock_embedder(query, docs):
        return [0.85]

    def mock_nli(premise, hypothesis):
        return 0.92

    scanner = GroundednessChecker(
        t_low=0.35,
        t_ground=0.70,
        embedder=mock_embedder,
        nli_classifier=mock_nli,
    )

    candidate = "Hello! The project deadline has been moved to next Friday. Thank you!"
    source = "The team decided to move the project deadline to next Friday."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is True
    assert res.score == 0.92
    assert res.reason is None
    assert len(res.metadata["ungrounded"]) == 0
    # Hello and Thank you were filtered out, only the real claim was evaluated
    assert res.metadata["claims_evaluated"] == 1


def test_crash_safety_fails_closed():
    def mock_embedder(query, docs):
        raise RuntimeError("Vector database connection timed out")

    scanner = GroundednessChecker(embedder=mock_embedder)
    res = scanner.check("Some claim to verify.", context={"source": "Some source."})

    assert res.passed is False
    assert res.error is True
    assert "RuntimeError" in res.reason


def test_structured_json_extracts_clean_claims_without_syntax():
    """Gap 1: Ensure JSON brackets, keys, numbers are stripped from claims."""
    evaluated_claims = []

    def mock_embedder(query, docs):
        evaluated_claims.append(query)
        return [0.85]

    def mock_nli(premise, hypothesis):
        return 0.95

    scanner = GroundednessChecker(embedder=mock_embedder, nli_classifier=mock_nli)

    raw_json = '{"summary": "Paris is the capital of France.", "stats": {"score": 99, "valid": true}}'
    parsed_dict = {"summary": "Paris is the capital of France.", "stats": {"score": 99, "valid": True}}

    res = scanner.check(raw_json, context={"source": "France's capital is Paris.", "parsed": parsed_dict})

    assert res.passed is True
    # Evaluated claim must be pure text, NOT {"summary": ...}
    assert evaluated_claims == ["Paris is the capital of France."]
    assert "{" not in evaluated_claims[0]
    assert "summary" not in evaluated_claims[0]


def test_top_k_chunks_finds_evidence_in_secondary_chunk():
    """Gap 2: Ensure top-k chunks are evaluated and highest entailment is taken."""
    evaluated_pairs = []

    def mock_embedder(query, docs):
        # Chunk 0 has 0.80 similarity, others have 0.75
        return [0.80 if i == 0 else 0.75 for i in range(len(docs))]

    def mock_nli(premise, hypothesis):
        evaluated_pairs.append((premise, hypothesis))
        if "Proof that revenue reached 10M" in premise:
            return 0.95  # Found proof in secondary chunk!
        return 0.20  # Chunk 0 has high cosine but no actual proof

    scanner = GroundednessChecker(top_k=2, embedder=mock_embedder, nli_classifier=mock_nli)

    # Multi-chunk source (> DEFAULT_CHUNK_SIZE of 200 words)
    source = ("Chunk 1: Intro to company history. " * 45) + ("Chunk 2: Proof that revenue reached 10M. " * 45)
    candidate = "Revenue reached ten million dollars."

    res = scanner.check(candidate, context={"source": source})

    assert res.passed is True
    assert res.score == 0.95
    assert len(evaluated_pairs) == 2
