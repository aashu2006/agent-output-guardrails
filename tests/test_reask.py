import pytest
from pydantic import BaseModel
from guardrails.engine import GuardrailEngine
from guardrails.config import PolicyConfig
from guardrails.policy import Action
from guardrails.providers import FakeProvider
from guardrails.reask import ReaskEngine
from guardrails.checks.groundedness import GroundednessChecker


class User(BaseModel):
    name: str
    age: int


def test_reask_engine_builds_repair_prompt_with_source():
    provider = FakeProvider(responses=["dummy"])
    engine = ReaskEngine(provider=provider, max_retries=2)

    prompt = engine.build_repair_prompt(
        original_prompt="Summarize financial results.",
        feedback="Missing required field 'revenue'.",
        source="Company revenue was $10M in 2023.",
    )

    assert "Summarize financial results." in prompt
    assert "[VALIDATION FEEDBACK]" in prompt
    assert "Missing required field 'revenue'." in prompt
    assert "[REFERENCE CONTEXT]" in prompt
    assert "Company revenue was $10M in 2023." in prompt


def test_reask_engine_exhausts_budget_and_raises():
    provider = FakeProvider(responses=["retry 1", "retry 2"])
    engine = ReaskEngine(provider=provider, max_retries=2)

    assert engine.retries_left == 2

    # Attempt 1
    engine.reask("prompt", "feedback 1")
    assert engine.attempts == 1
    assert engine.retries_left == 1

    # Attempt 2
    engine.reask("prompt", "feedback 2")
    assert engine.attempts == 2
    assert engine.retries_left == 0

    # Attempt 3 -> Should raise
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        engine.reask("prompt", "feedback 3")


def test_engine_run_reasks_and_succeeds_on_schema_error():
    """1st response has broken JSON; 2nd response fixes it -> ALLOW."""
    broken_response = '{"name": "Alice", "age": "thirty"}'
    fixed_response = '{"name": "Alice", "age": 30}'

    provider = FakeProvider(responses=[broken_response, fixed_response])
    config = PolicyConfig(enabled=["schema", "pii"], max_retries=2)
    engine = GuardrailEngine(User, config=config, provider=provider)

    decision = engine.run("Provide user details for Alice.")

    assert decision.action == Action.ALLOW
    assert decision.output == fixed_response
    assert provider.call_count == 2
    # Verify feedback was in the second prompt
    assert "[VALIDATION FEEDBACK]" in provider.history[1]
    assert "age" in provider.history[1]


def test_engine_run_reasks_and_succeeds_on_groundedness_error():
    """1st response is ungrounded; 2nd response is grounded -> ALLOW."""
    hallucinated = '{"name": "Alice was born in Antarctica.", "age": 25}'
    grounded = '{"name": "Alice was born in London.", "age": 25}'

    provider = FakeProvider(responses=[hallucinated, grounded])
    source = "Alice was born in London and works as a software engineer."

    def mock_embedder(q, d):
        return [0.85]

    def mock_nli(premise, hypothesis):
        if "Antarctica" in hypothesis:
            return 0.10  # Ungrounded
        return 0.95      # Grounded

    grounded_checker = GroundednessChecker(embedder=mock_embedder, nli_classifier=mock_nli)
    config = PolicyConfig(enabled=["schema", "groundedness"], max_retries=2)
    engine = GuardrailEngine(
        User,
        config=config,
        groundedness_checker=grounded_checker,
        provider=provider,
    )

    decision = engine.run("Extract Alice's details.", source=source)

    assert decision.action == Action.ALLOW
    assert decision.output == grounded
    assert provider.call_count == 2
    assert "[REFERENCE CONTEXT]" in provider.history[1]


def test_engine_run_blocks_when_max_retries_exhausted():
    """3 consecutive broken responses with max_retries=2 -> BLOCK."""
    broken_1 = '{"name": "Alice", age: invalid}'
    broken_2 = '{"name": "Alice", "age": "still wrong"}'
    broken_3 = '{"name": "Alice", "age": "never right"}'

    provider = FakeProvider(responses=[broken_1, broken_2, broken_3])
    config = PolicyConfig(enabled=["schema"], max_retries=2)
    engine = GuardrailEngine(User, config=config, provider=provider)

    decision = engine.run("Give me Alice.")

    assert decision.action == Action.BLOCK
    assert "schema validation failed" in decision.reason
    assert decision.output is None
    assert provider.call_count == 3  # Initial + 2 retries


def test_engine_run_blocks_immediately_on_toxicity():
    """Toxicity check should NEVER reask; it must fail closed immediately."""
    toxic_response = '{"name": "Alice", "age": 25, "note": "You stupid idiot"}'
    provider = FakeProvider(responses=[toxic_response, '{"name": "Alice", "age": 25}'])

    mock_toxic = lambda text: 0.95
    from guardrails.checks.toxicity import ToxicityScanner
    toxic_scanner = ToxicityScanner(classifier=mock_toxic)

    config = PolicyConfig(enabled=["schema", "toxicity"], max_retries=2)
    engine = GuardrailEngine(User, config=config, toxicity_scanner=toxic_scanner, provider=provider)

    decision = engine.run("Greet user.")

    assert decision.action == Action.BLOCK
    assert "toxic content detected" in decision.reason
    assert provider.call_count == 1  # Did NOT retry!


def test_engine_run_modifies_on_pii_without_reask():
    """PII redaction does not need a reask; it modifies and delivers."""
    leaky_response = '{"name": "Reach me at test@gmail.com", "age": 25}'
    provider = FakeProvider(responses=[leaky_response])

    config = PolicyConfig(enabled=["schema", "pii"], max_retries=2)
    engine = GuardrailEngine(User, config=config, provider=provider)

    decision = engine.run("Who is Alice?")

    assert decision.action == Action.MODIFY
    assert "[REDACTED_EMAIL]" in decision.output
    assert provider.call_count == 1  # Cleaned directly, no retry needed
