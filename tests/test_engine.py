from pydantic import BaseModel
from guardrails.engine import GuardrailEngine
from guardrails.checks.toxicity import ToxicityScanner
from guardrails.config import PolicyConfig
from guardrails.policy import Action


class UserProfile(BaseModel):
    username: str
    bio: str


def test_engine_toxicity_blocks_unsafe_output():
    mock_classifier = lambda text: 0.95  # always toxic
    toxic_scanner = ToxicityScanner(threshold=0.8, classifier=mock_classifier)
    config = PolicyConfig(enabled=["schema", "pii", "toxicity"])
    engine = GuardrailEngine(UserProfile, config=config, toxicity_scanner=toxic_scanner)

    candidate = '{"username": "bad_user", "bio": "You stupid idiot"}'
    decision = engine.validate(candidate)

    assert decision.action == Action.BLOCK
    assert "toxic content detected" in decision.reason
    assert "incident_id=" in decision.reason
    assert decision.output is None


def test_engine_toxicity_allows_safe_output():
    mock_classifier = lambda text: 0.05  # safe
    toxic_scanner = ToxicityScanner(threshold=0.8, classifier=mock_classifier)
    config = PolicyConfig(enabled=["schema", "pii", "toxicity"])
    engine = GuardrailEngine(UserProfile, config=config, toxicity_scanner=toxic_scanner)

    candidate = '{"username": "good_user", "bio": "Hello world, have a nice day!"}'
    decision = engine.validate(candidate)

    assert decision.action == Action.ALLOW
    assert decision.reason == "all checks passed"
    assert decision.output == candidate


def test_engine_skips_toxicity_when_not_in_enabled():
    mock_classifier = lambda text: 1 / 0  # would crash if called
    toxic_scanner = ToxicityScanner(classifier=mock_classifier)
    config = PolicyConfig(enabled=["schema", "pii"])  # toxicity disabled!
    engine = GuardrailEngine(UserProfile, config=config, toxicity_scanner=toxic_scanner)

    candidate = '{"username": "user1", "bio": "Normal bio"}'
    decision = engine.validate(candidate)

    # Must pass without crashing because toxicity was skipped!
    assert decision.action == Action.ALLOW


def test_engine_runs_groundedness_in_parallel():
    from guardrails.checks.groundedness import GroundednessChecker

    mock_embed = lambda claims, chunks: [[0.85] * len(chunks) for _ in claims]
    mock_nli = lambda pairs: [0.95 for _ in pairs]  # fully grounded
    grounded_checker = GroundednessChecker(embedder=mock_embed, nli_classifier=mock_nli)

    config = PolicyConfig(enabled=["schema", "pii", "groundedness"])
    engine = GuardrailEngine(UserProfile, config=config, groundedness_checker=grounded_checker)

    candidate = '{"username": "user1", "bio": "Paris is the capital of France."}'
    source = "France has Paris as its capital city."

    decision = engine.validate(candidate, source=source)

    assert decision.action == Action.ALLOW
    assert "parallel" in decision.timings
    assert decision.output == candidate


def test_engine_blocks_ungrounded_output():
    from guardrails.checks.groundedness import GroundednessChecker

    mock_embed = lambda claims, chunks: [[0.10] * len(chunks) for _ in claims]  # below t_low -> ungrounded
    grounded_checker = GroundednessChecker(embedder=mock_embed)

    config = PolicyConfig(enabled=["schema", "groundedness"])
    engine = GuardrailEngine(UserProfile, config=config, groundedness_checker=grounded_checker)

    candidate = '{"username": "user1", "bio": "Aliens landed in Times Square yesterday."}'
    source = "Stock market updates for tech companies."

    decision = engine.validate(candidate, source=source)

    assert decision.action == Action.BLOCK
    assert "cannot verify against source" in decision.reason
    assert decision.output is None
