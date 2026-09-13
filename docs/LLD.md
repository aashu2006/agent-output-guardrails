### LLD for agent-output-guardrails

**What this is**

This doc turns the HLD into concrete modules, classes, and interfaces. The HLD says what exists and why. This says exactly how we build it.

**Repo structure**

```
agent-output-guardrails/
├── src/guardrails/
│   ├── __init__.py
│   ├── results.py          # GuardrailResult model
│   ├── base.py             # Guardrail contract (ABC)
│   ├── checks/
│   │   ├── schema.py       # SchemaValidator
│   │   ├── pii.py          # PIIScanner
│   │   ├── groundedness.py # GroundednessChecker (Phase 2)
│   │   └── toxicity.py     # ToxicityScanner (Phase 2)
│   ├── aggregator.py       # ResultAggregator
│   ├── policy.py           # PolicyEngine + actions
│   ├── reask.py            # ReaskEngine (Phase 2)
│   ├── providers.py        # LLMProvider interface
│   ├── engine.py           # GuardrailEngine (coordinator, public API)
│   └── config.py           # engine configuration
├── tests/
│   ├── test_schema.py
│   ├── test_pii.py
│   ├── test_policy.py
│   ├── test_reask.py
│   └── redteam/            # adversarial cases
├── docs/
└── pyproject.toml
```

**Core data model (results.py)**

```python
from pydantic import BaseModel, Field

class GuardrailResult(BaseModel):
    guardrail: str                          # "schema" | "pii" | "groundedness" | "toxicity"
    passed: bool
    score: float | None = None              # ML checks only
    reason: str | None = None               # human-readable failure reason
    transformed_output: str | None = None   # PII's proposed cleaned version
    metadata: dict = Field(default_factory=dict)
```

Decision on the open question from the HLD: no separate MODIFIED status. "Modified" = `passed=True` + `transformed_output` present. One bool, policy checks the field. Simpler than an enum with overlap.

Errors are results too: a guardrail that crashes internally returns `passed=False, reason="internal error: ..."` with details in metadata. The pipeline never throws past a check.

**Guardrail contract (base.py)**

```python
from abc import ABC, abstractmethod

class Guardrail(ABC):
    name: str

    @abstractmethod
    def check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        ...
```

The contract enforces the rules by shape:

- Input is the candidate string plus optional context. No LLM client is ever passed in, so a check physically cannot call the LLM.
- Output is only a GuardrailResult. No side effects, no mutation of the candidate.
- `context` carries what a specific check needs: groundedness reads `context["source"]`, others ignore it.

**SchemaValidator (checks/schema.py)**

```python
class SchemaValidator(Guardrail):
    name = "schema"
    def __init__(self, schema: type[BaseModel]): ...
    def check(self, candidate, context=None) -> GuardrailResult: ...
```

- Try `json.loads`, then `schema.model_validate`. Any failure: `passed=False`, `reason` = flattened Pydantic error list ("field X: must be integer"), full errors in `metadata["errors"]`.
- On success, `metadata["parsed"]` holds the validated dict so downstream doesn't parse twice.
- Deterministic, no models, no config beyond the schema class.

**PIIScanner (checks/pii.py)**

```python
class PIIScanner(Guardrail):
    name = "pii"
    PATTERNS: dict[str, re.Pattern]   # category -> compiled regex
```

V1 categories and approach:

| Category | Detection | Replacement |
|---|---|---|
| Email | regex | `[REDACTED_EMAIL]` |
| Phone (incl. +91) | regex, 10-digit with optional +91/0 prefix | `[REDACTED_PHONE]` |
| Aadhaar-pattern | 12 digits, spaced or plain, word-boundary | `[REDACTED_ID]` |
| PAN | `[A-Z]{5}[0-9]{4}[A-Z]` | `[REDACTED_ID]` |
| Credit card | 13-19 digit candidates, then Luhn check to confirm | `[REDACTED_CARD]` |
| IP address | IPv4 regex with octet range check | `[REDACTED_IP]` |

- Runs all patterns, builds the sanitized string. Anything found: `passed=True, transformed_output=<sanitized>` (detection is not a failure, it's a modify case), `metadata["found"] = {"email": 2, ...}` with counts only. Raw matched values never go into reason or metadata. No PII in logs, ever.
- Order matters: card check runs before phone and Aadhaar so a 16-digit card doesn't get half-eaten by other number patterns. Luhn kills false positives on random digit runs.

**GroundednessChecker (checks/groundedness.py, Phase 2)**

- Split candidate into sentences (simple rule-based splitter, no heavy NLP dep).
- Prefilter: embedding cosine similarity of each sentence vs source chunks. Above high threshold: assume grounded, skip NLI. Below low threshold: no supporting chunk at all, mark ungrounded. In between: run NLI.
- NLI: local DeBERTa-class model, premise = best-matching source chunk, hypothesis = sentence. Entailment prob = sentence score.
- Result: `score` = min sentence score, `passed = score >= threshold`, failing sentences in `metadata["ungrounded"]`.
- Exact model and thresholds get fixed after the W3 spike benchmark. The interface doesn't change either way.

**ToxicityScanner (checks/toxicity.py, Phase 2)**

- Local classifier (DistilBERT-class), single forward pass on the candidate.
- `score` = toxicity prob, `passed = score < threshold`. Nothing else. Blocking is policy's job.

**Result aggregator (aggregator.py)**

```python
class AggregatedResults(BaseModel):
    results: list[GuardrailResult]
    def get(self, name: str) -> GuardrailResult | None: ...
    @property
    def transformed_output(self) -> str | None: ...   # from PII if present
```

Thin by design. Collects the parallel run's results, exposes lookup. No logic that resembles a decision.

**Policy engine (policy.py)**

```python
from enum import Enum

class Action(str, Enum):
    ALLOW = "allow"
    MODIFY = "modify"
    REASK = "reask"
    BLOCK = "block"

class Decision(BaseModel):
    action: Action
    reason: str
    output: str | None = None      # final text for ALLOW/MODIFY
    reask_feedback: str | None = None

class PolicyEngine:
    def __init__(self, config: PolicyConfig): ...
    def decide(self, agg: AggregatedResults, retries_left: bool) -> Decision: ...
```

`decide()` logic, in precedence order (block > re-ask > modify > allow):

1. Toxicity failed: BLOCK, incident id in reason.
2. Schema failed: REASK if retries_left else BLOCK (fail closed). Phase 1: always fail closed.
3. Groundedness failed: REASK if retries_left else BLOCK with "cannot verify against source".
4. PII transformed_output present: MODIFY with the sanitized text.
5. Everything clean: ALLOW with the original candidate.

Pure function of (results, retries_left, config). No LLM, no state, trivially unit-testable.

**Re-ask engine (reask.py, Phase 2)**

```python
class ReaskEngine:
    def __init__(self, provider: LLMProvider, max_retries: int = 2): ...
    def run(self, request: OriginalRequest, feedback: str, context: dict) -> str: ...
    @property
    def retries_left(self) -> bool: ...
```

- Decision on the HLD open question: one shared per-request budget (default 2). Simpler to reason about, one number to enforce, cost-bounded. Per-reason budgets can come later via config.
- Repair request = original prompt + "Your previous response failed validation: {feedback}. Return a corrected response." + re-attached source for groundedness re-asks.
- Multiple failures at once: feedback follows the same precedence order, the highest-priority failure's feedback goes in (a schema error makes the other feedback unreliable anyway).
- The engine only builds the request and counts attempts. The new response goes back through `GuardrailEngine.validate()` in full.

**Provider interface (providers.py)**

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str: ...
```

One method. Concrete impls (OpenAI-compatible, local) live behind it. Tests use a `FakeProvider` returning scripted responses, so the whole pipeline including re-ask is testable with zero API calls.

**Coordinator / public API (engine.py)**

```python
class GuardrailEngine:
    def __init__(self, schema, config, provider=None): ...
    def validate(self, candidate: str, source: str | None = None) -> Decision: ...
```

`validate()` flow:

1. Run SchemaValidator. Fail: policy(schema-fail). Phase 1 blocks, Phase 2 enters the re-ask loop.
2. Pass: run PII + groundedness + toxicity concurrently (`concurrent.futures.ThreadPoolExecutor`, 3 workers; model inference releases the GIL, so threads are enough, no async complexity in V1).
3. Aggregate, `policy.decide()`, return Decision.
4. REASK decision: `reask.run()`, then the new candidate goes back into `validate()` until allow/modify/block or the budget runs out.

Timers wrap each stage and the whole call; timings land in Decision metadata for observability. p50/p95 get computed by the benchmark harness, not in the hot path.

**Config (config.py)**

```python
class PolicyConfig(BaseModel):
    enabled: list[str] = ["schema", "pii", "groundedness", "toxicity"]
    groundedness_threshold: float = 0.7
    toxicity_threshold: float = 0.8
    max_retries: int = 2
```

Phase 1-2: plain Pydantic settings object, loadable from a dict. The YAML/DSL front-end is Phase 3 and parses into this same object, so nothing downstream changes when it lands.

**Testing plan**

- **Unit:** each check in isolation. Schema: valid/invalid/edge JSON. PII: per-category positives, near-miss negatives (a 15-digit non-Luhn number must NOT match card), overlap cases.
- **Policy:** table-driven tests, every combination of results to expected action. Pure function, cheap to cover fully.
- **Re-ask:** FakeProvider scripted to fail N times then succeed; assert attempt counts, budget exhaustion, fail-closed.
- **E2E:** full engine with FakeProvider, one test per decision path (allow, modify, re-ask-then-allow, block, fail-closed).
- **Red-team (tests/redteam/):** adversarial cases added alongside each feature from W3: obfuscated PII (spaced digits, unicode), almost-valid JSON, toxicity evasion phrasings.

**Build order**

1. `results.py` + `base.py` (the contract)
2. `checks/schema.py` + tests + a demo script that catches a deliberately broken response
3. `checks/pii.py` + tests
4. `policy.py` + `aggregator.py` (Phase 1 mode: fail closed, modify, allow)
5. `providers.py` + `reask.py` + the re-ask loop in engine (Phase 2 starts)
6. In parallel (Soumya): toxicity integration, NLI spike benchmark, eval set labeling
7. `checks/groundedness.py` with the prefilter
8. Benchmark harness, observability, then DSL only if the schedule holds

**Proposed answers to the HLD's open questions**

These three were left open in the HLD. Our proposed answers:

- Status semantics: no MODIFIED enum, modified = passed + transformed_output present.
- Retry budget: one shared per-request budget, not per-reason.
- Multi-failure repair prompts: highest-priority failure's feedback only.

Thresholds (0.7 / 0.8) and exact model choices are placeholders until the W3 benchmark and evaluation runs.
