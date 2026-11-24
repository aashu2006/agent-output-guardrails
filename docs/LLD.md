### LLD for agent-output-guardrails

How the code is actually built. The HLD says what exists and why; this doc matches the current source.

**Repo structure**

```
agent-output-guardrails/
├── src/guardrails/
│   ├── __init__.py          # public exports
│   ├── results.py           # GuardrailResult
│   ├── base.py              # Guardrail ABC: check() wraps _check()
│   ├── aggregator.py        # AggregatedResults
│   ├── policy.py            # Action, Decision, PolicyEngine
│   ├── engine.py            # GuardrailEngine: validate() + run()
│   ├── reask.py             # ReaskEngine, repair prompt, delimiter sanitizer
│   ├── providers.py         # LLMProvider ABC, FakeProvider
│   ├── config.py            # PolicyConfig
│   └── checks/
│       ├── schema.py        # SchemaValidator, extract_fenced_json
│       ├── pii.py           # PIIScanner, _redact_text
│       ├── toxicity.py      # ToxicityScanner (unitary/toxic-bert)
│       └── groundedness.py  # GroundednessChecker (MiniLM + DeBERTa NLI)
├── benchmarks/
│   └── latency_bench.py     # latency harness, p50/p95 per stage
├── examples/
│   └── demo.py
├── tests/
│   ├── conftest.py          # --run-slow option, skips @pytest.mark.slow by default
│   ├── test_schema.py  test_pii.py  test_toxicity.py  test_groundedness.py
│   ├── test_policy.py  test_engine.py  test_reask.py  test_benchmarks.py
│   ├── test_toxicity_model.py  test_groundedness_model.py   # real models, slow
│   └── redteam/             # adversarial bypass cases
├── docs/                    # HLD.md, LLD.md
└── pyproject.toml           # extras: dev (pytest), ml (transformers, torch, sentence-transformers)
```

**GuardrailResult (results.py)**

```python
class GuardrailResult(BaseModel):
    guardrail: str
    passed: bool
    error: bool = False                     # True only when the check crashed
    score: float | None = None
    reason: str | None = None
    transformed_output: str | None = None   # PII's redacted version
    metadata: dict = Field(default_factory=dict)
```

There is no MODIFIED status: "modify" means `passed=True` with `transformed_output` set.

**Crash-safe contract (base.py)**

```python
class Guardrail(ABC):
    name: str
    def check(self, candidate, context=None) -> GuardrailResult:   # never raises
        try:
            return self._check(candidate, context)
        except Exception as e:
            return GuardrailResult(guardrail=self.name, passed=False, error=True,
                                   reason=f"internal error: {type(e).__name__}")
    @abstractmethod
    def _check(self, candidate, context=None) -> GuardrailResult: ...
```

- Checks implement `_check()`. Callers use `check()`.
- Any exception, including a model that can't load, becomes `error=True`, and policy blocks on it.
- Checks only detect. They never mutate the candidate and never call the LLM. Redaction is proposed through `transformed_output`.
- Metadata holds counts, scores, flags and indices only, never raw text or PII values.

**SchemaValidator (checks/schema.py)**

- `json.loads` on the raw text first, with an `object_pairs_hook` that rejects duplicate keys. Valid JSON is never altered, even when a string value contains a code fence.
- Only if that parse fails, and `strip_markdown=True` (the default): `extract_fenced_json()` looks for exactly one fenced block and parses its body. Zero or several fences means fail.
- Then `schema.model_validate`.
- Failure reasons:
  - `invalid JSON: <msg> at position <n>`
  - `invalid JSON: duplicate key detected` (the key name is never echoed)
  - field errors as `loc: msg`, with `metadata["errors"]` holding only `loc`, `msg` and `type` (no rejected input values)
- Success: `metadata = {"parsed": ..., "fence_stripped": bool, "prose_outside_fence": bool}`.

**PIIScanner (checks/pii.py)**

| Category | Detection | Replacement |
|---|---|---|
| card | 13-19 digits (spaces/dashes), Luhn must pass | `[REDACTED_CARD]` |
| aadhaar | 4-4-4 digits | `[REDACTED_ID]` |
| phone | contiguous 10 digits (+91/0 prefix); `(ddd) ddd-dddd`; separated forms (`9 8 7 ...`, `98765 43210`, `987.654.3210`) must start with 6-9 | `[REDACTED_PHONE]` |
| email | `@`, `[at]`/`(at)`/`[dot]`/`(dot)`, worded `x at y dot com` (alphabetic TLD) | `[REDACTED_EMAIL]` |
| pan | `AAAAA9999A`; lowercase or mixed case needs a valid holder-type 4th letter or a nearby keyword (pan/tax/id) | `[REDACTED_ID]` |
| ip | exactly four dotted octets, not part of a longer dotted run | `[REDACTED_IP]` |

- Dict order is priority. On overlapping matches the earlier category wins, so a card isn't half-matched as a phone number.
- Matching runs on a copy with zero-width characters removed. Spans are mapped back and redacted in the original, so ZWJ emoji survive.
- With `context["parsed"]`, keys and string values are redacted recursively. `transformed_output = json.dumps(cleaned)` when anything was found. `context["outside_text"]` (prose around a fence) is scanned and counted too.
- Without `parsed`, the raw candidate is redacted.
- Always `passed=True`. `metadata = {"found": {category: count}}`.

**ToxicityScanner (checks/toxicity.py)**

- Model: `unitary/toxic-bert`, a HuggingFace pipeline with `top_k=None`, sigmoid outputs (multi-label), truncation at 512 and `batch_size=16`. It loads lazily, or you can inject a `classifier`.
- Texts scored: every string value of the parsed JSON. If there are none, the raw candidate is scored.
- Each text is split into 64-token windows with 16 overlapping, cut from the original text via tokenizer offsets. Without offsets it falls back to 48-word windows with 12 overlapping.
- All windows go through one classifier call.
- Per window, the score is the max over the toxic sublabels (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`). The check score is the max over all windows.
- `passed = score < threshold` (default 0.8). Output with unrecognised labels raises, which becomes an error result and then BLOCK.

**GroundednessChecker (checks/groundedness.py)**

1. **No source.** With `on_missing_source="block"` (the default) the result is `passed=False`, reason "no source provided…", `metadata["missing_source"]=True`. With `"skip"` the result is `passed=True` and `skipped="no_source"`.
2. **Split.** Strings come from `parsed` (or the raw candidate) and are split into sentences, keeping abbreviations, decimals and initials together.
3. **Claim filter.** Questions are skipped. Anchored leading filler ("Sure, here is the answer:", "Based on the provided context,") is stripped and the rest is checked. A sentence is skipped only if it is entirely chatter (greetings, thanks, "let me know if…", "I hope this helps", "feel free to…") or shorter than 4 words.
4. **Chunk.** The source is split into 200-word chunks with 40 words of overlap. Ranking covers all chunks; the optional `max_source_chunks` cap sets `source_truncated=True` when it applies.
5. **Embed.** One batched call to `embedder(claims, chunks) -> sim matrix`. The default is sentence-transformers `all-MiniLM-L6-v2`, encoding chunks once and claims once. If it can't load, it raises; there is no lexical fallback.
6. **Cosine only rejects.** Per claim, from the top `top_k`=3 chunks, only those with sim ≥ `t_low` (0.35) go to NLI. If none qualify, the claim scores 0.0 and NLI is skipped. High cosine never auto-accepts.
7. **NLI.** All (chunk, claim) pairs go through one batched call to `cross-encoder/nli-deberta-v3-xsmall` with `text`/`text_pair` input. The entailment probability is the pair's score, and a claim's score is the max over its chunks.
8. **Aggregate.** Score = min over claims, `passed = score >= t_ground` (0.7).
9. **Metadata** holds only counts, indices and scores: `sentences_total`, `sentences_skipped`, `skipped_sentence_indices`, `claims_evaluated`, `ungrounded_count`, `ungrounded_indices`, `details[{sentence_index, best_sim, action, score, chunks_evaluated}]`, `source_chunks_total`, `source_chunks_ranked`, `source_truncated`.

**AggregatedResults (aggregator.py)**

`results` list, `get(name)`, and `transformed_output` (the first non-None `transformed_output`). It makes no decisions.

**Policy (policy.py)**

```python
class Action(str, Enum): ALLOW, MODIFY, REASK, BLOCK

class Decision(BaseModel):
    action: Action
    reason: str
    output: str | None = None          # set for ALLOW / MODIFY only
    timings: dict[str, float] = {}     # schema, parallel, policy, total (+ total_run from run())
    metadata: dict                      # whitelisted summary, see below
```

`PolicyEngine.decide(agg, candidate, retries_left=0)` applies the first matching rule:

1. Any result with `error=True` → BLOCK `internal error in <check>`.
2. Toxicity failed → BLOCK `toxic content detected: incident_id=<8 hex>`. Never re-asked.
3. Schema failed → REASK if `retries_left > 0`, else BLOCK `schema validation failed: …`.
4. Groundedness failed:
   - missing source → BLOCK
   - otherwise → REASK if `retries_left > 0`, else BLOCK `cannot verify against source: …`
5. `transformed_output` present → MODIFY with the redacted output.
6. Otherwise → ALLOW with `output=candidate`.

`Decision.metadata` is `{"checks": {name: {"passed", "error", "score"?, "counts"?}}, "incident_id"?}`, plus `attempts` from `run()`.

- `counts` keeps only numeric and bool metadata values, plus `{category: int}` maps such as PII `found`.
- Parsed values, lists and strings never reach the decision.

**GuardrailEngine.validate() (engine.py)**

1. `SchemaValidator.check(candidate)` always runs first. It is timed as `schema`.
2. Choose what ships: the raw candidate, or the extracted fenced JSON if the schema check stripped a fence. Prose around a fence never ships.
3. Stage 2, in a `ThreadPoolExecutor(max_workers=3)` and timed as `parallel`:
   - **Schema passed:** each check in `config.enabled` runs. PII gets `{parsed, outside_text}`, toxicity gets `{parsed}`, and groundedness gets `{source, parsed}`, where source is the `source` argument or `context["source"]`.
   - **Schema failed:** toxicity and PII (if enabled) still run on the raw text, so toxic output is blocked rather than re-asked.
4. `policy.decide(...)`, timed as `policy`, then `total`.

**GuardrailEngine.run() and ReaskEngine (engine.py, reask.py)**

- `run()` requires `provider=`; without one it raises `ValueError` before any call.
- A fresh `ReaskEngine(provider, max_retries=config.max_retries)` is created per call, so retry state is per request.
- Loop: `provider.complete(prompt)`, then `validate(..., retries_left=reask.retries_left)`. On REASK with budget left, it builds a repair prompt, calls the provider again and validates the new response through the full pipeline.
- Budget: one shared count. With `max_retries=2` there are at most 3 provider calls; after that policy returns BLOCK.
- The previous response is always PII-redacted (`_redact_text`) before it goes into the repair prompt.
- Any provider exception (first call or a re-ask) → BLOCK `internal error in provider: <ExcType>`. Nothing escapes `run()`.
- The final decision carries `metadata["attempts"]` (re-asks used) and `timings["total_run"]`.
- Repair prompt layout: original prompt, then `[VALIDATION FEEDBACK]` (the policy reason), then `[PREVIOUS RESPONSE]`, then a correction instruction, then `[REFERENCE CONTEXT]` with the source when given.
- Feedback, previous response and source are all sanitized: any bracketed header-like token in any case (`[Reference Context]`, `[SYSTEM]`) is rewritten to `(…)`, so untrusted text can't spoof the section headers.

**Providers (providers.py)**

- `LLMProvider.complete(prompt, **kwargs) -> str` is the only method.
- `FakeProvider(responses)` returns scripted responses in FIFO order, records `history` and `call_count`, and raises `RuntimeError` once it runs out.
- Tests and the demo use only `FakeProvider`, so there are no network calls.

**Config (config.py)**

```python
class PolicyConfig(BaseModel):
    enabled: list[str] = ["schema", "pii"]           # schema always runs
    pii_categories: list[str] = ["email", "phone", "aadhaar", "pan", "card", "ip"]  # not wired yet
    groundedness_threshold: float = 0.7
    groundedness_t_low: float = 0.35
    groundedness_on_missing_source: Literal["block", "skip"] = "block"
    toxicity_threshold: float = 0.8
    max_retries: int = 2
```

Plain Pydantic object. The Phase 3 config DSL would parse into this.

**Benchmark (benchmarks/latency_bench.py)**

- `build_engine()` enables PII, toxicity and groundedness. An ML stage is skipped with a reason if its packages or HF-cached models are missing; nothing is downloaded.
- `run_latency_benchmark()` warms up on every payload, then reports end-to-end p50/p90/p95/p99 and p50/p95 for `schema`, `parallel`, `policy` and `total`.
- Run it with `python3 -m benchmarks.latency_bench`.

**Tests**

- The default suite uses fake models and `FakeProvider`.
- `@pytest.mark.slow` tests (real toxic-bert and groundedness models, plus the wall-clock p95 < 200ms check) run only with `--run-slow`.
