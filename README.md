### agent-output-guardrails

A guardrail layer between an LLM and the application. Every response is checked before it reaches the user, and a policy engine decides to allow, modify (redact), re-ask or block it.

**what it does**

Built:
- Schema validation (Pydantic). Parses raw JSON first; a single markdown code fence is stripped only if that fails. Duplicate keys are rejected.
- PII detection and redaction: email (incl. `[at]`/`at ... dot` forms), phone (+91, spaced/dotted/dashed), Aadhaar, PAN, Luhn-checked cards, IPv4. Zero-width characters are ignored for matching. Only counts are recorded, never values.
- Toxicity: `unitary/toxic-bert`, every JSON string value scored in 64-token windows, max over all toxic sublabels.
- Groundedness against a source: sentence split, claim filter, source chunking, cosine rejection below `t_low` (MiniLM), batched DeBERTa NLI (`cross-encoder/nli-deberta-v3-xsmall`) on the best chunks, score = min over claims. Fails closed when no source is given unless configured to skip.
- Policy engine: error > toxicity > schema > groundedness > PII modify > allow. Any check crash blocks.
- Engine coordinator: `engine.validate()` runs schema first, then PII, toxicity and groundedness in parallel.
- Re-ask loop: `engine.run()` with an `LLMProvider` (`FakeProvider` for tests), bounded by `max_retries`, fails closed when the budget is spent or the provider errors.
- Red-team tests (`tests/redteam/`) and a latency benchmark with p50/p95 per stage (`benchmarks/latency_bench.py`).

Phase 3, planned (not built):
- Config DSL (YAML front-end for `PolicyConfig`)
- Deployment packaging
- Observability exports

**run it**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[ml]"          # real models: toxic-bert, MiniLM, DeBERTa NLI (optional)

python3 examples/demo.py        # sections needing models are skipped if [ml] is not installed
python3 -m pytest tests/        # fast suite, no models
python3 -m pytest tests/ --run-slow   # adds real-model tests; first run downloads the models
python3 -m benchmarks.latency_bench --requests 100 --target-p95 200
```

The benchmark runs schema, PII, toxicity and groundedness. ML stages whose packages or models are not installed locally are skipped with a message.

**tech stack**

Python, Pydantic, HuggingFace Transformers, sentence-transformers, PyTorch, pytest

**status**

Phase 1 released as [v0.1.0](https://github.com/aashu2006/agent-output-guardrails/releases/tag/v0.1.0). Phase 2 complete. Tests: 146 passed, 11 skipped (real-model tests) by default; 157 passed with `--run-slow`.
