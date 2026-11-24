### agent-output-guardrails

A safety and reliability layer that sits between an LLM and the application, validating every response before it reaches the user.

**what it does**
- JSON schema validation of LLM output (Pydantic)
- PII detection and redaction (Luhn mod-10, regex, DFS tree traversal)
- Groundedness check against source context (Sentence splitting, cosine similarity rejection, DeBERTa NLI verification)
- Toxicity and safety gate (`unitary/toxic-bert` with 64-token sliding window chunking)
- Policy engine: allow / modify / re-ask / block with strict precedence
- Bounded re-ask loop (`ReaskEngine`) with `LLMProvider` abstraction and fail-closed fallback

**tech stack** 

Python · Pydantic · HuggingFace Transformers (`toxic-bert`, `DeBERTa-v3`) · PyTorch · pytest

**status** 

- **Phase 1 complete** and released as [v0.1.0](https://github.com/aashu2006/agent-output-guardrails/releases/tag/v0.1.0).
- **Phase 2 complete**: Real ML ToxicityScanner, 2-tier GroundednessChecker, LLMProvider interface, and bounded ReaskEngine with full `engine.run()` orchestration loop (73 automated unit & integration tests passing + 6 tracked red-team gaps for Phase 3).
- **Phase 3 (Next)**: Red-team bypass patching, latency benchmarking harness (<200ms p95), and production observability exports.
