### agent-output-guardrails

A safety and reliability layer that sits between an LLM and the application, validating every response before it reaches the user.

**what it does**
- JSON schema validation of LLM output (Pydantic)
- PII detection and redaction
- Groundedness check against source context
- Toxicity/safety gate
- Policy engine: allow / modify / re-ask / block
- Bounded re-ask loop with fail-closed fallback

**tech stack** 

Python · Pydantic · local classifier models · pytest

**status** 

Phase 1 complete and released as [v0.1.0](https://github.com/aashu2006/agent-output-guardrails/releases/tag/v0.1.0). Schema validation, PII redaction, policy engine and engine coordinator working (27 tests passing + 6 tracked red-team gaps). [PRD](https://docs.google.com/document/d/1Facjpln9snlBweCfPZkGhbQkcLV4DzSwRFU6imOF0y8/edit?tab=t.0), HLD and LLD completed.