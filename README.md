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

Python · Pydantic · Instructor · local classifier models · pytest

**status** 

Design phase - PRD and HLD completed
