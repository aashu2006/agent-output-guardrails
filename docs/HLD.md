### HLD for agent-output-guardrails

**What this is**

A safety layer between an LLM and the app. Every model response gets checked before the user sees it. Bad output gets fixed, cleaned, or blocked. Requirements and metrics live in the PRD. This doc explains how the system is built and why.

**What we are building (and not building)**

We are building a small, provider-agnostic guardrail engine. It catches four problems: broken format, PII leaks, ungrounded claims, and toxic content. Target: under 200ms of our own overhead.

We are NOT building:

- A conversation framework like NeMo. We check one response, not a whole chat.
- Input-side checks (prompt injection etc). That's out of scope for V1.
- A wrapper over Guardrails AI or LLM Guard. The whole point is building it ourselves.
- A "100% safe" system. We measure risk honestly, we don't claim to remove it.

**Core principles**

The whole design comes down to one rule:

**Guardrails detect ->  Results describe -> Policy decides -> Re-ask engine retries**

- A guardrail only inspects and reports. It never calls the LLM, never retries, never decides if the output ships.
- All parallel checks read the same unchanged candidate. PII doesn't edit it, it proposes a cleaned version in its result. Policy applies changes in one controlled place. No race conditions.
- One central re-ask engine owns retries: the budget, the attempt count, building repair requests, and failing closed when retries run out. A re-asked response goes through the full pipeline again. It gets zero trust from the previous attempt.
- The LLM sits behind a provider interface. OpenAI, Anthropic, Gemini, or local, the engine doesn't care.
- Config shapes the engine from outside. It's not a pipeline stage. Observability watches from the side, it never takes part in decisions.

**Runtime flow**

<img width="450" height="550" alt="image" src="https://github.com/user-attachments/assets/ab90c32f-91fe-445d-b073-d70426f17024" />

How a response moves through:

1. App sends a request, LLM responds.
2. Schema validation runs first, alone. A broken response is useless to the other checks, so this is the gate.
3. Schema fails: goes to the re-ask engine. Budget left: retry with the validation error attached. Budget gone: fail closed with a structured error. Never ship malformed output.
4. Schema passes: PII, groundedness, and toxicity run in parallel on the same candidate. Groundedness also gets the source/context as a side input.
5. Every check returns a standard GuardrailResult. The aggregator collects them.
6. The policy engine reads the results and picks one action: allow, modify, re-ask, or block.
7. Re-ask goes through the same central engine, and the new response starts from step 2 again.

Phase note: in Phase 1 there is no re-ask, schema failure just fails closed. Re-ask activates in Phase 2.

**Components**

<img width="450" height="550" alt="image" src="https://github.com/user-attachments/assets/34e7ac0e-5157-4783-a6cf-5e14a0cad531" />

- **Execution coordinator** - runs a request through the stages, handles the parallel dispatch
- **Schema validator** - JSON + Pydantic checks, returns structured errors
- **Guardrail pipeline** - holds the registered checks, runs them concurrently
- **PII / Groundedness / Toxicity** - the actual checks, all implementing one common contract
- **GuardrailResult** - the standard result every check returns: status, score, reason, proposed output, metadata
- **Result aggregator** - collects everything for policy
- **Policy engine** - the only place decisions happen
- **Re-ask engine** - the only place retries happen
- **LLM provider interface** - keeps vendors out of the core
- **Config / DSL** - external settings: which checks run, thresholds, actions, retry limits
- **Observability** - latency, pass/fail rates, scores, retry counts. Watches everything, decides nothing. Never logs raw PII.

**Guardrail contract**

Every check follows the same shape: take the candidate (plus context if needed), return a GuardrailResult. Nothing else.

| Check | Gets | Returns |
|---|---|---|
| Schema | candidate | result with validation errors |
| PII | candidate | result + proposed cleaned output |
| Groundedness | candidate + source | result with score and evidence |
| Toxicity | candidate | result with score |

This is why the system is pluggable: a new guardrail is just a new implementation of this contract. The core engine doesn't change.

**Policy and failure handling**

These are policy defaults, set in config. They are not hard-coded inside the checks.

| Check fails | Default action | If that doesn't work |
|---|---|---|
| Schema | Re-ask with the error (max 2 tries) | Fail closed, structured error |
| PII found | Modify: apply the cleaned output, deliver | n/a, redaction always works |
| Groundedness | Re-ask with source re-attached (max 1 try) | Block: "cannot verify against source" |
| Toxicity | Block immediately, log it. Never retry on unsafe text | - |

When results conflict, precedence is simple: **block > re-ask > modify > allow.** A toxicity block beats everything.

**Performance**

The three parallel checks mean total overhead is roughly max(slowest check), not the sum. The risky one is groundedness: the NLI model runs a transformer pass per sentence pair, which can blow the 200ms budget.

Plan: benchmark the NLI model early (W3). Add an embedding-similarity prefilter so NLI only runs on borderline sentences. If it still doesn't fit, groundedness gets its own declared latency budget instead of quietly breaking the target. We measure with timers per stage, reported as p50/p95. No guessing.

**Decisions we made and why**

- **Central re-ask engine, not validator-owned retries.** Scattered retries mean scattered state, conflicting loops, and no way to enforce a budget.
- **Policy engine instead of hard-coded reactions.** "Any failure = re-ask" is just wrong: PII needs modify, toxicity needs block. Policy makes it configurable.
- **Immutable candidate.** Letting PII edit shared text while other checks read it is a race condition waiting to happen.
- **Build our own engine.** Wrapping an existing framework kills the learning goal.
- **Schema gate before parallel checks.** The other checks need a valid structured candidate to work on.

**Still open (on purpose)**

These get decided in the LLD:

- Exact GuardrailResult status semantics (is "modified" a status, or a pass with a transformed output?)
- Retry budget: one shared per-request budget vs separate budgets per reason
- How repair prompts combine multiple failures at once
- Exact NLI model (picked after the W3 benchmark)
- DSL syntax (YAML vs JSON vs custom). DSL is Phase 3 and the first thing we cut if the schedule slips.

**Later, maybe**

Input-side rails (prompt injection), more guardrails through the same contract, streaming validation, a metrics dashboard.
