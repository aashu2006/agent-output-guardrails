import time
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from guardrails.aggregator import AggregatedResults
from guardrails.checks.pii import PIIScanner
from guardrails.checks.schema import SchemaValidator
from guardrails.config import PolicyConfig
from guardrails.policy import Decision, PolicyEngine


class GuardrailEngine:
    """Coordinator: runs the pipeline. Phase 1 scope: validate() only.
    run() (provider call + re-ask loop) comes with Phase 2."""

    def __init__(self, schema: type[BaseModel], config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()
        self.schema_check = SchemaValidator(schema)
        self.pii_check = PIIScanner()
        self.policy = PolicyEngine()

    def validate(self, candidate: str) -> Decision:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        # stage 1: schema gate (sequential - everything depends on it)
        t = time.perf_counter()
        schema_result = self.schema_check.check(candidate)
        timings["schema"] = time.perf_counter() - t
        results = [schema_result]

        # stage 2: remaining checks in parallel on the same immutable candidate
        if schema_result.passed and "pii" in self.config.enabled:
            parallel_checks = [
                (self.pii_check, {"parsed": schema_result.metadata.get("parsed")}),
                # Phase 2: groundedness + toxicity append here, zero engine changes
            ]
            t = time.perf_counter()
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [
                    pool.submit(check.check, candidate, ctx)
                    for check, ctx in parallel_checks
                ]
                results.extend(f.result() for f in futures)
            timings["parallel"] = time.perf_counter() - t

        # stage 3: aggregate + decide
        decision = self.policy.decide(AggregatedResults(results=results), candidate)
        timings["total"] = time.perf_counter() - t0
        decision.timings = timings
        return decision