import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel

from guardrails.aggregator import AggregatedResults
from guardrails.checks.groundedness import GroundednessChecker
from guardrails.checks.pii import PIIScanner
from guardrails.checks.schema import SchemaValidator
from guardrails.checks.toxicity import ToxicityScanner
from guardrails.config import PolicyConfig
from guardrails.policy import Action, Decision, PolicyEngine
from guardrails.providers import LLMProvider
from guardrails.reask import ReaskEngine


class GuardrailEngine:
    """Coordinator: runs the pipeline and bounded re-ask loop."""

    def __init__(
        self,
        schema: type[BaseModel],
        config: PolicyConfig | None = None,
        toxicity_scanner: ToxicityScanner | None = None,
        groundedness_checker: GroundednessChecker | None = None,
        provider: LLMProvider | None = None,
    ):
        self.config = config or PolicyConfig()
        self.provider = provider
        self.schema_check = SchemaValidator(schema)
        self.pii_check = PIIScanner()
        self.toxicity_check = toxicity_scanner or ToxicityScanner(
            threshold=self.config.toxicity_threshold
        )
        self.groundedness_check = groundedness_checker or GroundednessChecker(
            t_low=self.config.groundedness_t_low,
            t_ground=self.config.groundedness_threshold,
        )
        self.policy = PolicyEngine()

    def validate(
        self,
        candidate: str,
        context: dict | None = None,
        source: str | None = None,
        retries_left: int = 0,
    ) -> Decision:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        # stage 1: schema gate (sequential - everything depends on it)
        t = time.perf_counter()
        schema_result = self.schema_check.check(candidate)
        timings["schema"] = time.perf_counter() - t
        results = [schema_result]

        # stage 2: remaining checks in parallel on the same immutable candidate
        if schema_result.passed:
            parallel_checks = []
            parsed_data = schema_result.metadata.get("parsed")

            if "pii" in self.config.enabled:
                parallel_checks.append(
                    (self.pii_check, {"parsed": parsed_data})
                )
            if "toxicity" in self.config.enabled:
                parallel_checks.append(
                    (self.toxicity_check, {"parsed": parsed_data})
                )
            if "groundedness" in self.config.enabled:
                src = source or (context or {}).get("source")
                parallel_checks.append(
                    (self.groundedness_check, {"source": src, "parsed": parsed_data})
                )

            if parallel_checks:
                t = time.perf_counter()
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [
                        pool.submit(check.check, candidate, ctx)
                        for check, ctx in parallel_checks
                    ]
                    results.extend(f.result() for f in futures)
                timings["parallel"] = time.perf_counter() - t

        # stage 3: aggregate + decide
        decision = self.policy.decide(
            AggregatedResults(results=results), candidate, retries_left=retries_left
        )
        timings["total"] = time.perf_counter() - t0
        decision.timings = timings
        return decision

    def run(
        self,
        prompt: str,
        context: dict | None = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> Decision:
        """Full Phase 2 closed loop: Provider call -> Validate -> Reask loop -> Final Decision."""
        if self.provider is None:
            raise ValueError(
                "GuardrailEngine.run() requires an LLMProvider. Pass provider=... to __init__."
            )

        t_run_start = time.perf_counter()
        reask_engine = ReaskEngine(
            provider=self.provider, max_retries=self.config.max_retries
        )

        # 1. Initial generation
        candidate = self.provider.complete(prompt, **kwargs)

        # 2. Validation & bounded repair loop
        while True:
            decision = self.validate(
                candidate=candidate,
                context=context,
                source=source,
                retries_left=reask_engine.retries_left,
            )

            # Check if re-ask should trigger
            if decision.action == Action.REASK and reask_engine.retries_left > 0:
                previous_output = candidate
                candidate = reask_engine.reask(
                    original_prompt=prompt,
                    feedback=decision.reason,
                    source=source,
                    previous_candidate=previous_output,
                    **kwargs,
                )
                continue

            # Record cumulative metrics
            decision.metadata["attempts"] = reask_engine.attempts
            decision.timings["total_run"] = time.perf_counter() - t_run_start
            return decision