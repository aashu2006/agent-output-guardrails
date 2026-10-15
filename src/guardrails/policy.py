from enum import Enum
from pydantic import BaseModel
from guardrails.aggregator import AggregatedResults


class Action(str, Enum):
    ALLOW = "allow"
    MODIFY = "modify"
    REASK = "reask"
    BLOCK = "block"


class Decision(BaseModel):
    action: Action
    reason: str
    output: str | None = None
    timings: dict[str, float] = {}


class PolicyEngine:
    """Phase 1: fail closed, modify, allow. REASK activates in Phase 2."""

    def decide(self, agg: AggregatedResults, candidate: str) -> Decision:
        # 1. koi check crash hua -> fail closed, uska semantic action nahi chalta
        for r in agg.results:
            if r.error:
                return Decision(action=Action.BLOCK,
                                reason=f"internal error in {r.guardrail}")

        # 2. schema fail -> Phase 1 mein fail closed (re-ask Phase 2)
        schema = agg.get("schema")
        if schema and not schema.passed:
            return Decision(action=Action.BLOCK,
                            reason=f"schema validation failed: {schema.reason}")

        # 3. PII ne sanitized version propose kiya -> modify
        if agg.transformed_output is not None:
            return Decision(action=Action.MODIFY,
                            reason="PII redacted",
                            output=agg.transformed_output)

        # 4. sab clean -> allow
        return Decision(action=Action.ALLOW,
                        reason="all checks passed",
                        output=candidate)