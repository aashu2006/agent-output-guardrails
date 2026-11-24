import uuid
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

    def decide(
        self,
        agg: AggregatedResults,
        candidate: str,
        retries_left: int = 0,
    ) -> Decision:
        # 1. koi check crash hua -> fail closed, uska semantic action nahi chalta
        for r in agg.results:
            if r.error:
                return Decision(action=Action.BLOCK,
                                reason=f"internal error in {r.guardrail}")

        # 2. toxicity fail -> BLOCK immediately with incident ID (precedence over everything)
        # Toxicity BLOCK > PII MODIFY > ALLOW
        toxicity = agg.get("toxicity")
        if toxicity and not toxicity.passed:
            incident_id = uuid.uuid4().hex[:8]
            return Decision(
                action=Action.BLOCK,
                reason=f"toxic content detected: incident_id={incident_id}",
            )

        # 3. schema fail -> Phase 1 mein fail closed (re-ask Phase 2)
        schema = agg.get("schema")
        if schema and not schema.passed:
            if retries_left > 0:
                return Decision(
                    action=Action.REASK,
                    reason=f"schema validation failed: {schema.reason}",
                )
            return Decision(action=Action.BLOCK,
                            reason=f"schema validation failed: {schema.reason}")

        # 4. groundedness fail -> Step 7: fail -> RE-ASK with source; budget out -> BLOCK
        groundedness = agg.get("groundedness")
        if groundedness and not groundedness.passed:
            if retries_left > 0:
                return Decision(
                    action=Action.REASK,
                    reason=f"groundedness check failed: {groundedness.reason}",
                )
            return Decision(
                action=Action.BLOCK,
                reason=f"cannot verify against source: {groundedness.reason}",
            )

        # 5. PII ne sanitized version propose kiya -> modify
        if agg.transformed_output is not None:
            return Decision(action=Action.MODIFY,
                            reason="PII redacted",
                            output=agg.transformed_output)

        # 6. sab clean -> allow
        return Decision(action=Action.ALLOW,
                        reason="all checks passed",
                        output=candidate)