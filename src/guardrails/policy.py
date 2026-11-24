import re
import uuid
from enum import Enum
from pydantic import BaseModel, Field
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
    metadata: dict = Field(default_factory=dict)


_SAFE_KEY = re.compile(r"^[a-z_]{1,32}$")


def _safe_counts(metadata: dict) -> dict:
    """Whitelist: only numeric/bool values (scores, counts, flags) and {category: int} count maps.
    Never parsed values, lists, or strings (which may carry raw text / PII)."""
    out: dict = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not _SAFE_KEY.match(key):
            continue
        if isinstance(value, (bool, int, float)):
            out[key] = value
        elif isinstance(value, dict) and all(
            isinstance(k, str) and _SAFE_KEY.match(k) and isinstance(v, int) and not isinstance(v, bool)
            for k, v in value.items()
        ):
            out[key] = dict(value)
    return out


def _safe_summary(agg: AggregatedResults) -> dict:
    checks = {}
    for r in agg.results:
        summary: dict = {"passed": r.passed, "error": r.error}
        if r.score is not None:
            summary["score"] = r.score
        counts = _safe_counts(r.metadata)
        if counts:
            summary["counts"] = counts
        checks[r.guardrail] = summary
    return {"checks": checks}


class PolicyEngine:
    """Phase 1: fail closed, modify, allow. REASK activates in Phase 2."""

    def decide(
        self,
        agg: AggregatedResults,
        candidate: str,
        retries_left: int = 0,
    ) -> Decision:
        # Whitelisted per-check summary only (never raw check metadata)
        collected_metadata: dict = _safe_summary(agg)

        # 1. koi check crash hua -> fail closed, uska semantic action nahi chalta
        for r in agg.results:
            if r.error:
                return Decision(
                    action=Action.BLOCK,
                    reason=f"internal error in {r.guardrail}",
                    metadata=collected_metadata,
                )

        # 2. toxicity fail -> BLOCK immediately with incident ID (precedence over everything)
        # Toxicity BLOCK > PII MODIFY > ALLOW
        toxicity = agg.get("toxicity")
        if toxicity and not toxicity.passed:
            incident_id = uuid.uuid4().hex[:8]
            collected_metadata["incident_id"] = incident_id
            return Decision(
                action=Action.BLOCK,
                reason=f"toxic content detected: incident_id={incident_id}",
                metadata=collected_metadata,
            )

        # 3. schema fail -> Phase 1 mein fail closed (re-ask Phase 2)
        schema = agg.get("schema")
        if schema and not schema.passed:
            if retries_left > 0:
                return Decision(
                    action=Action.REASK,
                    reason=f"schema validation failed: {schema.reason}",
                    metadata=collected_metadata,
                )
            return Decision(
                action=Action.BLOCK,
                reason=f"schema validation failed: {schema.reason}",
                metadata=collected_metadata,
            )

        # 4. groundedness fail -> Step 7: fail -> RE-ASK with source; budget out -> BLOCK
        groundedness = agg.get("groundedness")
        if groundedness and not groundedness.passed:
            # no source -> re-asking cannot help: fail closed straight away
            if groundedness.metadata.get("missing_source"):
                return Decision(
                    action=Action.BLOCK,
                    reason=f"cannot verify against source: {groundedness.reason}",
                    metadata=collected_metadata,
                )
            if retries_left > 0:
                return Decision(
                    action=Action.REASK,
                    reason=f"groundedness check failed: {groundedness.reason}",
                    metadata=collected_metadata,
                )
            return Decision(
                action=Action.BLOCK,
                reason=f"cannot verify against source: {groundedness.reason}",
                metadata=collected_metadata,
            )

        # 5. PII ne sanitized version propose kiya -> modify
        if agg.transformed_output is not None:
            return Decision(
                action=Action.MODIFY,
                reason="PII redacted",
                output=agg.transformed_output,
                metadata=collected_metadata,
            )

        # 6. sab clean -> allow
        return Decision(
            action=Action.ALLOW,
            reason="all checks passed",
            output=candidate,
            metadata=collected_metadata,
        )