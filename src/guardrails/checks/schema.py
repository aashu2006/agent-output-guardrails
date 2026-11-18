import json
from pydantic import BaseModel, ValidationError
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult


class SchemaValidator(Guardrail):
    name = "schema"

    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            return GuardrailResult(
                guardrail=self.name, passed=False,
                reason=f"invalid JSON: {e.msg} at position {e.pos}",
            )
        try:
            validated = self.schema.model_validate(data)
        except ValidationError as e:
            # only loc/msg/type: e.errors() also carries the raw rejected "input"
            # (PII leak) and "ctx" (may hold non-JSON-serializable objects)
            safe_errors = [
                {
                    "loc": [str(l) for l in err["loc"]],
                    "msg": err["msg"],
                    "type": err["type"],
                }
                for err in e.errors()
            ]
            reasons = [
                f"{'.'.join(err['loc']) or 'root'}: {err['msg']}" for err in safe_errors
            ]
            return GuardrailResult(
                guardrail=self.name, passed=False,
                reason="; ".join(reasons),
                metadata={"errors": safe_errors},
            )
        return GuardrailResult(
            guardrail=self.name, passed=True,
            metadata={"parsed": validated.model_dump()},
        )