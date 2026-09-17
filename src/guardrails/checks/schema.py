import json
from pydantic import BaseModel, ValidationError
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult


class SchemaValidator(Guardrail):
    name = "schema"

    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
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
            errors = [f"{'.'.join(str(l) for l in err['loc'])}: {err['msg']}" for err in e.errors()]
            return GuardrailResult(
                guardrail=self.name, passed=False,
                reason="; ".join(errors),
                metadata={"errors": e.errors()},
            )
        return GuardrailResult(
            guardrail=self.name, passed=True,
            metadata={"parsed": validated.model_dump()},
        )