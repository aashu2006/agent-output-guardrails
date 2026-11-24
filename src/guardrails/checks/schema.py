import json
import re
from pydantic import BaseModel, ValidationError
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult


def _strip_markdown_fences(text: str) -> str:
    """Strips markdown code fences (e.g. ```json ... ```), including when wrapped with conversational text."""
    m = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _reject_duplicate_keys(pairs):
    """object_pairs_hook for json.loads that raises ValueError on duplicate keys."""
    seen = set()
    d = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key detected: '{key}'")
        seen.add(key)
        d[key] = value
    return d


class SchemaValidator(Guardrail):
    name = "schema"

    def __init__(self, schema: type[BaseModel], strip_markdown: bool = True):
        self.schema = schema
        self.strip_markdown = strip_markdown

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        if self.strip_markdown and isinstance(candidate, str):
            candidate = _strip_markdown_fences(candidate)

        try:
            data = json.loads(candidate, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as e:
            return GuardrailResult(
                guardrail=self.name, passed=False,
                reason=f"invalid JSON: {e.msg} at position {e.pos}",
            )
        except ValueError as e:
            return GuardrailResult(
                guardrail=self.name, passed=False,
                reason=f"invalid JSON: {str(e)}",
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
