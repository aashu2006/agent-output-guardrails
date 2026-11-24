import json
import re
from pydantic import BaseModel, ValidationError
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult

_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)\n?[ \t]*```", flags=re.DOTALL)


class DuplicateKeyError(ValueError):
    """Raised by the json object_pairs_hook on a repeated key. Never carries the key name."""


def extract_fenced_json(text: str) -> tuple[str, str] | None:
    """Returns (fenced_body, text_outside_fence) when the text holds exactly one code fence;
    None otherwise. Used only after a raw json.loads has already failed."""
    if not isinstance(text, str):
        return None
    matches = list(_FENCE_RE.finditer(text))
    if len(matches) != 1:
        return None
    m = matches[0]
    outside = (text[:m.start()] + " " + text[m.end():]).strip()
    return m.group(1).strip(), outside


def _reject_duplicate_keys(pairs):
    """object_pairs_hook for json.loads that raises on duplicate keys (key name is NOT echoed)."""
    seen = set()
    d = {}
    for key, value in pairs:
        if key in seen:
            raise DuplicateKeyError("duplicate key detected")
        seen.add(key)
        d[key] = value
    return d


class SchemaValidator(Guardrail):
    name = "schema"

    def __init__(self, schema: type[BaseModel], strip_markdown: bool = True):
        self.schema = schema
        self.strip_markdown = strip_markdown

    def _fail(self, reason: str) -> GuardrailResult:
        return GuardrailResult(guardrail=self.name, passed=False, reason=reason)

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        fence_stripped = False
        prose_outside = False

        # 1. Raw text first: valid JSON (even with a ``` inside a string value) is never touched
        try:
            data = json.loads(candidate, object_pairs_hook=_reject_duplicate_keys)
        except DuplicateKeyError:
            return self._fail("invalid JSON: duplicate key detected")
        except json.JSONDecodeError as e:
            # 2. Only on parse failure: try a single fenced block that holds the whole JSON payload
            extracted = extract_fenced_json(candidate) if self.strip_markdown else None
            if extracted is None:
                return self._fail(f"invalid JSON: {e.msg} at position {e.pos}")
            body, outside = extracted
            try:
                data = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
            except DuplicateKeyError:
                return self._fail("invalid JSON: duplicate key detected")
            except json.JSONDecodeError as e2:
                return self._fail(f"invalid JSON: {e2.msg} at position {e2.pos}")
            fence_stripped = True
            prose_outside = bool(outside)

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
            metadata={
                "parsed": validated.model_dump(),
                "fence_stripped": fence_stripped,
                "prose_outside_fence": prose_outside,
            },
        )
