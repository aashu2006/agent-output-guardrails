import json

from pydantic import BaseModel
from guardrails.checks.schema import SchemaValidator


class Invoice(BaseModel):
    customer: str
    amount: int
    paid: bool


v = SchemaValidator(Invoice)


def test_valid_passes_with_parsed():
    r = v.check('{"customer": "Acme", "amount": 5, "paid": true}')
    assert r.passed is True
    assert r.metadata["parsed"]["amount"] == 5


def test_broken_json_fails_with_position():
    r = v.check('{"customer": "Acme", amount: 5}')
    assert r.passed is False
    assert "position" in r.reason


def test_wrong_type_names_the_field():
    r = v.check('{"customer": "Acme", "amount": "lots", "paid": true}')
    assert r.passed is False
    assert "amount" in r.reason


def test_missing_field_fails():
    r = v.check('{"customer": "Acme", "amount": 5}')
    assert r.passed is False
    assert "paid" in r.reason


def test_crash_safety_returns_error_result():
    # None input must NOT raise - base catch-all converts it to error=True
    r = v.check(None)
    assert r.passed is False
    assert r.error is True


def test_error_never_with_passed():
    r = v.check(None)
    assert not (r.passed and r.error)


def test_rejected_input_value_not_in_metadata():
    # wrong-type value carrying a phone number must not leak into metadata
    r = v.check('{"customer": "Acme", "amount": "call 9876543210", "paid": true}')
    assert r.passed is False
    assert "9876543210" not in str(r.metadata)


def test_validation_failure_metadata_is_json_serializable():
    r = v.check('{"customer": "Acme", "amount": "lots", "paid": "maybe"}')
    assert r.passed is False
    json.dumps(r.metadata)  # must not raise


def test_error_entries_have_only_loc_msg_type():
    r = v.check('{"customer": 123, "amount": "lots"}')
    assert r.passed is False
    assert len(r.metadata["errors"]) == 3
    for err in r.metadata["errors"]:
        assert set(err) == {"loc", "msg", "type"}
        assert all(isinstance(part, str) for part in err["loc"])

# --- fence handling (points 13, 14) and duplicate keys (point 15) -------------------------

class Snippet(BaseModel):
    code: str


def test_valid_json_with_fence_inside_string_value_passes():
    payload = json.dumps({"code": "```python\nprint(1)\n```"})
    r = SchemaValidator(Snippet).check(payload)
    assert r.passed is True
    assert r.metadata["parsed"]["code"] == "```python\nprint(1)\n```"
    assert r.metadata["fence_stripped"] is False


def test_whole_text_fence_is_stripped_only_after_parse_fails():
    r = SchemaValidator(Snippet).check('```json\n{"code": "x = 1"}\n```')
    assert r.passed is True
    assert r.metadata["fence_stripped"] is True
    assert r.metadata["prose_outside_fence"] is False


def test_multiple_fences_are_not_guessed():
    r = SchemaValidator(Snippet).check('```json\n{"code": "a"}\n``` and ```json\n{"code": "b"}\n```')
    assert r.passed is False
    assert "invalid JSON" in r.reason


def test_prose_outside_fence_pii_is_scanned_and_never_shipped():
    from guardrails.config import PolicyConfig
    from guardrails.engine import GuardrailEngine
    from guardrails.policy import Action

    class Person(BaseModel):
        name: str

    candidate = 'Email me at ravi@example.com or call 9876543210. ```json {"name":"Ravi"} ```'
    d = GuardrailEngine(Person, config=PolicyConfig(enabled=["schema", "pii"])).validate(candidate)

    assert d.action == Action.MODIFY  # PII found in the raw text
    assert json.loads(d.output) == {"name": "Ravi"}  # only the extracted JSON ships
    dumped = d.model_dump_json()
    assert "ravi@example.com" not in dumped
    assert "9876543210" not in dumped
    assert d.metadata["checks"]["pii"]["counts"]["found"] == {"email": 1, "phone": 1}


def test_clean_fenced_json_with_prose_ships_extracted_json_only():
    from guardrails.config import PolicyConfig
    from guardrails.engine import GuardrailEngine
    from guardrails.policy import Action

    class Person(BaseModel):
        name: str

    candidate = 'Here you go:\n```json\n{"name": "Ravi"}\n```\nHope that helps!'
    d = GuardrailEngine(Person, config=PolicyConfig(enabled=["schema", "pii"])).validate(candidate)

    assert d.action == Action.ALLOW
    assert d.output == '{"name": "Ravi"}'


class Phone(BaseModel):
    phone: str


def test_duplicate_key_reason_does_not_echo_key_name():
    r = SchemaValidator(Phone).check('{"ravi@example.com": "1", "ravi@example.com": "2", "phone": "x"}')
    assert r.passed is False
    assert r.reason == "invalid JSON: duplicate key detected"
    assert "ravi@example.com" not in str(r.model_dump())


def test_duplicate_key_inside_fence_also_rejected_without_key_name():
    r = SchemaValidator(Phone).check('```json\n{"phone": "1", "phone": "2"}\n```')
    assert r.passed is False
    assert r.reason == "invalid JSON: duplicate key detected"
