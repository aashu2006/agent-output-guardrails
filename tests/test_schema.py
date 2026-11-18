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