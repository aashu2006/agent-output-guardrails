import json
import pytest
from pydantic import BaseModel
from guardrails.checks.pii import _redact_text, PIIScanner
from guardrails.checks.schema import SchemaValidator

# ----------------------------------------------------------------------------------

#Category 1
## spaced/dashed digits --> paced digits, scanner miss karta hai, toh xfail:

@pytest.mark.xfail(strict=True, reason="known gap: spaced digits bypass \\d{10}")
def test_phone_spaced_digits():
    text, counts = _redact_text("call me at 9 8 7 6 5 4 3 2 1 0")
    assert counts.get("phone", 0) == 1


# Doosra test — dashed digits, same gap:
@pytest.mark.xfail(strict=True, reason="known gap: dashed digits bypass \\d{10}")
def test_phone_dashed_digits():
    text, counts = _redact_text("reach me at 9-8-7-6-5-4-3-2-1-0")
    assert counts.get("phone", 0) == 1

# Teesra test — card spaced digits, yeh scanner PAKAD leta hai kyunki card regex alag hai:
def test_card_spaced_digits_caught():
    text, counts = _redact_text("pay with 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1")
    assert counts.get("card", 0) == 1


# ----------------------------------------------------------------------------------


#Category 2 (worded email)
##Email regex literal @ dhundta hai. [at] likha toh miss ho jaata hai.

@pytest.mark.xfail(strict=True, reason="known gap: [at] bypasses literal @ in regex")
def test_email_bracket_at():
    text, counts = _redact_text("john [at] gmail [dot] com")
    assert counts.get("email", 0) == 1

@pytest.mark.xfail(strict=True, reason="known gap: (at) bypasses literal @ in regex")
def test_email_paren_at():
    text, counts = _redact_text("john(at)gmail.com")
    assert counts.get("email", 0) == 1

#Ek normal email ka test --> ye pakdna chahiye
def test_normal_email_caught():
    text, counts = _redact_text("write to john@gmail.com")
    assert counts.get("email", 0) == 1


# ----------------------------------------------------------------------------------


#Category 3 (fenced JSON)

#Schema
class SampleSchema(BaseModel):
    name: str

validator = SchemaValidator(SampleSchema)

##Fenced JSON block karta hai lekin galat reason se — json.loads crash karta hai:
def test_fenced_json_blocked():
    fenced = '```json\n{"name": "Ravi"}\n```'
    result = validator.check(fenced)
    assert result.passed is False
    assert "invalid JSON" in result.reason


##Ideally fences strip hone chahiye — yeh gap hai:
@pytest.mark.xfail(strict=True, reason="pending: fence stripping in SchemaValidator pre-processing")
def test_fenced_json_should_strip_and_pass():
    fenced = '```json\n{"name": "Ravi"}\n```'
    result = validator.check(fenced)
    assert result.passed is True


#Category 4 (malformed JSON)

##Trailing comma — block hota hai, caught:
def test_trailing_comma_blocked():
    result = validator.check('{"name": "Ravi",}')
    assert result.passed is False


##Single quotes — block hota hai, caught:
def test_single_quotes_blocked():
    result = validator.check("{'name': 'Ravi'}")
    assert result.passed is False

class PhoneSchema(BaseModel):
    phone: str

##Duplicate keys — yeh serious gap hai, SchemaValidator ko reject karna chahiye:
@pytest.mark.xfail(strict=True, reason="known gap: duplicate keys should be rejected by SchemaValidator")
def test_duplicate_keys_gap():
    phone_validator = SchemaValidator(PhoneSchema)
    payload = '{"phone": "9876543210", "phone": "1111111111"}'
    result = phone_validator.check(payload)
    assert result.passed is False


